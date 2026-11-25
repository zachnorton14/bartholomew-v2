"""Prepare one experiment's token cache in a shared location, then link it in.

Shared by the cluster launchers (runs/modal_experiment.py, runs/hazel/). Configs
with the same dataset and tokenizer share one cache, so a sweep tokenizes once.

    python -m scripts.prepare_token_cache key CONFIG
    python -m scripts.prepare_token_cache prepare CONFIG SHARED_DIR [--hosted-repo REPO] [--allow-build]

`prepare` with --allow-build restores the hosted cache if it is usable, or builds
it from the dataset (download parquet, then tokenize). Without --allow-build it
only links an existing cache, which is what GPU jobs do so they never tokenize.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path


def cache_key(config: dict) -> str:
    """Configs with the same key can share one token cache: same text, same tokenizer.

    The cache's size is not part of the key. The harness re-checks that a cache is
    large enough and was made with the same tokenizer before reusing it.
    """
    dataset = config["dataset"]
    tokenizer = config["tokenizer"]
    tokenizer_id = tokenizer.get("source_experiment_id") or config["experiment_id"]
    raw = "{}@{}@val{}@{}".format(
        dataset["repo"], dataset.get("revision", "main"),
        dataset.get("validation_shard", "default"), tokenizer_id,
    )
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def prepare(config_path: str, shared: Path, hosted_repo: str, allow_build: bool) -> None:
    # Imported here so `key` works without the training dependencies.
    from scripts.experiment import Experiment
    from scripts.pretok_think import _existing_cache_satisfies, _tokenizer_fingerprint

    experiment = Experiment(config_path)
    experiment.initialize()
    experiment.prepare_tokenizer()

    def target_tokens():
        # Mirrors Experiment.prepare_pretokenized: the cache size the harness will ask for.
        pretok = experiment.config.get("pretokenize", {})
        training = experiment.config["training"]
        if pretok.get("target_tokens") is not None:
            return int(pretok["target_tokens"])
        slack = float(pretok.get("slack", 1.03))
        if training.get("target_tokens") is not None:
            return math.ceil(int(training["target_tokens"]) * slack)
        if training.get("target_param_data_ratio") is not None:
            batch = int(training.get("total_batch_size", 524_288))
            horizon = math.floor(
                float(training["target_param_data_ratio"]) * int(training["scaling_params"]) / batch
            ) * batch
            return math.ceil(horizon * slack)
        return -1

    def cache_ready(directory):
        pretok = experiment.config.get("pretokenize", {})
        return _existing_cache_satisfies(
            str(directory), target_tokens(), int(pretok.get("val_tokens", 20_971_520)),
            _tokenizer_fingerprint(str(experiment.tokenizer_dir)),
        )

    shared.parent.mkdir(parents=True, exist_ok=True)
    local = experiment.pretok_dir

    def local_is_own_cache():
        return not local.is_symlink() and cache_ready(local)

    def link_local_to_shared():
        """Point this experiment's pretok/ at the shared cache, as the shell launchers do."""
        if local.is_symlink():
            if local.resolve() != shared.resolve():
                raise RuntimeError(f"{local} points to {local.resolve()}, not {shared}")
            return
        if local.exists():
            shutil.rmtree(local)
        try:
            local.symlink_to(shared, target_is_directory=True)
        except OSError as exc:
            print(f"Symlink failed ({exc}); copying the shared cache instead", flush=True)
            shutil.copytree(shared, local)

    if allow_build and hosted_repo and not cache_ready(shared) and not local_is_own_cache():
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
        try:
            snapshot_download(
                repo_id=hosted_repo, repo_type="dataset", revision="main",
                local_dir=str(shared), token=os.environ["HF_TOKEN"],
            )
            print(f"Restored hosted token cache {hosted_repo}", flush=True)
        except HfHubHTTPError as exc:
            cause = exc.__cause__ or exc
            reason = " ".join(str(cause).split())[:300]
            print(
                f"Hosted token cache {hosted_repo} could not be downloaded "
                f"({type(cause).__name__}: {reason}). Building the cache from the dataset instead.",
                flush=True,
            )

    if local_is_own_cache():
        print(f"Keeping the existing experiment-local token cache at {local}", flush=True)
        shared = local
    elif cache_ready(shared):
        link_local_to_shared()
    elif not allow_build:
        raise RuntimeError(
            f"No usable token cache at {shared}. Run the launcher's prepare step first "
            "(runs/modal_experiment.py launch, or runs/hazel/launch.sh)."
        )
    else:
        # The notebook's "Prepare Base Inputs" path: download parquet, then tokenize.
        # pretok_think deletes and recreates its output directory, so build in a real
        # directory and publish it to the shared location afterwards.
        if local.is_symlink():
            local.unlink()
        local.mkdir(exist_ok=True)
        experiment.prepare_dataset()
        experiment.prepare_pretokenized()
        shutil.rmtree(shared, ignore_errors=True)
        shutil.move(str(local), str(shared))
        link_local_to_shared()

    # Reuses a ready cache without rereading it, then runs the harness's own
    # no-wrap and provenance validation.
    experiment.prepare_pretokenized()
    print(f"Token cache ready for {experiment.experiment_id}: {shared}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    key_parser = commands.add_parser("key", help="print the shared cache key for a config")
    key_parser.add_argument("config")

    prepare_parser = commands.add_parser("prepare", help="restore, build or link the token cache")
    prepare_parser.add_argument("config")
    prepare_parser.add_argument("shared", help="shared cache directory for this config's key")
    prepare_parser.add_argument("--hosted-repo", default="", help="HF dataset repo holding a prebuilt cache")
    prepare_parser.add_argument("--allow-build", action="store_true", help="download and tokenize if no cache is usable")

    args = parser.parse_args()
    if args.command == "key":
        with open(args.config, encoding="utf-8") as f:
            print(cache_key(json.load(f)))
    else:
        prepare(args.config, Path(args.shared), args.hosted_repo, args.allow_build)


if __name__ == "__main__":
    main()
