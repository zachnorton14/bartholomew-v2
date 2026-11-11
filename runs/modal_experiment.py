"""
Launch experiment-harness base runs on Modal.

This is the Modal counterpart of dev/colab_nanochat_experiment.ipynb: all the
work is done by scripts.experiment, and this file only supplies the machine.
It follows the notebook's base path (prepare, validate, train, optional eval).

    python3 runs/modal_experiment.py check
    python3 runs/modal_experiment.py launch --arm smoke
    python3 runs/modal_experiment.py launch --arm baseline,baseline-s43,lite,values-32m
    python3 runs/modal_experiment.py launch --config configs/base/<any-base-config>.json

Run it with a Python that has the `modal` package (the one behind the `modal` CLI).

How a launch works:

1. The local working tree is uploaded as the code for the run, and the app is
   deployed. A deployed app keeps running after this script exits, so the laptop
   can be closed.
2. A CPU container prepares the token cache once per dataset/tokenizer pair and
   stores it on a Modal Volume. No GPU is billed while data is being prepared.
3. One container with a single H100 is started per config. Each runs the GPU
   preflight, links the shared token cache, then `scripts.experiment train`.

One-time setup, run by a human because it handles credentials:

    modal secret create huggingface-secret HF_TOKEN=...
    modal secret create wandb-secret WANDB_API_KEY=...

These are the names Modal's dashboard gives its Hugging Face and Weights & Biases
secret templates. The HF token needs write access to the config's `artifacts.repo`, and the W&B key
needs access to the config's `wandb.entity`.

Follow a run with `modal app logs bartholomew-experiments`, the Modal dashboard,
or the W&B group named in the config. Stop everything with
`modal app stop bartholomew-experiments`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "bartholomew-experiments"
# Secret name -> the key it must hold. The names are Modal's dashboard templates.
SECRETS = {"huggingface-secret": "HF_TOKEN", "wandb-secret": "WANDB_API_KEY"}
VOLUME_NAME = "bartholomew-nanochat"

REMOTE_REPO = "/root/bartholomew"
VOLUME_MOUNT = "/workspace"
BASE_DIR = f"{VOLUME_MOUNT}/nanochat"

# "H100!" pins a real H100. Plain "H100" may be upgraded to an H200, which would
# make wall-clock comparisons between arms meaningless and fails gpu_preflight.
GPU = "H100!"

# The d24 token cache the shell launchers restore. It is private; when the token
# cannot read it, the cache is built from the dataset instead, as in the notebook.
HOSTED_PRETOK_REPO = "jbduran/clean1930s-d24-r12-ctx4096-fulltok-v1-pretok"

ARM_CONFIG = "configs/base/clean1930s-d12-r12-ctx4096-engram-{arm}-v1.json"
# 20-step runs that exercise the whole pipeline: plain model, and the Engram values path.
SMOKE_CONFIGS = {
    "smoke": "configs/base/clean1930s-d12-r12-ctx4096-modal-smoke-v1.json",
    "smoke-values": "configs/base/clean1930s-d12-r12-ctx4096-modal-smoke-values-v1.json",
}

# This module is imported both on the laptop and inside the container.
LOCAL_REPO = Path(__file__).resolve().parents[1] if modal.is_local() else Path(REMOTE_REPO)

image = (
    # Modal's image builder no longer offers Python 3.10 (the repo's .python-version).
    # 3.12 satisfies requires-python and uv.lock, and is what Colab runs.
    modal.Image.debian_slim(python_version="3.12")
    # git: harness metadata. build-essential: torch.compile/triton need a C compiler.
    .apt_install("git", "build-essential")
    # Exactly the dependencies in uv.lock, with the CUDA 12.8 torch build.
    .uv_sync(str(LOCAL_REPO), extras=["gpu"])
    .env({"NANOCHAT_BASE_DIR": BASE_DIR, "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        str(LOCAL_REPO),
        REMOTE_REPO,
        ignore=[
            ".git", ".venv", ".env", ".claude", ".pytest_cache",
            "**/__pycache__", "**/*.ipynb", "dev",
        ],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
secrets = [modal.Secret.from_name(name, required_keys=[key]) for name, key in SECRETS.items()]

app = modal.App(APP_NAME)
# A second, ephemeral app so `check` can run before the secret exists.
check_app = modal.App(f"{APP_NAME}-check")


# -----------------------------------------------------------------------------
# Helpers shared by the local launcher and the containers


def _load_config(path: str) -> dict:
    with open(LOCAL_REPO / path, encoding="utf-8") as f:
        return json.load(f)


def _cache_key(config: dict) -> str:
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


def _run(cmd: list[str], extra_env: dict | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.update(extra_env or {})
    subprocess.run(cmd, cwd=REMOTE_REPO, env=env, check=True)


# Runs inside the container, from the repo root, as its own process so that the
# harness is imported exactly as `python -m scripts.experiment` would import it.
_PREPARE = r'''
import math, os, shutil, sys
from pathlib import Path

from scripts.experiment import Experiment
from scripts.pretok_think import _existing_cache_satisfies, _tokenizer_fingerprint

config_path, shared, hosted_repo, allow_build = sys.argv[1:5]
shared = Path(shared)
allow_build = allow_build == "1"

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
        f"No usable token cache at {shared}. Start runs through "
        "`runs/modal_experiment.py launch`, which prepares it first."
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
'''


def _prepare(config_path: str, hosted_pretok_repo: str, allow_build: bool) -> None:
    key = _cache_key(_load_config(config_path))
    shared = f"{BASE_DIR}/shared_pretok/{key}"
    _run([
        sys.executable, "-u", "-c", _PREPARE,
        config_path, shared, hosted_pretok_repo, "1" if allow_build else "0",
    ])


# -----------------------------------------------------------------------------
# Remote functions


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=64 * 1024,
    timeout=24 * 60 * 60,
    # Preempted containers are restarted by Modal on the same input; the harness
    # then resumes from its latest complete checkpoint. One retry covers a
    # transient failure without replaying a real bug many times on a GPU.
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0, initial_delay=30.0),
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
)
def train(config_path: str, git_sha: str, eval_mode: str = "none") -> str:
    config = _load_config(config_path)
    experiment_id = config["experiment_id"]
    env = {"NANOCHAT_GIT_COMMIT_SHA": git_sha, "OMP_NUM_THREADS": "1"}
    print(f"=== {experiment_id} | code {git_sha} | {config_path}", flush=True)

    # Fails in seconds if CUDA, the Hopper FP8/FA3 kernels or torch.compile are broken.
    _run([
        sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=1", "-m", "scripts.gpu_preflight", "--expected-gpus", "1",
    ], env)

    _prepare(config_path, hosted_pretok_repo="", allow_build=False)

    _run([
        sys.executable, "-u", "-m", "scripts.experiment", "train",
        "--config", config_path, "--nproc-per-node", "1",
    ], env)

    if eval_mode != "none":
        cmd = [sys.executable, "-u", "-m", "scripts.experiment", "eval", "--config", config_path]
        if eval_mode == "bpb":
            cmd.append("--val-bpb-only")
        _run(cmd, env)

    volume.commit()
    return experiment_id


@app.function(
    image=image,
    cpu=8,
    memory=32 * 1024,
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT: volume},
    secrets=secrets,
)
def launch(
    config_paths: list[str], git_sha: str, eval_mode: str, hosted_pretok_repo: str,
    start_training: bool = True,
) -> list[str]:
    """Prepare each distinct token cache once on CPU, then start one GPU run per config."""
    prepared = set()
    for config_path in config_paths:
        key = _cache_key(_load_config(config_path))
        if key in prepared:
            continue
        print(f"=== Preparing token cache {key} using {config_path}", flush=True)
        _prepare(config_path, hosted_pretok_repo, allow_build=True)
        prepared.add(key)
    # GPU containers mount the Volume as it is when they start.
    volume.commit()

    call_ids = []
    if not start_training:
        print("Prepare only: no GPU runs started.", flush=True)
        return call_ids
    for config_path in config_paths:
        call = train.spawn(config_path, git_sha, eval_mode)
        call_ids.append(call.object_id)
        print(f"Started {config_path}: function call {call.object_id}", flush=True)
    return call_ids


@check_app.function(image=image, timeout=15 * 60, volumes={VOLUME_MOUNT: volume})
def check(config_paths: list[str]) -> dict:
    """CPU-only, no credentials: is the image usable and does the Volume behave?"""
    import shutil
    import torch

    report = {
        "python": sys.version.split()[0],
        # str(): torch.__version__ is a TorchVersion, which the laptop cannot unpickle.
        "torch": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "cc": shutil.which("cc"),
    }

    # The harness and trainer import cleanly from the uploaded tree.
    _run([sys.executable, "-m", "scripts.experiment", "--help"])
    _run([sys.executable, "-m", "scripts.base_train", "--help"])

    # Every config passes the harness's own validation and yields a train command.
    probe = (
        "import sys\n"
        "from scripts.experiment import Experiment\n"
        "for path in sys.argv[1:]:\n"
        "    e = Experiment(path)\n"
        "    e.validate_config()\n"
        "    print('config ok:', e.experiment_id)\n"
    )
    _run([sys.executable, "-c", probe, *config_paths])
    report["configs_valid"] = len(config_paths)

    # Symlinks on the Volume decide whether arms share one token cache or copy it.
    probe_dir = Path(BASE_DIR) / "_check"
    shutil.rmtree(probe_dir, ignore_errors=True)
    (probe_dir / "target").mkdir(parents=True)
    (probe_dir / "target" / "meta.json").write_text("{}")
    try:
        (probe_dir / "link").symlink_to(probe_dir / "target", target_is_directory=True)
        report["volume_symlinks"] = (probe_dir / "link" / "meta.json").read_text() == "{}"
    except OSError as exc:
        report["volume_symlinks"] = f"unsupported: {exc}"
    shutil.rmtree(probe_dir, ignore_errors=True)
    volume.commit()
    return report


# -----------------------------------------------------------------------------
# Local launcher


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=LOCAL_REPO, text=True).strip()


def _resolve_configs(arms: str | None, configs: list[str] | None) -> list[str]:
    paths = []
    for arm in (arms or "").split(","):
        arm = arm.strip()
        if arm:
            paths.append(SMOKE_CONFIGS.get(arm) or ARM_CONFIG.format(arm=arm))
    paths.extend(configs or [])
    if not paths:
        raise SystemExit("Give --arm or --config.")
    resolved = []
    for path in paths:
        full = (LOCAL_REPO / path).resolve()
        if not full.is_file():
            raise SystemExit(f"Config not found: {path}")
        relative = full.relative_to(LOCAL_REPO).as_posix()
        config = _load_config(relative)
        if config.get("stage", "base") != "base":
            raise SystemExit(f"{relative}: only base configs are supported here.")
        if config.get("branch") or config.get("mixture_schedule"):
            raise SystemExit(f"{relative}: branch and mixture configs are not supported here yet.")
        resolved.append(relative)
    if len(set(resolved)) != len(resolved):
        raise SystemExit("The same config was given twice.")
    return resolved


def _cmd_launch(args: argparse.Namespace) -> None:
    configs = _resolve_configs(args.arm, args.config)

    sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    if dirty:
        if not args.allow_dirty:
            raise SystemExit(
                "The working tree has uncommitted changes, and the working tree is what "
                "gets uploaded. Commit first so the run's code is recoverable, or pass "
                "--allow-dirty.\n" + dirty
            )
        sha += "-dirty"

    print(f"Code: {sha}")
    print(f"GPU:  one {GPU} per config")
    for path in configs:
        config = _load_config(path)
        wandb = config.get("wandb", {})
        print(f"  {config['experiment_id']}  (W&B group {wandb.get('group', '-')})")
    if args.dry_run:
        print("Dry run: nothing deployed or started.")
        return

    hosted = "" if args.no_hosted_pretok else HOSTED_PRETOK_REPO
    with modal.enable_output():
        app.deploy()
    call = modal.Function.from_name(APP_NAME, "launch").spawn(
        configs, sha, args.eval, hosted, not args.prepare_only
    )
    print(f"\nStarted. Preparation call: {call.object_id}")
    if args.prepare_only:
        print("Waiting for preparation to finish (CPU only, no GPU runs will start)...")
        call.get()
        print(f"Preparation finished. Details: modal app logs {APP_NAME}")
        return
    print(f"Logs:  modal app logs {APP_NAME}")
    print(f"Stop:  modal app stop {APP_NAME}")
    print("The GPU runs start once the token cache is ready. This terminal can be closed.")


def _cmd_check(args: argparse.Namespace) -> None:
    configs = sorted(
        path.relative_to(LOCAL_REPO).as_posix()
        for path in (LOCAL_REPO / "configs" / "base").glob("clean1930s-d12-r12-ctx4096-engram-*.json")
    ) + sorted(SMOKE_CONFIGS.values())
    with modal.enable_output(), check_app.run():
        report = check.remote(configs)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment-harness base configs on Modal")
    commands = parser.add_subparsers(dest="command", required=True)

    launch_parser = commands.add_parser("launch", help="deploy the code and start runs")
    launch_parser.add_argument("--arm", help="comma-separated Engram ablation arms, e.g. smoke or baseline,lite")
    launch_parser.add_argument("--config", action="append", help="path to a base config (repeatable)")
    launch_parser.add_argument(
        "--eval", choices=("none", "bpb", "full"), default="none",
        help="run `scripts.experiment eval` after training (default: none; training already logs val bpb)",
    )
    launch_parser.add_argument("--no-hosted-pretok", action="store_true", help="always build the token cache from the dataset")
    launch_parser.add_argument("--prepare-only", action="store_true", help="prepare the token cache on CPU and stop; start no GPU runs")
    launch_parser.add_argument("--allow-dirty", action="store_true", help="launch with uncommitted changes")
    launch_parser.add_argument("--dry-run", action="store_true", help="print what would run and exit")
    launch_parser.set_defaults(func=_cmd_launch)

    check_parser = commands.add_parser("check", help="CPU-only check of the image, configs and Volume (no credentials)")
    check_parser.set_defaults(func=_cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
