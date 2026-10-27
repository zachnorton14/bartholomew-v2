#!/usr/bin/env python3
"""
Drive the pre-1930 SFT curriculum sweep (C0-C5) and nanochat-default baseline on
top of nanochat's experiment harness.

The six curriculums are ordinary sft configs that live with the rest of nanochat's configs:
`<nanochat>/configs/sft/pre1930-curriculum-c<N>.json`. Each carries an embedded curriculum
spec under data.curriculum (see tasks/synth-pre1930.py for the format) and runs standalone
through the harness exactly like any other config, e.g.

  python -m scripts.experiment train --config configs/sft/pre1930-curriculum-c0.json \
         --parent-experiment-id <PRE1930_BASE_ID> --parent-step <STEP>

This wrapper just adds the three things the harness does not do itself: an offline pool-size
check, a back-to-back loop with the base injected once, and a cross-run ranking report. Run
it under nanochat's venv (needs pandas + huggingface_hub).

Examples:
  python run_curricula.py validate
  python run_curricula.py run --base-experiment-id think-d12-r11.25 --base-step 2362
  python run_curricula.py run --only c1 --dry-run --base-experiment-id X --base-step 0
  python run_curricula.py run --only c0,c2,c3,c4,c5,default \
         --base-experiment-id think-d12-r11.25 --base-step 2362
  python run_curricula.py compare --base-experiment-id think-d12-r11.25
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
ANALYSIS_DIR = REPO / "analysis"
PARQUET = ANALYSIS_DIR / "dataset_metadata.parquet"
CURRICULA = ["c0", "c1", "c2", "c3", "c4", "c5"]
RUN_CONFIGS = {
    **{name: f"pre1930-curriculum-{name}.json" for name in CURRICULA},
    "default": "nanochat-default-v1.json",
}
RUNS = list(RUN_CONFIGS)
# Keep in sync with tasks/synth-pre1930.py (_HOLDOUT_FRAC): pools shrink by the eval holdout.
HOLDOUT_FRAC = 0.015
DEFAULT_ARTIFACT_REPO = "jbduran/think.nano"


def _nanochat_dir(args):
    return Path(args.nanochat_dir or os.environ.get("NANOCHAT_DIR")
                or (Path.home() / "git" / "nanochat"))


def _config_path(nanochat, name):
    return nanochat / "configs" / "sft" / RUN_CONFIGS[name]


def _load_config(nanochat, name):
    with open(_config_path(nanochat, name), encoding="utf-8") as f:
        return json.load(f)


def _experiment_root():
    root = os.environ.get("NANOCHAT_EXPERIMENT_ROOT")
    if root:
        return Path(root)
    base = os.environ.get("NANOCHAT_BASE_DIR") or str(Path.home() / ".cache" / "nanochat")
    return Path(base) / "experiments"


def _artifact_repo(nanochat, name="c0"):
    """Return the model artifact repository configured for an SFT run."""
    config = _load_config(nanochat, name)
    return config.get("artifacts", {}).get(
        "repo",
        config.get("storage", {}).get("hf_model_repo", DEFAULT_ARTIFACT_REPO),
    )


def _upload_artifact(nanochat, repo, local_path, path_in_repo, message):
    """Upload one local SFT artifact to the configured model repo."""
    from huggingface_hub import HfApi

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"missing artifact: {local_path}")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model",
        commit_message=message,
    )
    print(f"Uploaded {local_path} -> {repo}/{path_in_repo}")


def _metrics_path(base, name, nanochat):
    exp_id = f"{base}-{_load_config(nanochat, name)['experiment_suffix']}"
    return (
        _experiment_root()
        / base
        / "sft"
        / exp_id
        / "checkpoints"
        / "eval_metrics.json"
    ), exp_id


def _upload_metrics(nanochat, base, name, repo):
    local_path, exp_id = _metrics_path(base, name, nanochat)
    remote_path = (
        f"experiments/{base}/sft/{exp_id}/checkpoints/eval_metrics.json"
    )
    _upload_artifact(
        nanochat,
        repo,
        local_path,
        remote_path,
        f"Upload SFT eval metrics for {exp_id}",
    )


# ---------------------------------------------------------------------------
# validate: pool-size check


def _spec_route_counts(spec):
    """Yield (route, count, threshold) for every graded route a spec draws from.
    Skips authentic (ungraded) and staged full-pool draws (count=None)."""
    default_thr = spec.get("threshold_default", 90)
    if spec.get("mode") == "staged":
        for stage in spec.get("stages", []):
            thr = stage.get("threshold", default_thr)
            for route in stage.get("routes", []):
                yield route, stage.get("count"), thr           # count None -> full pool
            if stage.get("calibration_qa"):
                yield "calibration_qa", None, thr
        return
    for route, cfg in spec.get("routes", {}).items():
        yield route, cfg.get("count"), cfg.get("threshold", default_thr)
    cal = spec.get("calibration_qa")
    if cal is not None:
        yield "calibration_qa", cal.get("count"), cal.get("threshold", default_thr)


def cmd_validate(args):
    import pandas as pd

    if not PARQUET.exists():
        sys.exit(f"missing {PARQUET}; run from the sft-extractor repo with the analysis parquet present")
    nanochat = _nanochat_dir(args)
    df = pd.read_parquet(PARQUET, columns=["route", "score"])
    ok = True
    for name in CURRICULA:
        spec = _load_config(nanochat, name)["data"]["curriculum"]
        print(f"\n=== {name.upper()} ({spec.get('name')}, mode={spec.get('mode')}) ===")
        for route, count, thr in _spec_route_counts(spec):
            pool = int(((df["route"] == route) & (df["score"] >= thr)).sum())
            usable = max(0, pool - math.ceil(HOLDOUT_FRAC * pool))
            if count is None:
                print(f"  {route:<20} >= {thr:<3} full pool ~{usable:,} (train after holdout)")
                continue
            flag = "OK "
            if count > usable:
                flag, ok = "!! ", False
            print(f"  {flag}{route:<20} >= {thr:<3} want {count:,} of ~{usable:,} usable (pool {pool:,})")
    print("\nAll counts fit." if ok else "\nSOME COUNTS EXCEED AVAILABILITY (marked !!) — lower them or relax the threshold.")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# run: back-to-back via scripts.experiment (base injected on the CLI)


def cmd_run(args):
    if not args.base_experiment_id:
        sys.exit("--base-experiment-id is required (the pre-1930 base run to fine-tune from)")
    names = [n.strip() for n in args.only.split(",")] if args.only else RUNS
    nanochat = _nanochat_dir(args)
    for name in names:
        if name not in RUN_CONFIGS:
            sys.exit(f"unknown SFT run {name!r}; choose from {RUNS}")
        config = _config_path(nanochat, name)
        cmd = [
            sys.executable, "-u", "-m", "scripts.experiment", "train",
            "--config", str(config.relative_to(nanochat)),
            "--parent-experiment-id", args.base_experiment_id,
            "--parent-step", str(args.base_step),
        ]
        print(f"\n########## {name.upper()}: {' '.join(cmd)} (cwd={nanochat})")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=nanochat)
        if result.returncode != 0:
            sys.exit(f"{name} failed (exit {result.returncode}); fix and re-run "
                     f"(finished runs are skipped by scripts.experiment).")
        _upload_metrics(nanochat, args.base_experiment_id, name,
                        _artifact_repo(nanochat, name))
    print("\nAll requested SFT runs finished. Run `compare` to rank them.")


# ---------------------------------------------------------------------------
# compare: gather eval_metrics.json + rank


def cmd_compare(args):
    nanochat = _nanochat_dir(args)
    root = _experiment_root()
    base = args.base_experiment_id
    rows = []
    for name in RUNS:
        exp_id = f"{base}-{_load_config(nanochat, name)['experiment_suffix']}"
        p = root / base / "sft" / exp_id / "checkpoints" / "eval_metrics.json"
        if not p.exists():
            print(f"  (skip {name}: no eval_metrics.json at {p})")
            continue
        with open(p, encoding="utf-8") as f:
            rows.append((name, json.load(f)))
    if not rows:
        sys.exit("no eval_metrics.json found yet — has any run finished?")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / "curriculum_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({name: m for name, m in rows}, f, indent=2)

    repo = _artifact_repo(nanochat, rows[0][0])
    for name, _ in rows:
        _upload_metrics(nanochat, base, name, repo)
    _upload_artifact(
        nanochat,
        repo,
        out,
        f"experiments/{base}/sft/curriculum_results.json",
        f"Upload curriculum comparison for {base}",
    )

    ranked = sorted(rows, key=lambda r: r[1].get("val_bpb", float("inf")))
    best_bpb = ranked[0][0]
    chatcore_rows = [
        row for row in rows
        if row[1].get("chatcore", {}).get("chatcore_metric") is not None
    ]
    best_cc = max(
        chatcore_rows,
        key=lambda r: r[1]["chatcore"]["chatcore_metric"],
    )[0] if chatcore_rows else None

    print(f"\n{'run':<18}{'val_bpb':>10}{'min_bpb':>10}{'chatcore':>10}{'register':>10}   note")
    print("-" * 76)
    for name, m in ranked:
        cc = m.get("chatcore", {}).get("chatcore_metric")
        note = []
        if name == best_bpb:
            note.append("best val_bpb")
        if name == best_cc:
            note.append("best chatcore")
        print(f"{name:<18}{m.get('val_bpb', float('nan')):>10.4f}"
              f"{m.get('min_val_bpb', float('nan')):>10.4f}"
              f"{(cc if cc is not None else float('nan')):>10.4f}"
              f"{'—':>10}   {', '.join(note)}")

    winner = ranked[0][1]
    if winner.get("per_route_bpb"):
        print(f"\nper-route val bpb — {best_bpb} (val_bpb winner):")
        for route, v in sorted(winner["per_route_bpb"].items(), key=lambda kv: kv[1]):
            print(f"  {route:<20}{v:>8.4f}")
    print(f"\nWrote {out}")
    if any(name == "default" for name, _ in rows):
        print("Caution: nanochat-default val_bpb uses its own SmolTalk/MMLU/GSM8K "
              "validation mixture, not the held-out pre-1930 curriculum set.")
    print("Note: 'register' column reserved for a future anachronism/register eval.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nanochat-dir", default="", help="path to the nanochat repo (default: $NANOCHAT_DIR or ~/git/nanochat)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="offline pool-size check against the parquet")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="run the SFT configs back-to-back via scripts.experiment")
    r.add_argument("--base-experiment-id", required=True, help="pre-1930 base run to fine-tune from")
    r.add_argument("--base-step", type=int, required=True, help="base checkpoint step")
    r.add_argument("--only", default="", help="comma list, e.g. c0,c2,default (default: all)")
    r.add_argument("--dry-run", action="store_true", help="print the harness commands only")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="gather eval_metrics.json and rank the runs")
    c.add_argument("--base-experiment-id", required=True)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
