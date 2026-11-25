#!/usr/bin/env bash
# Submit Bartholomew jobs to Hazel's Slurm scheduler. Run on a login node.
#
#   runs/hazel/submit.sh check
#   runs/hazel/submit.sh launch --arm smoke
#   runs/hazel/submit.sh launch --arm baseline,baseline-s43,lite,values-32m
#   runs/hazel/submit.sh launch --config configs/base/<any-base-config>.json [--eval bpb]
#   runs/hazel/submit.sh status
#
# launch options (same meaning as runs/modal_experiment.py launch):
#   --arm A,B          Engram ablation arms, or smoke / smoke-values
#   --config PATH      a base config (repeatable)
#   --eval MODE        none (default) | bpb | full, run after training
#   --no-hosted-pretok always build the token cache from the dataset
#   --prepare-only     submit only the CPU token-cache jobs
#   --short            train on the partner H100s under the 2-hour short_gpu QOS
#                      (max 2 H100s per user); for smoke tests, which start much sooner
#   --allow-dirty      launch with uncommitted changes
#   --dry-run          print what would be submitted
#
# Each launch freezes the code: the commit (or, with --allow-dirty, the working
# tree) is copied to $HAZEL_SHARE/code/<sha>, and the jobs run from that copy, so
# editing or pulling the repo while jobs wait in the queue does not change them.
# One CPU prepare job runs per distinct token cache; each config then gets one
# H100 job that starts only after its cache job succeeds.
#
# Extra sbatch flags: HAZEL_PREP_SBATCH_ARGS and HAZEL_TRAIN_SBATCH_ARGS, e.g.
#   HAZEL_TRAIN_SBATCH_ARGS="--time=3-00:00:00 --mem=128G" runs/hazel/submit.sh launch --arm values-128m

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$BART_REPO"

LOGS="$HAZEL_SHARE/logs"
mkdir -p "$LOGS"

need_env() {
    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
        echo "No environment at $UV_PROJECT_ENVIRONMENT. Run: bash runs/hazel/setup.sh" >&2
        exit 1
    fi
    if [ -z "${HF_TOKEN:-}" ] || [ -z "${WANDB_API_KEY:-}" ]; then
        echo "HF_TOKEN and WANDB_API_KEY must be set in $BART_SECRETS (see runs/hazel/README.md)" >&2
        exit 1
    fi
}

cmd_check() {
    need_env
    sbatch --parsable --export=ALL --output="$LOGS/%x-%j.out" "$HERE/check.sbatch" \
        | { read -r id; echo "Submitted check job $id"; echo "Log: $LOGS/bart-check-$id.out"; }
}

cmd_status() {
    squeue -u "$USER" -o "%.10i %.14j %.9P %.2t %.11M %.11l %.6D %R"
    echo
    echo "Logs: $LOGS"
}

arm_config() {
    case "$1" in
        smoke)        echo "configs/base/clean1930s-d12-r12-ctx4096-modal-smoke-v1.json" ;;
        smoke-values) echo "configs/base/clean1930s-d12-r12-ctx4096-modal-smoke-values-v1.json" ;;
        *)            echo "configs/base/clean1930s-d12-r12-ctx4096-engram-$1-v1.json" ;;
    esac
}

cmd_launch() {
    local arms="" eval_mode=none prepare_only=0 allow_dirty=0 dry_run=0 short=0
    local configs=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --arm) arms="$2"; shift 2 ;;
            --config) configs+=("$2"); shift 2 ;;
            --eval) eval_mode="$2"; shift 2 ;;
            --no-hosted-pretok) export NO_HOSTED_PRETOK=1; shift ;;
            --prepare-only) prepare_only=1; shift ;;
            --short) short=1; shift ;;
            --allow-dirty) allow_dirty=1; shift ;;
            --dry-run) dry_run=1; shift ;;
            *) echo "Unknown option $1" >&2; exit 2 ;;
        esac
    done
    case "$eval_mode" in none|bpb|full) ;; *) echo "--eval must be none, bpb or full" >&2; exit 2 ;; esac

    local arm arm_configs=()
    IFS=',' read -ra arm_list <<< "$arms"
    for arm in "${arm_list[@]}"; do
        arm="${arm// /}"
        if [ -n "$arm" ]; then arm_configs+=("$(arm_config "$arm")"); fi
    done
    configs=(${arm_configs[@]+"${arm_configs[@]}"} ${configs[@]+"${configs[@]}"})
    [ ${#configs[@]} -gt 0 ] || { echo "Give --arm or --config." >&2; exit 2; }
    need_env

    local sha
    sha="$(git rev-parse HEAD)"
    if [ -n "$(git status --porcelain)" ]; then
        if [ $allow_dirty != 1 ]; then
            echo "The working tree has uncommitted changes. Commit first so the run's code is" >&2
            echo "recoverable, or pass --allow-dirty." >&2
            git status --short >&2
            exit 1
        fi
        sha="$sha-dirty-$(date +%Y%m%d%H%M%S)"
    fi
    export NANOCHAT_GIT_COMMIT_SHA="$sha"

    # Same checks as modal_experiment.py: base stage only, no branch/mixture configs.
    # Prints "<config path>\t<cache key>\t<experiment id>" per config.
    local plan
    plan="$(bart_python - "${configs[@]}" <<'PY'
import json, sys
from pathlib import Path
from scripts.prepare_token_cache import cache_key
seen = set()
for path in sys.argv[1:]:
    p = Path(path)
    if not p.is_file():
        sys.exit(f"Config not found: {path}")
    rel = p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    if rel in seen:
        sys.exit("The same config was given twice.")
    seen.add(rel)
    config = json.loads(p.read_text())
    if config.get("stage", "base") != "base":
        sys.exit(f"{rel}: only base configs are supported here.")
    if config.get("branch") or config.get("mixture_schedule"):
        sys.exit(f"{rel}: branch and mixture configs are not supported here yet.")
    print(f"{rel}\t{cache_key(config)}\t{config['experiment_id']}")
PY
)"

    echo "Code:  $sha"
    echo "Group: $HAZEL_GROUP (account for the jobs)"
    local train_args="${HAZEL_TRAIN_SBATCH_ARGS:-}"
    if [ $short = 1 ]; then
        train_args="--partition=gpu_partners --qos=short_gpu --time=02:00:00 $train_args"
        echo "GPU:   one H100 per config, short_gpu QOS (2-hour limit)"
    else
        echo "GPU:   one H100 per config"
    fi
    printf '%s\n' "$plan" | while IFS=$'\t' read -r path key id; do echo "  $id  ($path)"; done
    if [ $dry_run = 1 ]; then
        echo "Dry run: nothing submitted."
        return
    fi

    # Freeze the code the jobs will run.
    local code_dir="$HAZEL_SHARE/code/$sha"
    if [ ! -d "$code_dir" ]; then
        mkdir -p "$code_dir.tmp"
        if [ $allow_dirty = 1 ]; then
            rsync -a --exclude .git --exclude .venv --exclude '__pycache__' --exclude .env \
                --exclude .claude --exclude dev ./ "$code_dir.tmp/"
        else
            git archive HEAD | tar -x -C "$code_dir.tmp"
        fi
        mv "$code_dir.tmp" "$code_dir"
    fi
    echo "Frozen code: $code_dir"

    declare -A prep_job=()
    local path key id
    while IFS=$'\t' read -r path key id; do
        [ -n "${prep_job[$key]:-}" ] && continue
        # shellcheck disable=SC2086
        prep_job[$key]="$(sbatch --parsable --export=ALL --job-name="bart-prep-$id" \
            --output="$LOGS/%x-%j.out" ${HAZEL_PREP_SBATCH_ARGS:-} \
            "$HERE/prepare.sbatch" "$code_dir" "$path" "$NANOCHAT_BASE_DIR/shared_pretok/$key")"
        echo "Prepare job ${prep_job[$key]}: token cache $key"
    done <<< "$plan"

    if [ $prepare_only = 1 ]; then
        echo "Prepare only: no GPU jobs submitted."
    else
        while IFS=$'\t' read -r path key id; do
            local job
            # shellcheck disable=SC2086
            job="$(sbatch --parsable --export=ALL --job-name="bart-$id" \
                --dependency="afterok:${prep_job[$key]}" --kill-on-invalid-dep=yes \
                --output="$LOGS/%x-%j.out" $train_args \
                "$HERE/train.sbatch" "$code_dir" "$path" "$eval_mode")"
            echo "Train job $job: $id (after ${prep_job[$key]})"
        done <<< "$plan"
    fi
    echo
    echo "Queue: runs/hazel/submit.sh status    Logs: $LOGS/<job-name>-<jobid>.out"
    echo "Cancel everything: scancel -u $USER"
}

case "${1:-}" in
    check) shift; cmd_check "$@" ;;
    launch) shift; cmd_launch "$@" ;;
    status) shift; cmd_status "$@" ;;
    *) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
