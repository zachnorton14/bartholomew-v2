#!/usr/bin/env bash
# Submitted by `runs/hazel/submit.sh launch`: one base config on one H100, the Hazel
# counterpart of modal_experiment.py's train(). Args: CODE_DIR CONFIG EVAL_MODE
# Resubmitting the same config resumes from its latest complete checkpoint.
#SBATCH --job-name=bart-train
#SBATCH --partition=gpu
# h100 specifically: gpu_preflight requires one, and it keeps wall-clock
# comparable with the Modal H100 runs.
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# The gpu QOS allows up to 4 days.
#SBATCH --time=2-00:00:00

set -euo pipefail
source "$BART_REPO/runs/hazel/env.sh"
CODE_DIR="$1"; CONFIG="$2"; EVAL_MODE="${3:-none}"
cd "$CODE_DIR"
export OMP_NUM_THREADS=1
echo "=== $CONFIG | code $NANOCHAT_GIT_COMMIT_SHA | node $(hostname) | job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

run() { echo "+ $*"; "$@"; }

# Fails in seconds if CUDA, the Hopper FP8/FA3 kernels or torch.compile are broken.
run bart_python -u -m torch.distributed.run --standalone --nproc-per-node=1 \
    -m scripts.gpu_preflight --expected-gpus 1

# Links the cache the prepare job built; never tokenizes on the GPU.
SHARED="$NANOCHAT_BASE_DIR/shared_pretok/$(bart_python -m scripts.prepare_token_cache key "$CONFIG")"
run bart_python -u -m scripts.prepare_token_cache prepare "$CONFIG" "$SHARED"

run bart_python -u -m scripts.experiment train --config "$CONFIG" --nproc-per-node 1

case "$EVAL_MODE" in
    none) ;;
    bpb)  run bart_python -u -m scripts.experiment eval --config "$CONFIG" --val-bpb-only ;;
    full) run bart_python -u -m scripts.experiment eval --config "$CONFIG" ;;
    *)    echo "Unknown eval mode $EVAL_MODE" >&2; exit 2 ;;
esac
