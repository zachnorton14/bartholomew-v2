#!/usr/bin/env bash
# Submitted by `runs/hazel/submit.sh check`. Answers, on a real H100 node:
# can the job reach Hugging Face and W&B, does the driver support CUDA 12.8 torch,
# and do FP8, FA3 and torch.compile work (scripts.gpu_preflight).
#SBATCH --job-name=bart-check
# The 2-hour short_gpu QOS on the partner nodes starts far sooner than the shared
# gpu partition, which has only 4 H100s.
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -uo pipefail
source "$BART_REPO/runs/hazel/env.sh"
cd "$BART_REPO"
status=0

echo "== node $(hostname), job $SLURM_JOB_ID, group $HAZEL_GROUP"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "C compiler for triton: $(command -v cc || command -v gcc || echo MISSING)"

echo "== outbound internet from the compute node"
for url in https://huggingface.co https://api.wandb.ai https://pypi.org; do
    if code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 "$url" 2>&1); then
        echo "  $url -> HTTP $code"
    else
        echo "  $url -> UNREACHABLE ($code)"
        status=1
    fi
done

echo "== credentials"
bart_python - <<'PY' || status=1
import os
from huggingface_hub import whoami
print("  HF user:", whoami(token=os.environ["HF_TOKEN"])["name"])
import wandb
print("  W&B login ok:", wandb.login(key=os.environ["WANDB_API_KEY"], verify=True))
PY

echo "== GPU preflight (FP8, FA3, torch.compile, NCCL)"
OMP_NUM_THREADS=1 bart_python -u -m torch.distributed.run --standalone --nproc-per-node=1 \
    -m scripts.gpu_preflight --expected-gpus 1 || status=1

echo "== done: $([ $status -eq 0 ] && echo ALL OK || echo SOMETHING FAILED, see above)"
exit $status
