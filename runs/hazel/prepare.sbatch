#!/usr/bin/env bash
# Submitted by `runs/hazel/submit.sh launch`: builds (or restores, or just links) the
# shared token cache for one dataset/tokenizer pair on a CPU node, so no GPU hours
# are spent tokenizing. Args: CODE_DIR CONFIG SHARED_DIR
#SBATCH --job-name=bart-prepare
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
source "$BART_REPO/runs/hazel/env.sh"
CODE_DIR="$1"; CONFIG="$2"; SHARED="$3"
cd "$CODE_DIR"
echo "=== prepare $CONFIG -> $SHARED | code $NANOCHAT_GIT_COMMIT_SHA | node $(hostname)"

args=(prepare "$CONFIG" "$SHARED" --allow-build)
if [ "${NO_HOSTED_PRETOK:-0}" != 1 ]; then
    args+=(--hosted-repo "$HOSTED_PRETOK_REPO")
fi
bart_python -u -m scripts.prepare_token_cache "${args[@]}"
