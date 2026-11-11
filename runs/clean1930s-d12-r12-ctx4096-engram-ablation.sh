#!/usr/bin/env bash

# Run one arm of the d12 r12 4K Engram ablation (issue #11) on one GPU.
# See dev/ENGRAM_ABLATION_PLAN.md for the arms, the runbook and what to record.
#
# Wave 1: baseline, baseline-s43, lite, values-32m
# Wave 2 (only for a variant that beats the seed gap): values-8m, values-128m, values-32m-bigram

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:-}"

case "$ARM" in
    baseline|baseline-s43|lite|values-32m|values-8m|values-32m-bigram)
        DEFAULT_MIN_FREE_GIB=80
        ;;
    values-128m)
        # Stretch arm: 33 GB table, final checkpoint only (save_every=-1).
        DEFAULT_MIN_FREE_GIB=150
        ;;
    *)
        echo "Usage: $0 {baseline|baseline-s43|lite|values-32m|values-8m|values-128m|values-32m-bigram}" >&2
        exit 2
        ;;
esac
CONFIG="clean1930s-d12-r12-ctx4096-engram-$ARM-v1.json"

export BASE_CONFIG_PATH="$SCRIPT_DIR/../configs/base/$CONFIG"
# The engram values table is single-process only (base_train asserts it); all arms use one GPU.
export NPROC_PER_NODE=1
export ALLOW_SINGLE_GPU=1
export REQUIRE_FULL_NVLINK="${REQUIRE_FULL_NVLINK:-0}"
export MIN_FREE_GIB="${MIN_FREE_GIB:-$DEFAULT_MIN_FREE_GIB}"
if [ -d /workspace ]; then
    DEFAULT_BASE_DIR=/workspace/nanochat
else
    DEFAULT_BASE_DIR="$HOME/.cache/nanochat"
fi
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${NANOCHAT_BASE_DIR:-$DEFAULT_BASE_DIR}/torchinductor-d12-ctx4096-engram-$ARM}"

exec bash "$SCRIPT_DIR/clean1930s-d24-r12.sh"
