#!/usr/bin/env bash
# Push-button embed run on the cloud GPU. Explodes pairs.jsonl -> questions, embeds
# them with Talkie, and ships the vectors off the ephemeral instance disk.
#
#   PAIRS=/workspace/pairs.jsonl bash encoder/run_embed.sh            # full run
#   SMOKE=1 PAIRS=/workspace/pairs.jsonl bash encoder/run_embed.sh    # first 100 rows only
#   HF_REPO=you/talkie-embeddings HF_TOKEN=hf_xxx PAIRS=... bash encoder/run_embed.sh
set -euo pipefail

PAIRS="${PAIRS:?set PAIRS=path/to/pairs.jsonl}"   # your generators' output, uploaded to the box
MODEL="${MODEL:-talkie-1930-13b-base}"
OUT_DIR="${OUT_DIR:-/workspace/out}"
SMOKE="${SMOKE:-0}"                                # 1 = first 100 rows only (A100 setup test)
HF_REPO="${HF_REPO:-}"                             # optional: upload results here
mkdir -p "$OUT_DIR"
cd /workspace/encoder/.. 2>/dev/null || cd "$(dirname "$0")/.."   # repo root (encoder/ importable)

INPUT="$PAIRS"
if [ "$SMOKE" = "1" ]; then
  INPUT="$OUT_DIR/pairs.smoke.jsonl"
  head -n 100 "$PAIRS" > "$INPUT"
  echo ">> SMOKE: first $(wc -l < "$INPUT") rows"
fi

tag="$(basename "$INPUT" .jsonl)"
python -m encoder.prepare_embed "$INPUT" --out "$OUT_DIR/$tag"
python -m encoder.talkie_encoder --model "$MODEL" \
  --texts "$OUT_DIR/$tag.txt" --out "$OUT_DIR/$tag.emb.npy" --max-len 64

echo ">> shapes:"
python - "$OUT_DIR/$tag" <<'PY'
import numpy as np, sys
stem = sys.argv[1]
e = np.load(stem + ".emb.npy")
n = sum(1 for _ in open(stem + ".keys.jsonl"))
print(f"  emb {e.shape}   keys {n}   aligned: {e.shape[0]==n}")
PY

if [ -n "$HF_REPO" ]; then
  huggingface-cli upload "$HF_REPO" "$OUT_DIR" "embeddings/$tag" --repo-type dataset
  echo ">> uploaded to hf://$HF_REPO/embeddings/$tag"
else
  echo "Results in $OUT_DIR — scp them off BEFORE destroying the instance:"
  echo "  scp -r root@<instance-ip>:$OUT_DIR ./out"
fi
