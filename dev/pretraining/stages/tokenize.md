# Tokenize stage

Future module: `stages/tokenize.py`. Status: documentation only.

## Input

A completed `pack` artifact and a pinned training-tokenizer artifact, plus token
limits, token-shard size, batching/thread settings, dtype, and boundary-token policy.

## Planned work

- Pretokenize final text while preserving declared train/validation membership and deterministic ordering.
- Validate vocabulary/token IDs against the output dtype, including any added boundary tokens.
- Record tokenizer/corpus fingerprints, BOS/document-boundary behavior, per-split/file counts, and requested versus actual coverage.
- Resume compatible completed token shards and reject incompatible partial caches.
- Produce a token-cache contract compatible with the existing training loader, without editing that loader during this milestone.

## Output and boundary

Produce token streams/shards, metadata, manifests, and summaries. A changed training
tokenizer should reuse filtering/packing artifacts and rebuild tokenization only.
Tokenizer fitting remains external initially and must use the final training split.

Reference only: [existing pretokenizer](../../../scripts/pretok_think.py) and
[token loader](../../../nanochat/pretok_dataloader.py). Future integration stays
within the authorized `dev/` scope unless the user explicitly changes it.
