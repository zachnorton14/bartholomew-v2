# Pack stage

Future module: `stages/pack.py`. Status: documentation only.

## Input

Final selected metadata-rich records with stable document IDs and already-decided
train/validation membership. `prior: off` still provides a compatible recorded input.

## Planned work

- Export deterministic text-only Parquet shards using the agreed packing/compression/row-group settings.
- Preserve whole-book rows and established split membership.
- Write document-ID to exported shard/row mappings separately from training data.
- Emit explicit train/validation file lists, checksums, counts, manifests, and summaries.
- Represent any current loader ordering conventions explicitly, including validation placement when required.

## Output and boundary

Produce a validated local text dataset and audit artifacts ready for inspection,
publication, or `tokenize`. Publishing remains separate. Do not recompute splits
or lose metadata alignment when repacking.

Reference contract: [old shard guide](../standard/old/DATASET_PIPELINE.md).
Its adoption or revision must be reviewed during contract implementation.
