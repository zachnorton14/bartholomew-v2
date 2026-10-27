# Document and artifact contracts plan

Future module: `common/contracts.py`. Status: documentation only.

## Document contract

Define a versioned record carrying stable `doc_id`, current text, native source
identity, original input reference/hash, normalized metadata, provenance, and
retained refinement/temporal evidence. Track current text hashes separately from
source identity so edits do not create unrelated document IDs.

Required versus optional fields and score units must be explicit. Each adapter
declares available metadata; each stage declares its requirements. Do not repair a
missing scientific measurement by inventing a default value.

## Artifact and stage-result contract

Specify versioned manifests containing input references, schema/capabilities,
resolved settings, stage fingerprint, required output files/checksums, and completion
state. Results also carry statistics, bounded audit samples, and failure information.

Keep processed/accepted/rejected/failed counts distinct. Audit rows join by document
ID. Final packing additionally records exported shard/row positions.

Internal metadata-rich records and final text-only Parquet are different contracts.
The metadata transition issue in [TO DO](<../TO DO.md>) must be resolved before
adopting these contracts in production.
