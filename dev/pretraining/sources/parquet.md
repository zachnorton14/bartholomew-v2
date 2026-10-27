# Parquet and existing-artifact adapter plan

Future module: `sources/parquet.py`. Status: documentation only.

## Input and responsibilities

- Load an explicitly selected local or pinned remote shard set, excluding metadata/report files from training text discovery.
- Map declared columns to the common document contract while preserving available identity and audit metadata.
- Validate compatible manifests when importing an earlier pipeline stage's result.
- Expose declared split membership and source capabilities instead of inferring them silently from filenames.

## Output and constraints

Emit normalized documents or a validated stage-artifact reference. A text-only export
cannot be treated as metadata-rich raw books: stages requiring absent dates/OCR
scores must reject the input or use an explicitly configured policy.

This adapter is the entry point for changing datasets midway through a recipe.
See [config schema](../configs/schema.md) and [contracts](../common/contracts.md).
