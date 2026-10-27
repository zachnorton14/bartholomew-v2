# Midtraining recipe plans

Status: planning only. Future files here will be JSON configs with distinct IDs.

Midtraining recipes will reuse the shared worker/stages with STEM-oriented sources
and explicitly chosen source-quality settings. Gathering new data and filtering
that data remain distinct responsibilities.

Planned recipe concerns:

- Source adapters for original-corpus reuse and Internet Archive acquisition.
- STEM subject selection, source provenance, and intentional reuse versus accidental duplication.
- OCR score units/missing-field behavior and treatment of legitimate equations, tables, and symbols.
- Shared refinement and temporal-rule implementations with recipe-specific parameters.
- Independent evaluation of prior filtering on technical text, preserving the common stage contract.
- Split, packing, and training-tokenizer references.

Mixture construction and anneal training are preserved in the archive but are not
additional implemented stages in this milestone. Their future integration should
be decided separately. See the [midtraining guide](../../midtraining/README.md),
[archived files](../../midtraining/old/README.md), and [shared schema](../schema.md).
