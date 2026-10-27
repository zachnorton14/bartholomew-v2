# Standard recipe plans

Status: planning only. Future files here will be JSON recipe configs with distinct IDs.

Recipes will describe the broad book corpus from acquisition through pretokenization,
using the shared stages rather than separate v1/v2/v3 implementations.

Planned recipe concerns:

- Source dataset revision, text/metadata field mapping, and selection scope.
- Agreed date/language/OCR/tokenizability gates and explicit missing-field handling.
- Refinement options, exact deduplication, temporal rule version, and prior mode.
- Split policy, packing contract, and pinned training tokenizer.
- A bounded development recipe and a separately identified full-corpus recipe.

Current thresholds are historical references, not approved new defaults. See the
[standard guide](../../standard/README.md), [archived code](../../standard/old/README.md),
and [shared config schema](../schema.md).
