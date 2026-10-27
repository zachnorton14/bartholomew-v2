# Gather stage

Future module: `stages/gather.py`. Status: documentation only.

## Input

Named source-adapter definitions with pinned revisions or captured acquisition
manifests, selection settings, field mappings, and bounded acquisition limits.

## Planned work

- Invoke registered source adapters and normalize their output into the shared document contract.
- Avoid repeatedly acquiring identical native source identifiers.
- Capture source selection, raw input identity/hashes, available metadata, and acquisition outcomes.
- Record source-query restrictions separately from later filtering decisions.

## Output and boundary

Produce traceable normalized records and an acquisition manifest/summary. Do not
discard identity or audit metadata to make training-ready text shards here.
General eligibility policy belongs to `screen`, and cross-source content grouping
belongs to `dedupe`.

See [source adapters](../sources/README.md) and [contracts](../common/contracts.md).
