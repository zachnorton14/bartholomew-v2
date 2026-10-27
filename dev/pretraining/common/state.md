# Execution state and resume plan

Future module: `common/state.py`. Status: documentation only.

## Planned behavior

- Track recipe identity, resolved config, attempt identity, stage status, and each stage's restartable work units.
- Distinguish pending, running, complete, failed, and intentionally skipped/pass-through stages.
- Write checkpoints atomically and promote work to complete only when all required artifacts are durable.
- Validate input/settings fingerprints before reuse; explain incompatible state rather than silently mixing results.
- Ensure retries do not duplicate records or skip partially written outputs.
- Persist deterministic sampling/order progress where results depend on it.

For remote commits, completion follows the successful commit of related files. For
local outputs, completion follows validation and final placement. Disk cleanup must
not remove artifacts still referenced by pending operations.

Exact state schemas and checkpoint granularity remain to be implemented and reviewed.
See [storage](storage.md), [contracts](contracts.md), and [validation plans](../validation/README.md).
