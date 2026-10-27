# Worker plan

Status: documentation only. Future module: `dev/pretraining/worker.py`.

The worker will read one recipe config and coordinate reusable source adapters and
stages. It should remain short; filtering, acquisition, and durable I/O belong in
their respective layers.

## Planned responsibilities

1. Validate config keys, unique stage IDs, registered stage/adapter types, and stage dependencies.
2. Resolve source revisions and external artifact manifests and save a fully resolved specification.
3. Display the execution plan, including inputs, intended outputs, parameters, and cache reuse.
4. Execute stages in dependency order, using compatible completed artifacts when available.
5. Record attempts, progress, summaries, failures, and durable completion through shared state/storage helpers.
6. Handle a separate publish action after local artifact validation.

The initial interface will distinguish validation/planning, local execution,
resuming a compatible attempt, inspecting results, and publishing. Exact command
names and flags are deferred; none exist at this milestone.

## Boundaries

- No source-specific parsing or filter expressions in the worker.
- No arbitrary Python/shell commands embedded in configs.
- No silent fallback from failed source discovery to an empty successful run.
- Changing an input mid-pipeline must be declared through a pinned artifact reference and recorded in lineage.
- Execution can start at a later stage only when its declared inputs satisfy the required contracts.

See [config schema](configs/schema.md), [stage contracts](stages/README.md),
[shared state](common/state.md), and the [proposal](standard/pipeline_proposal.md).
