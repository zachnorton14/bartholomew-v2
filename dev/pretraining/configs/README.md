# Configuration layer

Status: documentation only. No executable JSON recipes or schema validator exist yet.

Future configs will use JSON, following the training harness's conventions. They
describe the recipe identity, data sources, policy, stage inputs/parameters, and
execution settings. The worker resolves and validates them before processing data.

| Document / folder | Responsibility |
|---|---|
| [schema.md](schema.md) | Shared recipe contract, identity rules, defaults, and validation. |
| [standard/](standard/README.md) | Future recipes for the broad historical corpus. |
| [midtraining/](midtraining/README.md) | Future recipes for the STEM corpus. |

Keep data-selection/filtering settings separate from throughput settings such as
worker count and upload batching. Throughput settings can be excluded from result
identity only after deterministic behavior is established.

The illustrative config in the [proposal](../standard/pipeline_proposal.md) is a
design example, not a runnable recipe. Required revisions, score units, missing-field
behavior, and cutoff semantics must be settled before a production config is added.
