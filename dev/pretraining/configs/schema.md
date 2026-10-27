# Recipe schema plan

Status: documentation only. Future validation/resolution helper: `common/config.py`.

## Planned fields

| Field | Meaning |
|---|---|
| `schema_version` | Version of the recipe contract. |
| `pipeline_id` | Unique recipe identity; changed computational settings require a new identity. |
| `name`, `description` | Human-readable purpose and context; excluded from computational stage fingerprints. |
| `seed` | Deterministic sampling/order defaults, resolved into the affected stages. |
| `policy` | Shared historical-cutoff and other explicitly agreed policy, with units/semantics. |
| `sources` | Named registered adapters with acquisition settings, immutable revisions or captured source manifests, and required metadata behavior. |
| `stages` | Ordered entries with unique instance ID, registered type, explicit input reference, and typed parameters. |
| `execution` | Resource/throughput settings such as workers, staging location, and upload batching. |
| Artifact destination | A selected durable destination/root; subordinate artifact paths are derived from recipe, stage, and fingerprint identity. Exact field shape remains to be reviewed. |

## Planned validation

- Reject unknown keys, unsupported versions, duplicate IDs, invalid types/ranges, and dependencies on unavailable or later stages.
- Resolve defaults into the saved run specification; do not let future default changes alter an existing recipe silently.
- Distinguish a previous stage's output from an external artifact reference. External inputs require schema, lineage, and completion validation.
- Require explicit cutoff inclusivity, score units, missing-field behavior, and prior mode where relevant.
- Treat `prior: off` as a recorded pass-through rather than a missing output.
- Keep credentials outside configs and do not save secret environment values into resolved specifications.
- Record output-affecting rule versions, tokenizer identities, and fitted-statistic inputs.

See [document/artifact contracts](../common/contracts.md) and
[fingerprint planning](../common/fingerprints.md).
