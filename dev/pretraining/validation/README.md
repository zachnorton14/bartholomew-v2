# Verification layer plan

Status: documentation only. No test code, fixtures, or data-processing runs are added.

Future checks should focus on meaningful contracts and failure recovery:

- Config validation, unknown keys, explicit cutoff semantics, units, and input compatibility.
- Stable document identity and metadata joins through filtering/repacking, including the reported v1 -> v2 metadata problem and the v2 -> v3 transition.
- Deterministic representative/split selection across resumed runs and worker counts.
- Fingerprint reuse/invalidation when inputs, rules, prior calibration, or tokenizers change.
- Partial output/commit failures and recovery without duplicate or skipped records.
- Small representative refinement/temporal examples, including ambiguous historical terms and excessive-strip guardrails.
- Packed shard schema, explicit split/file manifests, and document-position audit mapping.
- Pretokenized dtype bounds, boundary handling, counts, split preservation, and incomplete-cache detection.

Begin with small local deterministic records and a bounded historical sample.
External network/model/GPU runs should be explicit integration checks, not required
for every local check. Compare the wrapped legacy behavior before changing policy.

The current milestone verifies only document structure, preserved archive contents,
local links, and scope. Implementation and execution wait for user review.
