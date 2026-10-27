# Identity and fingerprint plan

Future module: `common/fingerprints.py`. Status: documentation only.

## Planned behavior

- Separate a recipe's unique `pipeline_id` from generated execution `attempt_id` values.
- Reject reuse of a recipe identity with changed computational settings.
- Fingerprint each stage from its exact input identity, resolved parameters/policy, relevant helper dependencies, rules, fitted statistics, tokenizer identity, and seeds as applicable.
- Preserve the full repository revision in provenance while tracking relevant stage dependencies for reuse.
- Exclude names/descriptions from computational result identity. Exclude throughput settings only when they are proven not to change output.

## Expected reuse

A changed era rule set reuses gathering/refinement but invalidates `era` and affected
downstream outputs. A changed training tokenizer reuses packing and reruns `tokenize`.
A prior-calibration cache is reusable only for compatible corpus, split, tokenizer,
and sampling identities.

Cache lookup must validate completed manifests and required artifacts rather than
trust a familiar filename or directory. No cache is implemented at this milestone.
