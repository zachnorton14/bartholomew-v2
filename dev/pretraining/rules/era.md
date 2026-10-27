# Era rule contract plan

Status: documentation only. Future asset format and filenames remain to be agreed.

## Planned contents

- An explicit rule-set ID/version and historical-cutoff applicability.
- Normalized terms/phrases with confidence tiers and corroboration behavior.
- Allow-list and strip-only/format signals with distinct meanings.
- Source/evidence notes for curation decisions and ambiguous historical senses.
- A stable checksum and representation for inclusion in resolved configs/manifests.

The rule loader will validate allowed tiers, duplicate/conflicting entries, and
policy compatibility. The `era` stage will consume this reviewed contract and
report its exact identity with every result.

Reference only: [standard banned-list construction](../standard/old/scripts/build_list.py)
and [midtraining variant](../midtraining/old/scripts/build_list.py). Choosing one or
reconciling them is future work, not performed in this structural milestone.
