# Era stage

Future module: `stages/era.py`. Status: documentation only.

## Input

Deduplicated records, retained refinement evidence, and an explicitly versioned
temporal rule set consistent with the recipe's cutoff policy.

## Planned work

- Match agreed banned terms, phrases, and format evidence in documented scan windows.
- Apply reviewed tier/corroboration rules for whole-document rejection.
- Keep strip-only signals and ambiguous historical meanings distinguishable from decisive temporal evidence.
- Record matched terms, reasons, rule identity, kept/rejected counts, and bounded inspection samples.

## Output and boundary

Produce surviving records and temporal decision/audit artifacts. This stage enforces
historical content policy; duplicate/split contamination checks remain elsewhere.
Rule-list improvements are separate work from wiring the stage.

See [rule plans](../rules/era.md) and the
[old matching implementation](../standard/old/scripts/filter_lib.py).
