# Prior stage

Future module: `stages/prior.py`. Status: documentation only.

## Input

Era-filtered records with stable split membership, a declared scoring tokenizer,
calibration settings, and a selected `off`, `report`, or `filter` mode.

## Planned work

- Treat `off` as an explicitly recorded pass-through with no fitted/scored output.
- Fit/reuse token-frequency priors only against compatible training-input, tokenizer, seed, and calibration identities.
- In `report` mode, produce scores/diagnostics without rejecting documents.
- In `filter` mode, apply reviewed thresholds and record rejection reasons and retention statistics.
- Preserve split membership and keep the scoring tokenizer separately identified from the training tokenizer.

## Output and boundary

Produce pass-through/scored/filtered records, calibration artifacts where applicable,
and a summary that clearly identifies the mode. The current method is a sampled
token-frequency outlier signal, not a language-model quality verdict.

Whether to keep, change, or remove this filter remains an open investigation in
[TO DO](<../TO DO.md>). Reference only: [old prior cell](../standard/old/cells/cell5_prior.py).
