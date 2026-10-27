# Rule asset plans

Status: documentation only. No fresh rule lists, regex assets, or executable rule
loaders have been added.

This layer will hold versioned, reviewable policy assets consumed by stages rather
than duplicating large embedded lists in multiple notebooks or recipe configs.

The first planned asset family is [era rules](era.md): banned terms, allow-list,
confidence tiers, strip-only patterns, and curation evidence. Rule content and
changes must be identifiable in stage fingerprints and audit outputs.

Legacy rule implementations remain inside the standard/midtraining `old/` trees.
They provide reference evidence, not automatically approved defaults for the new
pipeline. Reviewing terms/tiers remains PT #9 in [TO DO](<../TO DO.md>).
