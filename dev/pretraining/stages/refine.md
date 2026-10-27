# Refine stage

Future module: `stages/refine.py`. Status: documentation only.

## Input

Screened document records retaining original source references and metadata.

## Planned work

- Trim agreed header/footer, library, digitization, copyright/reprint, and other boilerplate patterns.
- Normalize selected OCR/Unicode artifacts, whitespace, and wrapped/hyphenated lines.
- Apply documented guardrails against excessive removal and recheck cleaned length/structural quality.
- Preserve document identity while recording changed-text hashes, removed evidence, and flags for ambiguous cases.

## Output and boundary

Produce cleaned records with enough retained evidence for later temporal decisions.
Do not equate harmless scanner boilerplate with proof that an entire book is modern.
Whole-book anachronism rejection belongs to `era`.

Historical references: [standard footer library](../standard/old/scripts/footer_lib.py)
and [midtraining structural cleaning](../midtraining/old/scripts/structural_clean.py).
