# Institutional Books adapter plan

Future module: `sources/institutional.py`. Status: documentation only.

## Input and responsibilities

- Read a pinned [Institutional Books dataset](https://huggingface.co/datasets/institutional/institutional-books-1.0) revision using an explicitly selected text field and file set.
- Normalize publication-date evidence, language measurements, OCR scores, tokenizability, subject tags, and duplicate identifiers into documented fields/units.
- Preserve native identifiers and a raw source reference for stable document IDs.
- Capture provenance and schema/capability information needed by `screen`.

## Output and constraints

Emit normalized documents and acquisition statistics, without applying undocumented
quality-policy defaults. Missing or malformed measurements must be distinguishable
from genuine zero scores. Date interpretation must preserve the original evidence.

Reference only: [old Institutional Books builder](../standard/old/repackage_institutional_books.py).
The fresh adapter does not import or run it yet.
