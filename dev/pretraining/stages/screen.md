# Screen stage

Future module: `stages/screen.py`. Status: documentation only.

## Input

Normalized gathered records with declared date, language, subject, quality, and
fragment metadata capabilities.

## Planned work

- Apply explicit publication-date/inclusivity and undated-record policy.
- Apply agreed language/subject selection and available OCR/tokenizability gates.
- Handle missing measurements and score units explicitly.
- Reject pathological fragments while keeping the distinction between raw source measurements and cleaned-text checks.
- Record a stable rejection reason and counts for each gate.

## Output and boundary

Produce surviving metadata-rich records and audit/statistics artifacts. Cheap
eligibility checks should run before expensive scoring where possible. A separate
OCR computation stage may be proposed later if resource needs justify it; it is
not implemented or assumed in this initial scaffold.

Threshold decisions remain open under [TO DO](<../TO DO.md>). Reference only:
[old source filter](../standard/old/repackage_institutional_books.py).
