# Internet Archive adapter plan

Future module: `sources/internet_archive.py`. Status: documentation only.

## Input and responsibilities

- Accept declared collections/queries, subject/date selection, acquisition limits, and bounded concurrency/retry settings.
- Capture candidate identifiers and the selected text-file identity before expensive processing.
- Download text with explicit handling for missing files, transient failures, and exhausted retries.
- Preserve source metadata, acquisition timestamps, hashes, and normalized source IDs.

## Output and constraints

Emit normalized documents plus an acquisition manifest with fetched, missing,
failed, and skipped counts. A live query alone is insufficient to reproduce a run;
reuse a captured candidate/file manifest for reproducible continuation.

This adapter acquires material. The corpus quality recommendation remains a recipe
and filtering-stage responsibility. Reference only:
[old extraction notebook](../midtraining/old/midtraining_extraction.ipynb).
