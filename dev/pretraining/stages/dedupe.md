# Dedupe stage

Future module: `stages/dedupe.py`. Status: documentation only.

## Input

Refined records with native identifiers, current text hashes, and available
source-provided duplicate relationships.

## Planned work

- Start with exact duplicate grouping using agreed stable keys and cleaned-text hashes.
- Select representatives deterministically and preserve all source-member provenance.
- Distinguish intentional corpus reuse from accidental duplicate copies.
- Coordinate versioned train/validation group membership with the shared split helper before any corpus-fitted filtering.

## Output and boundary

Produce representative records, duplicate-group/member mappings, stable split
assignments, and summary counts. A resumed or differently parallelized run must
not choose different representatives simply because input arrival order changed.

Near-duplicate detection is a later prototype. See [split plans](../common/splits.md).
