# Split membership plan

Future module: `common/splits.py`. Status: documentation only.

## Planned behavior

- Assign exact duplicate groups consistently to train or validation after grouping and before fitting corpus-derived filters.
- Persist/version membership so resumes, worker counts, output shard sizes, and downstream text packing cannot move existing books between splits.
- Define how newly gathered duplicates inherit an existing group's membership.
- Preserve explicit compatible split assignments when importing a prior stage artifact; reject ambiguous conflicting assignments.
- Make prior fitting and any future tokenizer fitting consume training documents only.

The first implementation should agree on a deterministic grouping key, representative
selection, seed, and split fraction. These values are pending design/config review.

Final text exports may preserve the existing last-shard validation convention for
loader compatibility, but that ordering represents already-decided membership.
`pack` and `tokenize` do not resample the evaluation set.
