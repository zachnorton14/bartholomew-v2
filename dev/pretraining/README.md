# Pretraining pipeline scaffold

Status: structure and planning documents only. The fresh pipeline has no worker,
stage implementations, executable configs, or tests yet. Implementation pauses at
this milestone for user review.

Changes for this work stay inside `dev/`, primarily inside `dev/pretraining/`.
Existing training/runtime modules outside `dev/` are reference material for later
integration. This milestone adds no executable pipeline code.

## Layout

```text
pretraining/
  README.md
  worker.md
  TO DO.md
  configs/
    README.md
    schema.md
    standard/README.md
    midtraining/README.md
  sources/
    README.md
    institutional.md
    internet_archive.md
    parquet.md
  common/
    README.md
    contracts.md
    fingerprints.md
    splits.md
    state.md
    storage.md
  stages/
    README.md
    gather.md
    screen.md
    refine.md
    dedupe.md
    era.md
    prior.md
    pack.md
    tokenize.md
  rules/
    README.md
    era.md
  validation/README.md
  standard/
    README.md
    pipeline_proposal.md
    DATASET_PIPELINE.md          historical-document pointer
    FILTER_AUDIT.md              historical-document pointer
    old/                        previous standard files
  midtraining/
    README.md
    old/                        previous midtraining files
```

The Markdown files stand in for future modules and describe responsibilities,
contracts, and constraints. Their intended Python filenames are documented inside
them; no placeholder `.py` files are added.

## Layers and responsibilities

| Layer | Plan |
|---|---|
| Worker | [worker.md](worker.md): config-driven planning, dispatch, resume, and reporting. |
| Configuration | [configs/](configs/README.md): identity, shared policy, typed stage parameters, and recipe-specific choices. |
| Source adapters | [sources/](sources/README.md): acquisition and normalization of source records. |
| Shared contracts | [common/](common/README.md): documents, manifests, fingerprints, split membership, state, and storage. |
| Processing stages | [stages/](stages/README.md): eight reusable stages from gathering through pretokenization. |
| Rule assets | [rules/](rules/README.md): versioned temporal rules and their evidence. |
| Verification | [validation/](validation/README.md): future checks and fixtures, without test implementation. |
| Standard recipe | [standard/](standard/README.md): the broad historical book corpus. |
| Midtraining recipe | [midtraining/](midtraining/README.md): the STEM corpus and its source-specific needs. |

Planned order:

```text
gather -> screen -> refine -> dedupe -> era -> prior -> pack -> tokenize
```

`prior` is configurable as off/report/filter. Publishing is a delivery action,
separate from building and inspecting local artifacts.

## Proposal, backlog, and archived sources

- [Pipeline proposal](standard/pipeline_proposal.md) contains the design rationale and illustrative config.
- [TO DO](<TO DO.md>) keeps the unresolved core work and optional improvements; no implementation tasks are marked complete by this structural milestone.
- [Previous standard implementation](standard/old/README.md) preserves the v1/v2/v3 code, notebook cells, notebooks, and supporting documentation.
- [Previous midtraining implementation](midtraining/old/README.md) preserves the extraction, cleaning, mixture, and anneal files.
- [Source manifest](../SCRIPT_SOURCES.json) records the relocated imported files and checksums.

Legacy executable files are preserved as reference material. The fresh layers do
not import or invoke them yet. Cutoff semantics, log-prior policy, and other
scientific decisions remain unresolved until reviewed.

SFT builders remain in [../sft](../sft/README.md), and benchmark builders remain in
[../evaluation](../evaluation/README.md).
