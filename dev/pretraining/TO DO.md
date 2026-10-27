# TO DO

Planning backlog for the pretraining and midtraining data pipelines. Every item below
is pending. This document records requested work; it does not implement fixes or
resolve the copied GitHub issues.

Each item owns a distinct part of the work. Shared requirements are referenced
instead of repeated: metadata correctness belongs to new to-do 1, pipeline
orchestration to new to-do 2, log-prior research to new to-do 3, OCR/alphanumeric
criteria to PT #7, document sourcing to PT #8, and banned-word policy to PT #9.

## High-priority changes

### 1. Fix the v1 -> v2 metadata error and check v2 -> v3

- [ ] Reproduce the reported JSON/metadata ("metajson") error when moving from v1 to v2. Record the exact exception, failing command, affected file, and expected metadata schema before choosing a fix.
- [ ] Fix the underlying metadata-writing or metadata-reading problem in the v1 -> v2 transition.
- [ ] Check the same behavior in v2 -> v3 and ensure the error is not carried into that transition.
- [ ] Verify that audit metadata, shard metadata, and manifests remain consistent with surviving documents after filtering and repacking. Check that data readers select the intended text shards rather than treating metadata artifacts as training rows.
- [ ] Verify both a small end-to-end run and an interrupted/resumed run, including metadata alignment at each transition.

Relevant datasets: [v1](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v1),
[v2](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v2), and
[v3](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3).

### 2. Combine the v1, v2, and v3 filtering processes

- [ ] Create one master script that can run the complete standard-data pipeline, from gathering source data through the final filtered corpus.
- [ ] Keep the master script brief: put stage implementations in helper scripts/modules, such as `stage-1`, `stage-2`, and `stage-3`, while the master handles configuration and orchestration.
- [ ] Inventory the existing v1/v2/v3 filters and map every filter to a stage so combining the pipelines does not silently omit or duplicate filtering behavior.
- [ ] Allow the dataset to change mid-pipeline. Make each stage's input dataset, revision, and prefix configurable so a later stage can consume a different dataset or existing intermediate output without rerunning all earlier stages.
- [ ] Define each stage's input/output contract and output location, using the metadata requirements established by new to-do 1. Preserve the selected dataset lineage when switching inputs.
- [ ] Support running the full pipeline or starting/resuming at a selected stage. Detect incompatible cached state when source datasets or filter settings change.
- [ ] Rename and regroup stages using short, one-word names. Agree on the final names before implementation; preserve the themes below.

Provisional stage names:

| Name | Intended responsibility |
|---|---|
| `gather` | Acquire source documents and the metadata needed for later filtering. |
| `screen` | Apply source-level selection and quality gates, such as date, language, OCR quality, tokenizability, fragment checks, and deduplication. |
| `refine` | Clean OCR/formatting artifacts and trim headers, footers, copyright/reprint boilerplate, and library furniture while preserving legitimate book text. |
| `prior` | Run log-prior calibration and filtering as its own configurable stage, subject to the investigation below. |
| `leakage` | Detect banned terms and other post-cutoff contamination, then reject documents according to the chosen evidence/tier rules. |
| `publish` | Package the final shards, metadata, manifests, and reports and handle the configured upload. |

- [ ] Make the log-prior stage independently selectable so the pipeline can compare runs with and without it and accommodate a later keep/change/remove decision.
- [ ] Verify the master's stage order, dataset switching, shard/validation conventions, and resumability. Use the metadata checks from new to-do 1 rather than duplicating that investigation here.
- [ ] Document the master command, helper responsibilities, stage order, and examples of switching datasets between stages.

This item owns the script structure and stage wiring. Changes to quality thresholds,
log-prior policy, or banned terms belong to their dedicated tasks below.

Existing implementation guides: [standard](standard/old/README.md) and
[midtraining](midtraining/old/README.md).

### 3. Investigate log-prior filtering: keep, change, or remove?

- [ ] Review how the current mean GPT-2 token log-prior score is constructed, calibrated, and applied, including the p2.5-p97.5 acceptance band.
- [ ] Identify the intended benefit and inspect what it actually removes: corrupted OCR, repetitive boilerplate, unusual but legitimate historical prose, and specialized STEM content.
- [ ] Compare no log-prior filtering, the current filter, and plausible alternative thresholds using the same source data, tokenizer, validation split, and training/evaluation budget.
- [ ] Measure documents/tokens retained, subject coverage, false rejections, residual corruption, runtime/cost, and downstream model quality. Interpret validation bits-per-byte alongside other evaluations because removing easy boilerplate changes the prediction task.
- [ ] Assess the standard corpus and midtraining corpus separately; a score calibrated on general books may behave differently on equations, tables, technical vocabulary, and historical STEM.
- [ ] Record an evidence-based decision: keep the stage, change its method/thresholds, or remove it. State whether the measured benefit justifies its cost and complexity.

Related GitHub issue: [#7 — PT - Reevaluate our midtraining filtering](https://github.com/zachnorton14/bartholomew-v2/issues/7).
Current implementations: [standard prior cell](standard/old/cells/cell5_prior.py) and
[midtraining stage 4-5 notebook](midtraining/old/midtraining_stage4_cleaning.ipynb).
This item owns the log-prior investigation mentioned in PT #7; record its results
here and use that decision when recommending the overall midtraining quality bar.

## Existing PT GitHub issues

Snapshot of the three open PT issues in
[zachnorton14/bartholomew-v2](https://github.com/zachnorton14/bartholomew-v2/issues),
recorded on 2026-10-27. Original issue titles are preserved; descriptions and action
items are summarized as text with shared work assigned to the items above. The
GitHub issues themselves are unchanged and remain the source of live status.

### PT #7 — Reevaluate our midtraining filtering

Original title: **PT - Reevaluate our midtraining filtering**  
Source: [GitHub #7](https://github.com/zachnorton14/bartholomew-v2/issues/7)  
Priority: High. Status when recorded: Open.

Review whether the midtraining OCR and alphanumeric gates preserve useful historical
STEM while rejecting corrupted text. The issue contrasts the original corpus's
10-point agreement requirement between OCR measurements with midtraining's
minimum-score gates above 85 and alphanumeric threshold of 65.

- [ ] Compare minimum-score gates with the original OCR/OCRoscope agreement rule, focusing on documents where the measurements disagree.
- [ ] Evaluate whether the alphanumeric threshold of 65 rejects valid equations, tables, and symbols or admits corrupted OCR.
- [ ] Recommend OCR/alphanumeric criteria using accepted/rejected samples and document/token retention counts. Incorporate the log-prior decision from new to-do 3 into the overall quality recommendation.

Scope: OCR and alphanumeric quality criteria. Log-prior experiments are owned by
new to-do 3, banned-word policy by PT #9, and collecting additional documents by PT #8.

Relevant files: [extraction notebook](midtraining/old/midtraining_extraction.ipynb),
[OCRoscope filtering notebook](midtraining/old/midtraining_ocroscope_filtering.ipynb),
and [original corpus filtering rules](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v1#filters).

### PT #8 — Find more midtraining documents

Original title: **PT - Find more midtraining documents**  
Source: [GitHub #8](https://github.com/zachnorton14/bartholomew-v2/issues/8)  
Priority: Medium-high. Status when recorded: Open.

Expand the historical mathematics, science, technology, and medicine corpus. Current
sourcing uses the original corpus and Internet Archive; establish whether additional
material from those sources or alternatives is useful before scaling collection.

- [ ] Review Internet Archive collections and search queries, spot-check candidates, and confirm that the source contains valuable documents meeting the cutoff.
- [ ] Assess other sources, including Project Gutenberg, for STEM coverage, publication dates, text quality, and overlap with the existing corpus.
- [ ] Identify useful STEM books omitted from the original midtraining selection. Separately assess intentional reuse of pretraining material versus accidental duplicates.
- [ ] Gather and assess candidate additions using the criteria established by PT #7 and PT #9 and the log-prior decision from new to-do 3. Report usable documents/tokens by source and subject.

Scope: finding, collecting, and assessing additional documents. Filter design stays
with PT #7/PT #9 and new to-do 3; extraction should consume those decisions.

Relevant links: [extraction notebook](midtraining/old/midtraining_extraction.ipynb),
[Internet Archive texts](https://archive.org/details/texts),
[Project Gutenberg](https://www.gutenberg.org/),
[original corpus](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v1),
and [current midtraining corpus](https://huggingface.co/datasets/zachnorton03/bart-midtrain).

### PT #9 — Refine banned words for midtraining anachronism prevention

Original title: **PT - Refine banned words for midtraining anachronism prevention**  
Source: [GitHub #9](https://github.com/zachnorton14/bartholomew-v2/issues/9)  
Priority: Low. Status when recorded: Open.

Improve detection of post-cutoff content while preserving legitimate historical
meanings. Review both missed contamination and false rejections caused by ambiguous
terms such as "compiler," "drone," and "twitter."

- [ ] Audit the banned list, allow-list, and term tiers against the historical cutoff; document evidence for additions, removals, and confidence changes.
- [ ] Determine when ambiguous terms require context or corroborating evidence before rejecting a book.
- [ ] Inspect retained books for missed modern forewords, annotations, reprint material, and terminology; propose targeted terms/patterns supported by examples.
- [ ] Compare decisions on representative accepted/rejected documents and report documents/tokens affected, detection improvements, and false positives.

Scope: anachronism terms, tier rules, and rejection policy. General header/footer
trimming is grouped under `refine` in new to-do 2; OCR quality belongs to PT #7.

Relevant files: [banned-list construction](midtraining/old/scripts/build_list.py),
[term matching and rejection rules](midtraining/old/scripts/filter_lib.py), and
[the existing v3 tiered filter](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3#the-tiered-anachronism-filter).

---

## Lower-priority / prototype changes

Small optimizations and quality-of-life follow-ups identified from the current
scripts. The core work above takes precedence; the original GitHub issue priorities
remain as recorded. These suggestions do not introduce new filtering policies or
repeat the metadata, pipeline-design, log-prior, sourcing, or banned-word investigations.
Every item remains pending.

### QOL 1. Give the two v3 notebook builders distinct outputs

- [ ] Add an explicit output-path option and distinct default filenames for the inline v3 notebook and the modular runner. Print which implementation was generated so regenerating one does not silently replace the other.

Current behavior: [build_nb_1930s.py](standard/old/build_nb_1930s.py) and
[build_nb_runner.py](standard/old/build_nb_runner.py) both write
`think_dataset_clean_1930s.ipynb`, despite building different implementations.
Scope: generated notebook filenames and usability; stage regrouping stays in
high-priority change 2.

### QOL 2. Distinguish a limited upload run from a local preview

- [ ] Label `DRY_RUN_LIMIT` as a shard limit in messages/documentation, since it still uploads. Add a small no-upload preview mode that processes a bounded sample and writes its report locally without creating or changing a Hugging Face repo.

Current behavior: [run_filter.py](standard/old/scripts/run_filter.py) and
[strip_footers.py](standard/old/scripts/strip_footers.py) create the destination and flush
commits even when `DRY_RUN_LIMIT=1`. This is an operator convenience, separate from
the full-pipeline orchestration work in high-priority change 2.

### QOL 3. Make configuration and listing failures actionable

- [ ] Report invalid numeric environment settings with the variable name and expected range instead of silently falling back to a default. Distinguish an actually empty shard list from a failed Hugging Face listing, and return a clear error/nonzero exit status for the failure case.

Current behavior: `_env_int()` in [config.py](standard/old/scripts/config.py) silently
defaults on invalid integers, and `list_repo_files_safe()` in
[common.py](standard/old/scripts/common.py) returns an empty list after an exception.
Scope: configuration/network diagnostics, not the metadata-schema fix in change 1.

### QOL 4. Keep local cache and upload staging disk usage visible

- [ ] Show free disk space and the sizes of source-cache/output staging directories at startup and between upload batches. Offer opt-in cleanup of local shard files only after their batch is confirmed committed; retain files referenced by pending uploads and keep durable state/audit artifacts.

The standard [filter](standard/old/scripts/run_filter.py) and
[footer-strip](standard/old/scripts/strip_footers.py) drivers retain local source and
output shards. [BatchCommitter](standard/old/scripts/common.py) already distinguishes
pending and committed batches, providing a place to attach a small cleanup hook.
Scope: local storage management; resume/lineage semantics remain owned by change 2.

### QOL 5. Make existing audit samples easier to inspect

- [ ] Add a small local inspection helper for existing hit logs and footer-removal samples, with filters for shard, rejection reason, or term and a seeded sample limit. Make the footer sample cap exact and sample beyond just the earliest matching documents in a shard.

Current behavior: [strip_footers.py](standard/old/scripts/strip_footers.py) fills a sample
list from the first qualifying documents and checks its cap before appending up to
20 more lines, so it can exceed the nominal cap. Filter hit records are produced by
[run_filter.py](standard/old/scripts/run_filter.py). The helper should read existing
artifacts and print a compact review; it does not need a dashboard or new evaluation
framework. PT #7 and PT #9 own the conclusions drawn from these samples.

### QOL 6. Batch the midtraining notebook's per-shard uploads

- [ ] Try one commit containing a shard and its associated stats/hit/sample files instead of separate uploads for each file in the stage 4-5 notebook. Reuse the existing batching approach where practical and compare wall time and commit count on the same small shard set before adopting it.

Current behavior: [midtraining_stage4_cleaning.ipynb](midtraining/old/midtraining_stage4_cleaning.ipynb)
uses several `api.upload_file()` calls per shard. The standard pipeline's
[BatchCommitter](standard/old/scripts/common.py) and the mixture-repair notebook's
`robust_commit()` in [midtraining_mix_repair.ipynb](midtraining/old/midtraining_mix_repair.ipynb)
provide existing patterns. Keep paths, file contents, and completed-shard behavior
unchanged; extending batching across multiple shards is optional.

### Prototype 1. Write cleaned Parquet output in bounded batches

- [ ] Prototype incremental Parquet writing by row group rather than collecting an entire output shard in memory. Compare peak RAM and throughput on representative shards, and adopt only if output rows, order, text, schema, validation assignment, and filter counts stay equivalent.

Current behavior: [run_filter.py](standard/old/scripts/run_filter.py) accumulates
`kept_texts`, and [strip_footers.py](standard/old/scripts/strip_footers.py) accumulates
`out_texts`, before constructing the final Arrow table. Keep this experiment local
to output writing; it does not change filters, corpus selection, or the broader
pipeline architecture.
