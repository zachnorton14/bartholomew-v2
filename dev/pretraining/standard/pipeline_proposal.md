# Data pipeline proposal

Status: proposed architecture, recorded on 2026-10-27. This document does not
implement the worker, config schema, or stages shown below. Example settings are
illustrative and must be reviewed before implementation.

The [shared scaffold](../README.md) now lays out these responsibilities as Markdown
plans. Previous standard and midtraining implementations live in their respective
`old/` folders. This structure-only milestone stops for user review before any
fresh executable implementation is added. All implementation edits stay inside
`dev/`, primarily `dev/pretraining/`, unless the user explicitly changes that scope.

## Goal

Build a configuration-driven gathering and filtering harness with the adaptability
of the [training harness](../../../EXPERIMENTS.md). One worker reads a config,
validates its inputs, runs the requested stages with their parameters, and records
durable outputs and lineage. The same stage implementations serve standard and
midtraining recipes through different configs and source adapters.

Each config has a unique `pipeline_id`, a human-readable `name`, a `description`,
source definitions, shared policy, stage parameters, and execution settings.
The worker stays brief; source adapters and stage helpers contain the implementation.

## Recommended stage order

```text
gather -> screen -> refine -> dedupe -> era -> prior -> pack -> tokenize
```

`prior` is optional. Publishing is a separate delivery command after local output
has been built and inspected.

| Stage | Responsibility | Reason for its position |
|---|---|---|
| `gather` | Acquire source documents and metadata; translate source fields into a common document format. | All later stages operate on consistent records regardless of source. |
| `screen` | Apply eligibility and available source-quality checks: date, language, subject, OCR scores, tokenizability, and pathological fragments. | Reject obvious failures before more processing. Cheap metadata checks should precede expensive fetching or recomputed scores where the source allows it. |
| `refine` | Trim headers, footers, copyright/reprint boilerplate, and library furniture; normalize OCR/formatting; recheck cleaned-text length and structural quality. | Later decisions should inspect the useful book text. |
| `dedupe` | Identify duplicate books using source identifiers and cleaned-text hashes. | Cleanup exposes duplicates previously hidden by different scanning boilerplate. |
| `era` | Detect post-cutoff contamination and apply the chosen banned-term/tier rules. | Remove clear temporal contamination before fitting statistical quality filters. |
| `prior` | Optionally calculate token log-prior scores and report or reject outliers. | Fit on the surviving training corpus rather than on obvious corruption or temporal contamination. |
| `pack` | Write final text-only Parquet shards, audit mappings, manifests, and summaries. | Convert rich intermediate records to the training dataset contract. |
| `tokenize` | Pretokenize the packed corpus using the exact training tokenizer; write token shards and token-cache metadata. | Run after final text and split membership are fixed. |

### Naming and stage boundaries

- Prefer `era` to `leakage` for historical-cutoff enforcement. Train/validation contamination is handled through duplicate grouping and split checks.
- Stage names describe responsibilities rather than historical dataset versions. v1/v2/v3 remain useful provenance labels, but the combined implementation should use the operation names above.
- Keep small related filters inside their owning stage. A separate stage becomes useful when it has a distinct input/output contract, an expensive reusable result, or different resource requirements.
- OCRoscope computation may warrant a separate scoring stage if it needs its own GPU worker/checkpoint boundary. Reading existing OCR measurements remains part of `screen`.
- Begin with exact cross-source deduplication. Near-duplicate detection can remain a later prototype.

### Ordering details

Preserve the original input reference and record evidence removed by `refine`.
Removing a modern scanner footer should not erase the record of that signal, and
a modern foreword should not be treated the same way as a digitization footer.
The `era` stage can consider cleaned text alongside the retained evidence.

Avoid repeated source identifiers during gathering, then perform cross-source
content deduplication after refinement. Establish duplicate-group split membership
before fitting any data-derived filter. `pack` preserves that membership rather
than deciding it afresh from shard order.

## Worker and helper structure

Proposed layout under `dev/pretraining/`:

```text
worker.py
configs/
  standard/
  midtraining/
sources/
  institutional.py
  internet_archive.py
  parquet.py
stages/
  gather.py
  screen.py
  refine.py
  dedupe.py
  era.py
  prior.py
  pack.py
  tokenize.py
```

The worker should:

1. Validate the config, known stage types, parameter names, and input contracts.
2. Resolve source revisions and save the fully resolved run specification.
3. Print a plan showing stage inputs, outputs, parameters, and reusable results.
4. Dispatch stage helpers in dependency order.
5. Manage checkpoints, completed outputs, summaries, and failure reporting.
6. Publish artifacts only through the requested delivery action.

Source adapters own acquisition and source-field mapping. Stage helpers own data
operations. Shared storage/state utilities own durable writes and artifact lookup.
Configs should select registered adapters/stages and parameters rather than contain
arbitrary Python or shell commands.

Start with an ordered stage list whose inputs reference earlier stages or pinned
external artifacts. This provides the needed flexibility without requiring a
general workflow scheduler at the outset.

## Illustrative config

Keep JSON to match the training harness. The following demonstrates the proposed
shape; adapters, rule IDs, and fields are not yet an implemented interface.
`<pinned-commit>` and other angle-bracket values must be replaced before use.
The cutoff shown is an example of an explicit policy, not a decision that resolves
the current cutoff inconsistency.

```json
{
  "schema_version": 1,
  "pipeline_id": "books-standard-001",
  "name": "Standard historical corpus",
  "description": "Combined gathering, filtering, and pretokenization recipe.",
  "seed": 42,
  "policy": {
    "cutoff": { "year": 1930, "inclusive": false }
  },
  "sources": {
    "books": {
      "adapter": "institutional",
      "repo": "institutional/institutional-books-1.0",
      "revision": "<pinned-commit>"
    }
  },
  "stages": [
    {
      "id": "gather",
      "type": "gather",
      "input": "source:books",
      "params": {}
    },
    {
      "id": "screen",
      "type": "screen",
      "input": "stage:gather",
      "params": { "language": "eng", "english_min": 0.9 }
    },
    {
      "id": "refine",
      "type": "refine",
      "input": "stage:screen",
      "params": { "strip_footers": true }
    },
    {
      "id": "dedupe",
      "type": "dedupe",
      "input": "stage:refine",
      "params": { "method": "exact" }
    },
    {
      "id": "era",
      "type": "era",
      "input": "stage:dedupe",
      "params": { "rules": "historical-1930-v1" }
    },
    {
      "id": "prior",
      "type": "prior",
      "input": "stage:era",
      "params": { "mode": "report", "fit_split": "train" }
    },
    {
      "id": "pack",
      "type": "pack",
      "input": "stage:prior",
      "params": { "chars_per_shard": 250000000 }
    },
    {
      "id": "tokenize",
      "type": "tokenize",
      "input": "stage:pack",
      "params": {
        "tokenizer": {
          "repo": "<tokenizer-repository>",
          "revision": "<pinned-commit>",
          "subfolder": "<tokenizer-path>"
        },
        "shard_tokens": 100000000,
        "dtype": "uint16"
      }
    }
  ],
  "execution": {
    "workers": 4,
    "upload_batch_size": 20
  }
}
```

An empty `params` object uses schema-defined defaults. Save those defaults explicitly
in the resolved run specification so a later default change cannot silently alter
an existing run. Keep credentials in the environment rather than in configs.

### Switching datasets midway

Every stage has an explicit input. A later stage may reference a pinned external
stage artifact instead of the previous stage's output. The worker should validate
the artifact's schema, required metadata, checksums, and lineage before continuing.

For example, a new recipe could reuse an earlier run's completed `refine` output,
then apply different deduplication or era rules. It should not rerun gathering or
silently pretend that the substituted input came through this recipe's earlier stages.
Use a new recipe identity for the changed processing settings.

An already packed text-only dataset may lack metadata required by `screen`.
Make missing-field behavior explicit and reject incompatible stage inputs rather
than guessing or filling scientific measurements with defaults.

## Identity, caching, and output contracts

### Recipe identity and execution attempts

`pipeline_id` identifies the recipe. A generated `attempt_id` identifies each worker
invocation, including retries of the same recipe. A processing fingerprint identifies
the computational settings and dependencies. Reject reuse of an existing recipe ID
with changed processing settings. Human-readable names/descriptions should not
invalidate cached stage results.

### Cache individual stages

A stage fingerprint should depend on:

- Its input artifact fingerprint.
- Resolved stage parameters and applicable shared policy.
- Relevant helper code and output-affecting dependency versions.
- Rule files, tokenizer identity, fitted statistics, and sampling seeds as applicable.

Changing the banned list should reuse gathering/refinement and rerun `era` plus
affected downstream stages. Changing only the training tokenizer should reuse all
filtering and packing results and rerun `tokenize`.

Record the full repository revision for provenance; cache invalidation should track
the relevant stage dependencies. Cosmetic name changes should not rebuild a corpus.
This follows the previous-state-plus-transformation principle described in
[Hugging Face Datasets fingerprinting](https://huggingface.co/docs/datasets/about_cache#fingerprint).

### Rich intermediate records, simple training exports

Intermediate records should retain stable `doc_id`, text, source identity, metadata,
and provenance. Preserve source identity across text edits and track text hashes
separately. Join audit records by IDs, not by assuming row positions stay aligned.

`pack` exports the existing [text-only shard contract](old/DATASET_PIPELINE.md) and a
separate mapping from document IDs to exported shard/row positions. Keep metadata
and training-shard discovery separate.

Each stage should emit a manifest identifying its inputs, parameters, fingerprints,
required files, checksums, and completion state, plus a summary with processed,
accepted, rejected, and failed counts. Use deterministic samples for inspection.

### Durable completion and resume

A shard is complete only when its required data and audit artifacts are durable
and recorded in the manifest. File existence alone is insufficient. Commit related
artifacts together where possible, then mark the unit complete. Retries must not
duplicate output rows or incorrectly skip partially written results.

Derive run artifact paths from recipe/stage identity, following the training harness's
approach. Put reusable stage results behind fingerprinted artifact references so
new recipes can reuse unchanged work without overwriting previous runs.

### Train/validation isolation

Assign duplicate groups consistently to train or validation before fitting statistical
filters. Fit prior statistics on the training split only. Freeze/version the split
mapping so changes to shard sizes, worker counts, or resume points cannot move books
between splits. Packing and pretokenization must preserve that mapping.

## Pretokenization stage

`tokenize` consumes the final packed corpus and a pinned training tokenizer. It
produces token shards, token counts, and metadata compatible with the training loader.
Use the existing [pretokenizer](../../../scripts/pretok_think.py) and
[pretokenized loader](../../../nanochat/pretok_dataloader.py) as integration references.

The stage should:

- Fingerprint both the input corpus and tokenizer, including document-boundary/BOS handling and output-affecting tokenization parameters.
- Preserve train/validation membership and record per-split/per-shard token counts.
- Validate that token IDs fit the configured dtype; the existing writer uses `uint16`.
- Resume from completed token shards with compatible manifests.
- Record requested token limits and actual coverage so a truncated cache is distinguishable from a complete corpus export.
- Keep the tokenizer for prior scoring separately identified from the training tokenizer.

If a recipe needs a new tokenizer, fit it on the final training split before
`tokenize`, then pin its artifact. Tokenizer fitting may initially remain with the
training harness; the data stage can consume that exact artifact. An independently
checkpointed tokenizer-fitting stage can be added later if needed.

## Pitfalls and decisions to resolve

| Concern | Proposed treatment |
|---|---|
| Inconsistent cutoff semantics | Decide whether the corpus includes 1930. The current v1 builder rejects years `>= 1930`, while the anachronism config describes post-1930 content as disallowed. One explicit policy should govern both. |
| Different source schemas and score units | Normalize fields/units in source adapters. Configure whether absent dates/OCR measurements cause rejection, recomputation, or an explicitly identified fallback. |
| Live sources changing | Pin Hugging Face inputs to commits. For Internet Archive, preserve selected identifiers, fetched-file hashes, and acquisition metadata; a query alone cannot reproduce the same corpus. |
| Reusing stale prior thresholds | Tie calibration to cleaned-input identity, tokenizer, sampling settings, and split. Existing filenames such as `_prior/thresholds.json` are insufficient proof of compatibility. |
| Misinterpreting log-prior | The current implementation estimates token frequencies from a sample using a GPT-2 tokenizer. Treat it as an outlier signal whose benefit needs measurement. Support `off`, `report`, and `filter`; define `off` as explicit pass-through with a skipped-stage record. |
| Parallelism changing results | Make source order, duplicate representative selection, sampling, and shuffling deterministic. A global seed alone does not guarantee this. |
| Stripping useful evidence or prose | Retain raw input references and removal evidence. Keep footer removal and whole-document temporal rejection as different operations. |
| Excessive intermediate storage | Choose materialized checkpoint boundaries deliberately and define retention for reproducible artifacts. Preserve audit/manifests when removing disposable caches. |
| Turning config into an overly generic programming language | Validate a small set of registered stages and typed parameters. Defer arbitrary execution graphs and complex inheritance until a concrete recipe needs them. |

Hugging Face downloads support specific commit revisions; the
[download guide](https://huggingface.co/docs/huggingface_hub/guides/download#from-specific-version)
documents revision selection and full commit hashes.

## Suggested implementation sequence

1. Define the config schema, common document contract, stage manifests, and resume rules.
2. Wrap the current gather/filter implementations behind those contracts and reproduce existing behavior on a small corpus.
3. Add explicit external-stage inputs and verify dataset switching and reuse of unchanged stages.
4. Integrate `pack` and `tokenize` with the training harness's existing loader contracts.
5. Evaluate filtering-policy changes separately, including the keep/change/remove decision for log-prior.

The first milestone is a reproducible combined pipeline. New filters and substantial
policy changes remain separate work in [TO DO](<../TO DO.md>).
