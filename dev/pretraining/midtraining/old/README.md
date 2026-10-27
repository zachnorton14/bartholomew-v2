# Previous midtraining corpus and mixtures: file guide

Archived implementation: these are the previous corpus, mixture, and anneal files,
preserved for reference during the fresh start. Executable file contents are
unchanged; embedded paths may describe the earlier checkout layout. The fresh
scaffold does not invoke these notebooks or helpers.

See the [active midtraining guide](../README.md) and
[shared scaffold](../../README.md) for the replacement plans.

This directory builds the historical STEM corpus and blends it with the broad
[standard pretraining corpus](../../standard/old/README.md). The process is subject
extraction / Internet Archive harvest -> OCR filtering -> structural and log-prior
cleaning -> footer and anachronism filtering -> training mixtures.

The corpus and its intermediate outputs live in
[zachnorton03/bart-midtrain](https://huggingface.co/datasets/zachnorton03/bart-midtrain).
The final deliverable is the [corpus/ folder](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/corpus).

## Directory map

| Folder | Purpose |
|---|---|
| [scripts/](scripts) | Four reusable, local cleaning/filtering modules called by the stage 4-5 notebook. The notebook handles Hugging Face I/O and persistence. |
| [configs/](configs) | Historical YAML recipes for annealing from an existing training checkpoint and comparing mixture ratios. These configure training experiments rather than corpus extraction. |

## Top-level notebooks and files

| File | What it does | Related dataset / output |
|---|---|---|
| [README.md](README.md) | This inventory, pipeline order, current dataset locations, and historical configuration notes. | [Dataset card](https://huggingface.co/datasets/zachnorton03/bart-midtrain) |
| [midtraining_extraction.ipynb](midtraining_extraction.ipynb) | Stages 1-2. Selects underrepresented/rich STEM subjects from book metadata, then harvests additional pre-1930 Internet Archive books. Applies topic/date selection, deduplication, OCR heuristics, sharding, and QA sampling. | Stage 1 uses [dataset v1](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v1); the published harvest is [pipeline/stage2/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/pipeline/stage2). |
| [midtraining_ocroscope_filtering.ipynb](midtraining_ocroscope_filtering.ipynb) | Stage 3. Scores harvested documents with OCRoscope, checkpoints scores per source shard, lets the operator inspect score distributions, filters on OCR probability and alphanumeric ratio, and repacks/uploads survivors. | Reads stage 2; writes [pipeline/stage3/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/pipeline/stage3). |
| [midtraining_stage4_cleaning.ipynb](midtraining_stage4_cleaning.ipynb) | Stages 4-5. Uses `scripts/` for boilerplate/OCR cleanup and structural gates, calibrates/applies the GPT-2 log-prior band, strips residual footers, and applies the tiered anachronism filter. Handles per-shard resume state, uploads, manifests, reports, and read-back verification. | Intermediate output: [pipeline/stage4/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/pipeline/stage4). Final books: [corpus/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/corpus). Final-stage audits: [pipeline/stage5/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/pipeline/stage5). |
| [midtraining_mix_ratios.ipynb](midtraining_mix_ratios.ipynb) | Historical mixture builder. Indexes the midtrain and old-corpus pools, samples and shuffles documents, plans shards, records provenance, and uploads resumable blends. Its document-count ratios under-deliver the intended midtrain token share; retained for reproducing earlier experiments. | Superseded [mixtures/v1-by-docs/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/mixtures/v1-by-docs). |
| [midtraining_mix_repair.ipynb](midtraining_mix_repair.ipynb) | Current mixture builder. Loads the run's exact tokenizer, measures tokens, deduplicates source books, reserves validation, tracks source disjointness, and interleaves whole documents to the target token ratio. Writes shards, token-count sidecars, provenance, manifests, and verification results. | Current [21% mixture](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/mixtures/v2-by-tokens/ratio_21) and [45% mixture](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/mixtures/v2-by-tokens/ratio_45), blended with [dataset v3](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3). |
| [midtraining_anneal_ablation.ipynb](midtraining_anneal_ablation.ipynb) | Training runner for anneal experiments. Loads a YAML recipe, resolves the parent checkpoint/tokenizer, derives matching architecture settings, prepares and pretokenizes a mixture, runs additional training with the selected learning-rate schedule, and uploads checkpoints/configs with W&B logging. | Consumes mixtures; writes model experiment artifacts to the configured Hugging Face model repo. This notebook trains a model rather than constructing the corpus. |
| [d32-branch-sample-config.json](d32-branch-sample-config.json) | Example d32 experiment-harness configuration for the repaired token-weighted mixtures. Records the staged 21%/45% schedule, parent-run assumptions, verified shard/token accounting, mixture revision, pretokenization settings, and run identity. | Uses the [v2-by-tokens mixtures](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/mixtures/v2-by-tokens); its stored repo paths and revisions describe the historical run. |
| [UPSTREAM_README.md](UPSTREAM_README.md) | Original `bart-midtrain-scripts` overview, including the corpus stages, mixture bug and repair, anneal experiments, and upstream repository layout. | [Dataset card](https://huggingface.co/datasets/zachnorton03/bart-midtrain) |
| [LICENSE](LICENSE) | MIT license retained from the upstream midtraining-script repository. | Applies to the imported source snapshot. |

Use `midtraining_mix_repair.ipynb` when constructing the token-weighted blends.
The historical nominal 0%/30%/60% mixtures and the repaired 21%/45% mixtures belong
to different generations; their ratio labels are not interchangeable.

## `scripts/`: stage 4-5 cleaning libraries

These modules implement the local text operations. They are the midtrain-specific
variants, whose settings can be assigned directly by the notebook, and should stay
beside the notebooks that import them.

| File | What it does |
|---|---|
| [structural_clean.py](scripts/structural_clean.py) | Stage 4 cleanup and structural gate. Removes Google Books, HathiTrust, Gutenberg, and library boilerplate; normalizes Unicode/ligatures and whitespace; repairs wrapped or hyphenated lines; trims front matter; then checks raw/clean length, printable ratio, and OCR-artifact density. Returns cleaned text or a rejection reason. The log-prior calculation itself remains in the notebook. |
| [build_list.py](scripts/build_list.py) | Builds the curated 1930 banned terms from an embedded older seed, a pre-1931 allow-list, and post-1930 additions. Assigns tiers and returns terms plus curation/audit metadata. The notebook caches and uploads the results. |
| [footer_lib.py](scripts/footer_lib.py) | Stage 5 line-level removal of modern footer and reproduction boilerplate. Returns cleaned text, removed-line details, and a flag when the removal-fraction guardrail requires keeping the original document. |
| [filter_lib.py](scripts/filter_lib.py) | Stage 5 fast word/phrase/punctuation/format matching and tiered whole-book decisions. One tier-1 hit drops a book; a corroborated combination needs at least two distinct tier-2/3 hits including tier 2. Tier-3 or strip-only signals cannot drop books alone. |

The resulting curation lists, strip audits, and filter statistics are published under
[pipeline/stage5/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/pipeline/stage5).

## `configs/`: anneal recipes

These files are inputs to `midtraining_anneal_ablation.ipynb`. Parameters that must
match the parent model are derived from checkpoint metadata; the recipe identifies
the parent, extra training horizon, mixture selection, and optional learning-rate
overrides.

| File | What it does |
|---|---|
| [anneal_ablation_sample.yaml](configs/anneal_ablation_sample.yaml) | Annotated schema/example: resumes the historical d12 parent at step 2362 for 500 extra steps on a nominal 30% mixture. Documents unique experiment tags, checkpointing, and linear/constant schedules with per-group or multiplier-based learning-rate overrides. |
| [anneal_d24_ratio0.yaml](configs/anneal_d24_ratio0.yaml) | Historical d24 control arm: resumes the clean-1930s parent at step 5500 for 250 additional steps using 0% midtrain data, decays the learning rate to zero, and saves the final checkpoint. |
| [anneal_d24_ratio60.yaml](configs/anneal_d24_ratio60.yaml) | Historical d24 comparison arm with the same parent and 250-step horizon, selecting the nominal 60% mixture and final-checkpoint-only saving. This refers to the earlier mixture naming rather than the repaired 45% blend. |

## Published paths versus paths in the scripts

The imported code retains `zachnorton03/think-midtrain` and the original directory
layout. The current Hugging Face repo is `zachnorton03/bart-midtrain`; both the repo
name and the internal folder names have changed.

| Historical path in code | Current Hugging Face location |
|---|---|
| `stage2/data` | [pipeline/stage2/data/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/pipeline/stage2/data) |
| `stage3/data` | [pipeline/stage3/data/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/pipeline/stage3/data) |
| `stage4/data` | [pipeline/stage4/data/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/pipeline/stage4/data) |
| `stage5/data` | [corpus/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/corpus) |
| Other stage 5 artifacts | [pipeline/stage5/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/pipeline/stage5) |
| `mixed/v2/ratio_21/data` | [mixtures/v2-by-tokens/ratio_21/data/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/mixtures/v2-by-tokens/ratio_21/data) |
| `mixed/v2/ratio_45/data` | [mixtures/v2-by-tokens/ratio_45/data/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/mixtures/v2-by-tokens/ratio_45/data) |
| Earlier document-count mixtures | [mixtures/v1-by-docs/](https://huggingface.co/datasets/zachnorton03/bart-midtrain/tree/main/mixtures/v1-by-docs) |

## Working directories and configuration

Start the corpus notebooks with `dev/pretraining/midtraining` as the working
directory, or set `SCRIPTS_DIR` explicitly in the stage 4-5 notebook. It uses
`Path("scripts")` to import the local helpers. Generated `__pycache__/` directories,
if present, contain interpreter caches rather than pipeline source.

Set the intended repo IDs, source/destination prefixes, local paths, and tokenizer
locations before executing a notebook. For token-exact mixtures, also align the
corpus/tokenizer revisions with the target training run. The retained historical
configs are examples with specific parent checkpoints and output identities.

The anneal notebook needs a full training checkout. Current training entrypoints
remain in the project's top-level `scripts/`; evaluation notebooks from the same
upstream source are organized separately in [../../evaluation/colab](../../../evaluation/colab).

## Source provenance

The source snapshot comes from [OwenVoorhees/bart-midtrain-scripts](https://github.com/OwenVoorhees/bart-midtrain-scripts).
See [UPSTREAM_README.md](UPSTREAM_README.md) for its original documentation and
[../../SCRIPT_SOURCES.json](../../../SCRIPT_SOURCES.json) for exact revisions and file
checksums. Dataset shards, source books, checkpoints, and saved execution outputs
are not included in this source directory.
