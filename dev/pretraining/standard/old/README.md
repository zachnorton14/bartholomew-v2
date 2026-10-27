# Previous standard pretraining corpus: file guide

Archived implementation: these are the previous builders, notebooks, and helpers,
preserved for reference during the fresh start. The executable file contents are
unchanged; embedded paths in code/notebook text may describe the earlier checkout
layout. This archive is not invoked by the fresh scaffold.

See the [active standard guide](../README.md) and
[shared scaffold](../../README.md) for the replacement plans.

This directory contains the preparation pipeline for the broad historical book corpus:
Institutional Books -> v1 -> v2 -> v3. Each surviving book remains a complete row in
text-only Parquet shards. The final corpus feeds standard pretraining; the separate
[midtraining directory](../../midtraining/old/README.md) builds the STEM corpus and mixtures.

## Stages and Hugging Face datasets

| Stage | What it does | Dataset |
|---|---|---|
| Source | Original books, publication metadata, OCR measurements, and subject tags | [Institutional Books 1.0](https://huggingface.co/datasets/institutional/institutional-books-1.0) |
| v1 | Selects English books before 1930, applies OCR and tokenizability gates, deduplicates, shuffles, and shards | [bartholomew-dataset-v1](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v1) |
| v2 | Removes boilerplate and OCR corruption, then filters on mean GPT-2 token log-prior | [bartholomew-dataset-v2](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v2) |
| v3 | Strips residual footer lines and removes books with strong evidence of post-1930 content | [bartholomew-dataset-v3](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3) |

## Directory map

| Folder | Purpose |
|---|---|
| [cells/](cells) | Ordered notebook source fragments for the v1 -> v2 cleaning pass. |
| [cells_1930s/](cells_1930s) | Earlier inline notebook implementation of the v2 -> v3 anachronism filter. |
| [cells_runner/](cells_runner) | Notebook cells that configure and invoke the modular v3 pipeline in `scripts/`. |
| [scripts/](scripts) | Reusable v3 libraries and command-line stages, copied from the scripts published with the v3 dataset. |

## Top-level files

| File | What it does |
|---|---|
| [README.md](README.md) | This inventory, the dataset-stage map, and notes on running or regenerating the notebooks. |
| [pipeline_proposal.md](../pipeline_proposal.md) | Proposed config-driven worker, stage order through pretokenization, config example, dataset switching, caching/resume contracts, and design tradeoffs. |
| [repackage_institutional_books.py](repackage_institutional_books.py) | Builds the initial v1-style corpus from Institutional Books. Applies metadata, English-language, date, OCR, fragment-size, and tokenizability filters; deduplicates books; shuffles and writes shards; maintains resumable state and audit metadata; supports statistics-only checks and incremental Hugging Face uploads. |
| [repackage_data_reference.py](repackage_data_reference.py) | Original nanochat sharding reference for [FineWeb-EDU](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) and [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle). Demonstrates shuffling, text conversion, Parquet packing, and upload. These are modern reference corpora, outside the historical v1 -> v3 chain. |
| [think_dataset_clean.ipynb](think_dataset_clean.ipynb) | Complete v1 -> v2 notebook: setup, optional reset, structural cleaning, log-prior calibration, resumable shard processing, and aggregate reporting. Generated from `cells/`. |
| [think_dataset_clean_1930s.ipynb](think_dataset_clean_1930s.ipynb) | The supplied modular v3 runner: authenticates, publishes/loads `scripts/`, builds the banned-term list, strips footers, filters books, and generates reports and visualizations. Its editable notebook cells live in `cells_runner/`. |
| [build_nb.py](build_nb.py) | Assembles `cells/` and explanatory Markdown into `think_dataset_clean.ipynb`, then validates the notebook structure. |
| [build_nb_1930s.py](build_nb_1930s.py) | Assembles the older inline filter from `cells_1930s/` into `think_dataset_clean_1930s.ipynb`. This replaces the supplied modular notebook with the earlier implementation. |
| [build_nb_runner.py](build_nb_runner.py) | Assembles `cells_runner/` into the modular `think_dataset_clean_1930s.ipynb`. This is the generator corresponding to the supplied v3 notebook. |
| [DATASET_PIPELINE.md](DATASET_PIPELINE.md) | v1 shard contract, premium filter defaults, local/Drive state layout, resume and upload behavior, and build-verification instructions. |
| [FILTER_AUDIT.md](FILTER_AUDIT.md) | Rationale for the Institutional Books quality thresholds and results of the original metadata and shard smoke checks. |
| [clean_dataset_prompt.txt](clean_dataset_prompt.txt) | Historical design brief for the v2 notebook, including decisions to keep whole books, skip the pre-1900 physics veto, and use a moderate log-prior band. It is project background, not an executable script. |
| [UPSTREAM_README.md](UPSTREAM_README.md) | Original documentation from `bart-dataset-scripts`, with the upstream layout, pipeline rationale, and historical dataset names. |
| [LICENSE](LICENSE) | MIT license retained from the upstream cleaning-script repository. |

## `cells/`: v1 -> v2 notebook source

These files share the notebook's execution state and run in the order shown. The
result is the [v2 corpus](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v2).

| File | What it does |
|---|---|
| [cell1_install.py](cells/cell1_install.py) | Installs notebook dependencies and imports dataset, Parquet, tokenizer, numerical, progress, and Hugging Face utilities. |
| [cell2_config.py](cells/cell2_config.py) | Selects source/destination repos, resolves Hugging Face authentication, defines structural and prior-filter thresholds, sets shuffle/sample seeds, and creates local work directories. |
| [cell3_wipe.py](cells/cell3_wipe.py) | Defines and runs the optional destination-reset step, controlled by `CONFIRM_WIPE`; supports restarting the cleaning pass from scratch. |
| [cell4_cleanlib.py](cells/cell4_cleanlib.py) | Implements Google Books/HathiTrust/Gutenberg and library-stamp removal, Unicode/OCR normalization, line reflow, front-matter cleanup, structural quality gates, and source/destination shard helpers. |
| [cell5_prior.py](cells/cell5_prior.py) | Loads cached GPT-2 token priors and thresholds or estimates them from a sample of structurally cleaned books. Computes per-document mean log-prior and the percentile acceptance band; persists prior artifacts to Hugging Face. |
| [cell6_main.py](cells/cell6_main.py) | Processes each unfinished source shard through cleaning and prior filtering; writes matching text-only Parquet shards and per-shard statistics, and uploads results with resume support. |
| [cell7_report.py](cells/cell7_report.py) | Aggregates shard statistics, removal reasons, and character accounting into the cleaning report and dataset README. |

## `cells_1930s/`: earlier inline v3 notebook source

This is the older self-contained implementation. Its filter counts distinct banned
signals against `MIN_BANNED_HITS`; it does not use the modular pipeline's tiered
decision rule. Regenerating from this folder changes the implementation as well as
the notebook layout.

| File | What it does |
|---|---|
| [cell1_install.py](cells_1930s/cell1_install.py) | Installs dependencies and imports the Hugging Face, Parquet, numerical, and reporting utilities used by the inline notebook. |
| [cell2_config.py](cells_1930s/cell2_config.py) | Sets source/destination repos, authentication, the 1930 cutoff, scan windows, minimum hit count, output settings, local directories, and reset controls. |
| [cell3_wipe.py](cells_1930s/cell3_wipe.py) | Implements the guarded removal of previous destination outputs so the inline filter can restart. |
| [cell_build_list.py](cells_1930s/cell_build_list.py) | Constructs and caches a 1930-adjusted banned list from the embedded seed, allow-list, and additions; compiles the fast word/phrase/format matchers and uploads list artifacts. |
| [cell_filter.py](cells_1930s/cell_filter.py) | Scans book heads/tails for banned signals, applies the distinct-hit threshold, and runs positive/negative example checks. It removes whole books rather than editing their prose. |
| [cell_main.py](cells_1930s/cell_main.py) | Lists unfinished shards, downloads and scans books, writes kept texts, records per-shard statistics/hit logs, uploads outputs, and reports progress. |
| [cell_report.py](cells_1930s/cell_report.py) | Combines per-shard results into an anachronism-cleaning report and dataset README. |

## `cells_runner/`: modular v3 notebook source

These cells orchestrate `scripts/`. They correspond to the supplied v3 notebook and
the [published v3 pipeline modules](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3/tree/main/scripts).

| File | What it does |
|---|---|
| [cell1_bootstrap.py](cells_runner/cell1_bootstrap.py) | Installs dependencies, authenticates to Hugging Face, exports pipeline settings as environment variables, and ensures the destination repo exists. |
| [cell2_push_scripts.py](cells_runner/cell2_push_scripts.py) | Uploads the local `scripts/` directory to the destination dataset, clones or updates that repo in Colab, and locates the scripts to execute. |
| [cell3_wipe.py](cells_runner/cell3_wipe.py) | Invokes the optional guarded `scripts/wipe.py` reset, with wipe flags disabled by default. |
| [cell4_build_list.py](cells_runner/cell4_build_list.py) | Invokes `scripts/build_list.py` to build or reuse the tiered banned-term artifacts. |
| [cell4b_strip_footers.py](cells_runner/cell4b_strip_footers.py) | Runs footer stripping on one shard and produces strip statistics/samples for inspection. |
| [cell4c_strip_all.py](cells_runner/cell4c_strip_all.py) | Resumes footer stripping over every remaining source shard after the sample pass. |
| [cell5_run_filter.py](cells_runner/cell5_run_filter.py) | Runs the anachronism filter on one footer-stripped shard, using the destination repo's `stripped/` layer as input. |
| [cell6_run_all.py](cells_runner/cell6_run_all.py) | Resumes anachronism filtering over all remaining footer-stripped shards. |
| [cell7_report.py](cells_runner/cell7_report.py) | Invokes the report stage to summarize completed shards and publish the cleaning report and dataset README. |

## `scripts/`: modular v3 pipeline

The normal sequence is list construction -> footer stripping -> anachronism filtering
-> reporting. The wipe command is an optional reset. Script stage numbers are local
to this v3 pass, separate from dataset versions v1/v2/v3.

| File | What it does |
|---|---|
| [config.py](scripts/config.py) | Shared settings for repo IDs/prefixes, cutoff policy, scan windows, footer guardrails, Parquet output, upload batching, run limits, and work directories. Environment variables override defaults. Importing this module does not authenticate or access the network. |
| [common.py](scripts/common.py) | Shared Hugging Face authentication/API client, repo creation, shard listing and downloads, output-path mapping, and batched commit helpers. Used by the stages that read or write dataset repos. |
| [build_list.py](scripts/build_list.py) | Builds or reloads the banned terms, allow-list, tier assignments, and curation metadata; caches/uploads them under the destination's `_banned/` folder. |
| [filter_lib.py](scripts/filter_lib.py) | Compiles fast word, phrase, punctuation, and format matchers; scans configurable head/tail windows; and applies the tier rule: one tier-1 hit drops a book, or at least two distinct tier-2/3 hits with at least one tier-2 hit. Strip-only format signals do not independently drop books. |
| [footer_lib.py](scripts/footer_lib.py) | Detects short modern footer/boilerplate lines and removes them while preserving book text. If the removed-line fraction exceeds the configured cap, it returns the original document and flags it for audit. |
| [strip_footers.py](scripts/strip_footers.py) | Resumable driver around `footer_lib.py`: reads `STRIP_SRC_REPO`, writes same-named shards under `STRIP_PREFIX` in `DST_REPO`, and uploads per-shard strip statistics and removed-line samples. |
| [run_filter.py](scripts/run_filter.py) | Resumable whole-book filter: loads the banned list, scans source shards, writes kept text-only shards, and uploads statistics and hit logs in commit batches. |
| [report.py](scripts/report.py) | Aggregates completed shard statistics, removal reasons, term frequencies, and character counts into `cleaning_report_1930s.json` and the destination README. Can report partial progress. |
| [wipe.py](scripts/wipe.py) | Deletes generated destination outputs only when `CONFIRM_WIPE=1`. By default retains the stripped corpus and banned list; `WIPE_STRIPPED` and `WIPE_BANNED_LIST` opt into removing those layers too. |

Useful published artifacts are the [v2 prior calibration](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v2/tree/main/_prior),
[v3 banned-term artifacts](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3/tree/main/_banned),
and [v3 footer-stripped corpus](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3/tree/main/stripped).

## Working directories and historical configuration

The `cells*` files are notebook fragments, with shared state and some IPython `%pip`
or `!` commands. Use their generated notebooks rather than executing individual
fragments as standalone Python scripts. Generated `__pycache__/` directories, if
present, are interpreter caches rather than pipeline sources.

Run the v3 notebook or CLI with this directory as its working directory so its
`scripts/` references select these dataset modules. Notebook generators resolve
their sources relative to their own file and can be called from the project root:

```bash
# Rebuild the v2 notebook.
python dev/pretraining/standard/old/build_nb.py

# Rebuild the supplied modular v3 notebook.
python dev/pretraining/standard/old/build_nb_runner.py
```

`build_nb_1930s.py` and `build_nb_runner.py` both write
`think_dataset_clean_1930s.ipynb`. Choose the generator matching the implementation
you intend to use.

| Historical ID in the scripts | Current dataset |
|---|---|
| `jbduran/think-dataset` | [bartholomew-dataset-v1](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v1) |
| `jbduran/think-dataset-clean` | [bartholomew-dataset-v2](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v2) |
| `jbduran/think-dataset-clean-1930s` | [bartholomew-dataset-v3](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3) |

Configure source/destination repos, prefixes, and work directories before a build.
For the modular scripts, the main controls are `SRC_REPO`, `DST_REPO`, `SRC_PREFIX`,
`STRIP_SRC_REPO`, `STRIP_PREFIX`, and `WORK_DIR` in `scripts/config.py`. The v1
builder and historical notebooks retain their original upload defaults.

## Source provenance

The notebook sources come from [OwenVoorhees/bart-dataset-scripts](https://github.com/OwenVoorhees/bart-dataset-scripts).
The `scripts/` files are the versions published with the v3 dataset. In particular,
`scripts/build_list.py` and `scripts/filter_lib.py` differ from the checked GitHub
versions, so their published versions are retained here. Exact revisions and file
checksums are in [../../SCRIPT_SOURCES.json](../../../SCRIPT_SOURCES.json).
