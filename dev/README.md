# Development scripts

Dataset preparation is grouped by the stage that consumes its output.

| Folder | Contents |
|---|---|
| [pretraining/standard](pretraining/standard/README.md) | Institutional Books -> dataset v1 -> v2 -> v3 |
| [pretraining/midtraining](pretraining/midtraining/README.md) | STEM extraction and harvest, OCR filtering, cleaning, and training mixtures |
| [sft](sft/README.md) | Authentic Q&A extraction, synthetic routes, robustness rows, and register-filter support |
| [evaluation](evaluation/README.md) | Vintage-CORE dataset construction and evaluation notebooks |
| `hosting/`, `providers/` | Deployment and training-machine setup |
| `archive/` | Superseded full training pipeline notebook |

General experiment, chat, tokenizer, scaling, and cost-estimation notebooks remain in
this directory. Training entrypoints remain in the repository's top-level `scripts/`.

## Sources

[SCRIPT_SOURCES.json](SCRIPT_SOURCES.json) records each imported file's original path,
repository revision, upstream checksum, and local checksum, plus the Hugging Face
repositories checked. Files were retrieved on 2026-10-27. Saved outputs and execution
counts were cleared from imported notebooks; their code and configuration are preserved.

The standard v3 modules come directly from
[jbduran/bartholomew-dataset-v3](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3/tree/main/scripts).
The v1 builder was already present locally. Dataset v1, v2, and
[bart-midtrain](https://huggingface.co/datasets/zachnorton03/bart-midtrain) do not currently
publish script files, so the remaining pipelines come from the sources listed in this
project or linked by the dataset cards:

| Source | Local home |
|---|---|
| [OwenVoorhees/bart-dataset-scripts](https://github.com/OwenVoorhees/bart-dataset-scripts) | `pretraining/standard/` |
| [OwenVoorhees/bart-midtrain-scripts](https://github.com/OwenVoorhees/bart-midtrain-scripts) | `pretraining/midtraining/`; CORE notebooks in `evaluation/colab/` |
| [zachnorton14/sft-extractor](https://github.com/zachnorton14/sft-extractor) | `sft/sft_extractor/` |
| [johnny0595/vintage-core](https://github.com/johnny0595/vintage-core) | `evaluation/vintage_core/` |

The related authentic, synthetic, robustness, Vintage-CORE, Vintage-GSM8K,
Vintage-GSM8K-filtered, HISTORY-EVENT, and Gutenberg dataset repositories were also
checked. Their current revisions do not contain standalone scripts. The SFT and
Vintage-CORE sources above provide the available related builders; no separate
GSM8K or HISTORY-EVENT builder was found in those dataset repositories.

These are source snapshots. Dataset shards, source books, model weights, generated
results, logs, caches, and credentials are excluded. Existing training/evaluation
entrypoints and duplicated copies of the standard cleaning pipeline in the midtrain
source were not imported again. Upstream documentation and available licenses are
retained alongside each snapshot.

## Historical configuration

Imported sources retain their original `think-*` dataset names, Colab paths, and output
settings. Some repositories have since been renamed and their internal folders moved.
Review the stage README and configuration before building a corpus or uploading data.
The reorganization and source checks do not run dataset builds, model API calls, or uploads.
