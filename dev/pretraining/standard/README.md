# Standard pretraining corpus

| Stage | Scripts / notebooks | Output |
|---|---|---|
| Institutional Books -> v1 | [repackage_institutional_books.py](repackage_institutional_books.py) | OCR-, language-, date-, and tokenizability-filtered book shards |
| v1 -> v2 | [think_dataset_clean.ipynb](think_dataset_clean.ipynb); sources in `cells/`; generator [build_nb.py](build_nb.py) | Structural/boilerplate cleanup and GPT-2 log-prior filtering |
| v2 -> v3 | [think_dataset_clean_1930s.ipynb](think_dataset_clean_1930s.ipynb); sources in `cells_1930s/` and `cells_runner/` | Footer stripping and tiered anachronism filtering |
| v3 shared modules | `scripts/build_list.py`, `strip_footers.py`, `run_filter.py`, `report.py`, `wipe.py`, and supporting libraries | Modular list-building, stripping, filtering, reporting, and reset stages |
| Original nanochat reference | [repackage_data_reference.py](repackage_data_reference.py) | Historical FineWeb/ClimbMix preparation example |

The original v1 contract and audit are in [DATASET_PIPELINE.md](DATASET_PIPELINE.md)
and [FILTER_AUDIT.md](FILTER_AUDIT.md). The reference repackager uses modern corpora;
it documents the sharding approach and is not part of the historical book corpus.

## Notebook sources

`cells/`, `cells_1930s/`, and `cells_runner/` contain notebook cell fragments. Some use
IPython `%pip` or `!` syntax and must run inside a notebook, not as standalone Python files.
The generators resolve their cell directories relative to their own file, so they can
be called from the project root:

```bash
python dev/pretraining/standard/build_nb.py
python dev/pretraining/standard/build_nb_1930s.py
```

`build_nb_runner.py` builds the modular runner from `cells_runner/`. It writes the same
`think_dataset_clean_1930s.ipynb` filename as `build_nb_1930s.py`; choose the desired
version before regenerating that notebook.

The modular runner expects this folder's `scripts/` directory beside the notebook.
For direct CLI use, run from `dev/pretraining/standard` so commands such as
`python scripts/report.py` refer to these dataset modules.

## Dataset names

| Historical ID in imported code | Current dataset |
|---|---|
| `jbduran/think-dataset` | [jbduran/bartholomew-dataset-v1](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v1) |
| `jbduran/think-dataset-clean` | [jbduran/bartholomew-dataset-v2](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v2) |
| `jbduran/think-dataset-clean-1930s` | [jbduran/bartholomew-dataset-v3](https://huggingface.co/datasets/jbduran/bartholomew-dataset-v3) |

The shared modules use environment variables defined in `scripts/config.py`, including
`SRC_REPO`, `DST_REPO`, `SRC_PREFIX`, `STRIP_SRC_REPO`, `STRIP_PREFIX`, and `WORK_DIR`.
The v1 builder's historical destination defaults and the notebooks' upload settings are
preserved. Configure the intended destination before running a build.

See [UPSTREAM_README.md](UPSTREAM_README.md), [LICENSE](LICENSE), and
[../../SCRIPT_SOURCES.json](../../SCRIPT_SOURCES.json) for source documentation and provenance.
The published v3 versions of `scripts/build_list.py` and `scripts/filter_lib.py` differ
from the checked GitHub versions; this snapshot keeps the modules distributed with
the dataset itself.
