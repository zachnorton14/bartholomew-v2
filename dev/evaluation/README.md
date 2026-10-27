# Evaluation and benchmark preparation

| Path | Purpose |
|---|---|
| `colab/Vintage_CORE_Reference_Models_Durable.ipynb` | Existing reference-model evaluation notebook, moved from `dev/vintage_core_colab/` |
| `colab/Vintage_CORE_Eval*.ipynb` | Historical evaluation notebooks from the midtraining source repository |
| [vintage_core](vintage_core/README.md) | Vintage-CORE benchmark preparation and evaluation package |

The [jbduran/vintage-core](https://huggingface.co/datasets/jbduran/vintage-core) dataset
card links the builder at [johnny0595/vintage-core](https://github.com/johnny0595/vintage-core).
Its source is included separately from the training datasets. The current
Vintage-GSM8K, Vintage-GSM8K-filtered, and HISTORY-EVENT dataset repositories were
also checked; they do not publish standalone builder scripts at those revisions.

Benchmark datasets, generated bundles, checkpoints, and saved notebook outputs are
excluded from the imported snapshot. Existing runtime evaluation scripts remain in
the project's top-level `scripts/`. See [../SCRIPT_SOURCES.json](../SCRIPT_SOURCES.json)
for exact source revisions.

The historical Colab notebooks clone a separate evaluator branch or pinned commit.
Their `dev/vintage_core_colab` references address that external checkout, so those
paths are preserved even though the local notebooks have moved. For the current
builder package, use `vintage_core/` and its documented dependencies.
