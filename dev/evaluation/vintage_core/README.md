# Vintage-CORE source snapshot

This folder contains the benchmark builder and evaluation package linked by the
[Vintage-CORE dataset card](https://huggingface.co/datasets/jbduran/vintage-core).

| Path | Purpose |
|---|---|
| `src/vintage_core/` | Temporal filtering, restyling, regeneration/repair, repackaging, validation, and evaluation |
| `notebooks/Vintage_CORE_Eval.ipynb` | Upstream evaluation notebook |
| `pyproject.toml` | Package dependencies and `vintage-core-validate` / `vintage-core-eval` entrypoints |
| [docs/BUILDING.md](docs/BUILDING.md) | Upstream build instructions |
| [DATA_PROVENANCE.md](DATA_PROVENANCE.md) | Benchmark data provenance |

Install this subproject separately when needed, for example with
`python -m pip install -e dev/evaluation/vintage_core` from the project root.
The original evaluation bundles and build outputs are external inputs and are not
included in this source snapshot. Configuration lives in `src/vintage_core/config.py`.

See [UPSTREAM_README.md](UPSTREAM_README.md), [LICENSE](LICENSE), [NOTICE](NOTICE),
and [../../SCRIPT_SOURCES.json](../../SCRIPT_SOURCES.json) for upstream documentation,
attribution, and the pinned revision.
