# Building and construction code

The committed `data/filtered` and `data/restyled` directories are the final
Vintage CORE v1.0.0 release. Consuming, validating, or evaluating those bundles
does not require rerunning the construction pipeline.

The modules under `src/vintage_core` preserve the experiment's filtering,
backfill, restyle, repair, reconciliation, and packaging code for inspection and
reproducibility. Their defaults place intermediate work under
`~/.cache/vintage-core/build`, keeping tracked release data immutable.

## Prerequisites

Download and extract Original CORE to `~/.cache/nanochat/eval_bundle`, or set
`VINTAGE_CORE_ORIGINAL_DIR` to an equivalent directory containing `core.yaml`,
`eval_meta_data.csv`, and `eval_data/`.

```bash
uv sync --group dev
cp .env.example .env
```

LLM-assisted stages read the environment-variable name configured by
`VINTAGE_LLM_API_KEY_ENV`, which defaults to `OPENCODE_API_KEY`. Offline
validation, deterministic repair checks, and tests do not use that credential.

## Pipeline record

The authoritative sequence and decisions are documented in:

- [`PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md)
- [`BUILD_LOG.md`](BUILD_LOG.md)
- [`FINAL_REPORT.md`](FINAL_REPORT.md)
- [`audits/`](audits/)

Representative module entry points include:

```bash
uv run python -m vintage_core.filter --help
uv run python -m vintage_core.backfill --help
uv run python -m vintage_core.regenerate --help
uv run python -m vintage_core.restyle --help
uv run python -m vintage_core.bundle_validation --help
```

Generation is never implicit in the documented release-validation workflow.
Several construction modules require an explicit `--generate`, `--commit`, or
stage/apply subcommand before they call an external model or write accepted
changes. Read the module help and the pipeline design before using them.
