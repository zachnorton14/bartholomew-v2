# Vintage CORE

Vintage CORE is a period-aligned version of the DataComp-LM CORE benchmark for
language models with a 1930 knowledge cutoff. It keeps CORE's likelihood-based
evaluation mechanics while reducing temporal mismatch from modern facts,
institutions, technologies, and prose.

The v1.0.0 release contains two aligned 20-task bundles:

- **Filtered** removes examples that require post-1930 knowledge and restores
  reviewed small tasks to their original sizes.
- **Restyled** preserves the filtered bundle's task configuration, row order,
  choices, answers, and protected continuations while recasting eligible prose
  toward an 1800–1930 register.

The exact final bundles are committed under [`data/filtered`](data/filtered) and
[`data/restyled`](data/restyled), and distributed through the
[`jbduran/vintage-core`](https://huggingface.co/datasets/jbduran/vintage-core)
Hugging Face dataset. No evaluation rows were added or regenerated while
extracting this standalone repository from the original experiment branch.

GitHub is the source of truth for code, documentation, tests, and the offline
data mirror. Hugging Face is the source of truth for downloading the versioned
dataset release. The v1.0.0 payloads are identical; `DATA_MANIFEST.sha256` here
maps directly to `SHA256SUMS` there after removing GitHub's `data/` prefix.

## Results

The experiment compares all bundles over **Common-20**, the 20 task labels shared
by Original CORE and both Vintage CORE variants.

| Model | Original | Filtered | Restyled | Filtered − Original | Restyled − Filtered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Think Unbounded d32 | 16.3% | 17.5% | 17.6% | +1.2 points | +0.1 points |
| Modern nanochat d24 | 26.3% | 26.2% | 25.3% | −0.1 points | −0.9 points |
| GPT-1900 d34 | 12.1% | 12.7% | 13.9% | +0.6 points | +1.2 points |

These are the recorded full-benchmark results, not newly generated measurements.
See the [final report](docs/FINAL_REPORT.md) for the protocol, per-task analysis,
construction method, costs, limitations, and source record.

## Install

[uv](https://docs.astral.sh/uv/) is the recommended environment manager. Python
3.10 or newer is required.

```bash
git clone https://github.com/johnny0595/vintage-core.git
cd vintage-core
uv sync --group dev
```

For checkpoint evaluation, install the GPU dependencies as well:

```bash
uv sync --extra eval --group dev
```

Copy `.env.example` to `.env` only when credentials are needed. Validation of the
committed bundles is fully offline and does not require an API key.

To download only the versioned dataset rather than cloning the code repository:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jbduran/vintage-core",
    repo_type="dataset",
    revision="v1.0.0",
    allow_patterns=["filtered/**", "restyled/**", "SHA256SUMS"],
    local_dir="vintage-core-v1.0.0",
)
```

## Validate the release

Run the strict paired-bundle release gate:

```bash
uv run vintage-core-validate
```

The expected final line is:

```text
validated_rows=47092 issues=0
```

Verify every committed data file against the extraction manifest:

```bash
shasum -a 256 -c DATA_MANIFEST.sha256
```

Run the test suite:

```bash
uv run pytest -q
```

## Evaluate a model

The evaluator reproduces nanochat's CORE prompt rendering and scoring rules. It
uses the committed Vintage bundles locally, downloads Original CORE, and fetches a
pinned model checkpoint plus its matching runtime. CUDA is required for the
registered checkpoints.

A short smoke run on a registered model:

```bash
uv run vintage-core-eval \
  --model modern-d24 \
  --bundles original,filtered,restyled \
  --max-per-task 10 \
  --output-dir results/modern-d24-smoke
```

A full run evaluates every row and should be performed on an A100-class GPU:

```bash
uv run vintage-core-eval \
  --model modern-d24 \
  --bundles original,filtered,restyled \
  --max-per-task -1 \
  --output-dir results/modern-d24
```

Available model IDs are `think-d12-r30`, `modern-d24`, and `gpt1900-d34`.
Some model artifacts may require `HF_TOKEN`; the GPT-1900 d34 run requires an
A100-class GPU. Results are written incrementally as JSON, followed by
`summary.csv`, `task_accuracy.csv`, and `task_deltas.csv`.

For a hosted workflow, open
[`notebooks/Vintage_CORE_Eval.ipynb`](notebooks/Vintage_CORE_Eval.ipynb) in
Google Colab.

## Repository map

```text
data/
  filtered/                 immutable filtered v1.0.0 bundle
  restyled/                 canonical restyled v1.0.0 bundle
docs/
  FINAL_REPORT.md           complete experiment report and final results
  BENCHMARK_AUDIT.md        task-by-task temporal audit
  BUILD_LOG.md              filtering, backfill, and packaging record
  audits/                   compact human and deterministic audit summaries
src/vintage_core/
  bundle_validation.py      offline paired-bundle release gate
  evaluate.py               pinned model evaluator
  *.py, repairs/            archived construction and repair pipeline
tests/                      validation, evaluator, and transformation tests
```

The construction pipeline is preserved for research transparency. It is not
needed to consume or validate the release and can invoke external LLM APIs when
generation flags are used. See [Building and provenance](docs/BUILDING.md) before
running it.

## Data and licensing

Code and original project documentation are released under the [MIT License](LICENSE).
The committed benchmark rows are adaptations of third-party evaluation datasets
and remain subject to their applicable source terms; MIT does not relicense those
underlying datasets. Some incorporated tasks do not have a blanket redistribution
grant documented by this project; public availability is not a representation that
all dataset content is MIT-licensed. See [Data provenance](DATA_PROVENANCE.md) and
[NOTICE](NOTICE).

Vintage CORE was developed by Jonathan Duran-Ortiz at Unbounded Labs. The
standalone evaluator includes behavior adapted from Andrej Karpathy's nanochat,
and the benchmark is based on the DataComp-LM CORE suite.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite DataComp-LM and the original
evaluation datasets used by CORE when publishing results.
