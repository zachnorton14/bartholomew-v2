# SFT dataset preparation

These scripts prepare instruction-tuning data, so they live separately from pretraining.

| Path | Purpose |
|---|---|
| [gen_synthetic_data.py](gen_synthetic_data.py) | Original nanochat identity-conversation generation example, moved from `dev/` |
| `sft_extractor/run.py`, `sft_extractor/authentic/` | Extract, repair OCR, pair, enrich, and filter authentic period Q&A |
| `sft_extractor/synth/`, `sft_extractor/run_curricula.py` | Harvest excerpts, generate task routes, verify/grade/filter rows, construct curricula, and publish |
| `sft_extractor/synth/robustness/` | Mine pools and build typo, era, conversation, and malformed-input examples |
| `sft_extractor/encoder/` | Register-detection support and related encoder experiments |

The imported source is [zachnorton14/sft-extractor](https://github.com/zachnorton14/sft-extractor),
linked from [authentic-pre1930-sft-conversational](https://huggingface.co/datasets/zachnorton03/authentic-pre1930-sft-conversational).
It also contains the synthetic and robustness pipelines corresponding to
[synthetic-pre1930-sft](https://huggingface.co/datasets/zachnorton03/synthetic-pre1930-sft)
and [vintage-sft-robustness](https://huggingface.co/datasets/zachnorton03/vintage-sft-robustness).
The dataset repos themselves contain no script files at the checked revisions.

Run the imported CLI from `dev/sft/sft_extractor`, keeping `authentic/`, `synth/`, and
`encoder/` together so their package imports resolve. For example,
`python run.py --help` lists the available commands. Dependencies, source book inputs
(`authentic/data/*.txt`), generated excerpt pools, model weights, and API credentials
must be supplied separately. Small robustness pool JSON files used directly by the
builders are included. Raw corpora and generated examples are not copied.

The original identity generator still expects `knowledge/self_knowledge.md` at the
project root and `OPENROUTER_API_KEY`. Its relative knowledge path was adjusted for
the move; from the project root it can be invoked as `python -m dev.sft.gen_synthetic_data`.

See [sft_extractor/UPSTREAM_README.md](sft_extractor/UPSTREAM_README.md) and
[../SCRIPT_SOURCES.json](../SCRIPT_SOURCES.json) for the pipeline details and source revisions.
