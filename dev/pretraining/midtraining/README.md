# Midtraining corpus and mixtures

| Stage | Notebook / scripts | Purpose |
|---|---|---|
| 1-2 | [midtraining_extraction.ipynb](midtraining_extraction.ipynb) | Extract STEM subjects from the v1 corpus; harvest, deduplicate, and filter Internet Archive books |
| 3 | [midtraining_ocroscope_filtering.ipynb](midtraining_ocroscope_filtering.ipynb) | OCRoscope quality and alphanumeric-ratio filtering |
| 4-5 | [midtraining_stage4_cleaning.ipynb](midtraining_stage4_cleaning.ipynb), `scripts/` | Structural cleanup, log-prior filtering, footer stripping, and tiered anachronism filtering |
| Mixtures, current | [midtraining_mix_repair.ipynb](midtraining_mix_repair.ipynb) | Token-weighted 21% and 45% mixtures |
| Mixtures, historical | [midtraining_mix_ratios.ipynb](midtraining_mix_ratios.ipynb) | Superseded mixtures built using document counts |
| Anneal experiments | [midtraining_anneal_ablation.ipynb](midtraining_anneal_ablation.ipynb), `configs/`, `d32-branch-sample-config.json` | Historical anneal and mixture-ratio experiments |

`scripts/` contains the midtrain-specific `build_list`, `filter_lib`, `footer_lib`, and
`structural_clean` modules. These variants take notebook configuration directly, so
keep them beside these notebooks. The stage 4-5 notebook expects `Path("scripts")`
relative to its working directory; start it from `dev/pretraining/midtraining` or set
`SCRIPTS_DIR` to that directory explicitly.

The anneal notebook runs against a full nanochat checkout. The current training and
evaluation entrypoints remain in the project's top-level `scripts/`. CORE evaluation
notebooks from the same source are in [../../evaluation/colab](../../evaluation/colab).

## Published dataset layout

The scripts retain their original `zachnorton03/think-midtrain` ID and historical
folder names. The current dataset is
[zachnorton03/bart-midtrain](https://huggingface.co/datasets/zachnorton03/bart-midtrain).
Its layout has changed as well as its name:

| Historical folder | Current published folder |
|---|---|
| `stage2/data`, `stage3/data`, `stage4/data` | `pipeline/stage2/data`, `pipeline/stage3/data`, `pipeline/stage4/data` |
| `stage5/data` | `corpus/` |
| Stage 5 audit artifacts | `pipeline/stage5/` |
| `mixed/v2/ratio_21/data`, `mixed/v2/ratio_45/data` | `mixtures/v2-by-tokens/ratio_21/data`, `mixtures/v2-by-tokens/ratio_45/data` |
| Earlier document-count mixtures | `mixtures/v1-by-docs/` |

Set repository IDs, source/destination prefixes, tokenizer locations, and local paths
for the intended build before executing a notebook. The token-repair notebook also
pins historical corpus/tokenizer revisions; keep those aligned with the intended run.

See [UPSTREAM_README.md](UPSTREAM_README.md), [LICENSE](LICENSE), and
[../../SCRIPT_SOURCES.json](../../SCRIPT_SOURCES.json) for source documentation and provenance.
