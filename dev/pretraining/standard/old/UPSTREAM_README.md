# BART Dataset Scripts

The cleaning pipeline behind the **BART** pretraining corpus — pre-1930 English books filtered out
of [Institutional Books 1.0](https://huggingface.co/datasets/institutional/institutional-books-hl).

Built by [Unbounded Labs](https://unboundedlab.com).

> **Filenames still say `think`.** These scripts predate the dataset rename and their names were
> deliberately left alone so existing notebook outputs and manifests keep matching. The table below
> maps each one to the dataset it actually produces.

## What produces what

| Script | Reads | Writes |
|---|---|---|
| `think_dataset_clean.ipynb` | [`bart-dataset-v1`](https://huggingface.co/datasets/jbduran/bart-dataset-v1) | [**`bart-dataset-v2`**](https://huggingface.co/datasets/jbduran/bart-dataset-v2) |
| `think_dataset_clean_1930s.ipynb` | [`bart-dataset-v2`](https://huggingface.co/datasets/jbduran/bart-dataset-v2) | [**`bart-dataset-v3`**](https://huggingface.co/datasets/jbduran/bart-dataset-v3) |

Both notebooks are **generated**, not edited directly — see [Layout](#layout).

### v1 → v2 — boilerplate and OCR corruption

`think_dataset_clean.ipynb` targets boilerplate and OCR corruption that got past Institutional
Books' own filters: useless headers, footers and library stamps, alongside gibberish symbols and
antiquated characters, through heuristic regex filters. It then applies Michael Hla's log-prior
filter, which estimates a sequence's unconditional log-prior probability by summing token-level log
probabilities and trims improbable or out-of-distribution tokens, keeping a p2.5–p97.5 band.

We expected a major improvement in validation bpb. Performance actually *declined* — this phase
removed most of the easy items for a model to replicate, i.e. the boilerplate.

### v2 → v3 — the 1930 cutoff

`think_dataset_clean_1930s.ipynb` focuses on time period. Earlier stages filtered by date, but
leaks and bad labeling still let modern documents contaminate the corpus. It strips missed footers,
then applies a **regex anachronism filter**: a tiered banned-words list that flags documents and
removes them on too many hits.

Rather than trimming, entire documents are removed when the conditions are met. One Tier 1 hit
discards a document; Tier 2 words require two occurrences; Tier 3 words never trigger removal
alone, only counting alongside at least one Tier 2 hit. The same strategy was later reused for
midtraining — see [bart-midtrain-scripts](https://github.com/OwenVoorhees/bart-midtrain-scripts).

## Layout

| Path | Purpose | Dataset |
|---|---|---|
| `build_nb.py` | builds `think_dataset_clean.ipynb` from `cells/` | v2 |
| `build_nb_1930s.py` | builds `think_dataset_clean_1930s.ipynb` from `cells_1930s/` | v3 |
| `build_nb_runner.py` | builds the runner notebook from `cells_runner/` | v3 |
| `cells/` | notebook cells: config, clean library, log-prior, main loop, report | v2 |
| `cells_1930s/` | notebook cells: banned-list build, filter, main loop, report | v3 |
| `cells_runner/` | runner cells: footer stripping, filter, batch run, report | v3 |
| `scripts/` | the shared library the cells import | both |

`scripts/` holds the real logic: `build_list.py` (tiered banned-words list), `filter_lib.py`
(anachronism filter), `footer_lib.py` + `strip_footers.py` (footer and boilerplate stripping),
`run_filter.py` (driver), plus `common.py`, `config.py`, `report.py`, `wipe.py`.

## Not in this repo

The **IB → v1** step — OCR score ≥ 0.90 across both OCRoscope and Google Books metadata (agreeing
within 0.10), date filtering with undated books rejected, English filtering, and a tokenizability
threshold of 95, which trimmed 242B tokens to 27B before barcode dedup and shuffling — is not part
of this repository. Its output is the starting point here.

## Note on dataset ids in code

`cells/cell2_config.py`, `cells_1930s/cell2_config.py` and `scripts/config.py` still hardcode the
pre-rename ids (`jbduran/think-dataset`, `-clean`, `-clean-1930s`). Those resolve fine — the Hub
redirects renamed repos — but they no longer match the current names.

## License

MIT — see [LICENSE](LICENSE).
