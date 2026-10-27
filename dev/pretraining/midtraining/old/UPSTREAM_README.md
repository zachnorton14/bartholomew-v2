# BART Midtrain Scripts

The pipeline that built the **BART** midtraining corpus — pre-1930 math, science, technology and
medicine — plus the notebooks that blended it into training mixtures and ran the anneal.

Output: [`zachnorton03/bart-midtrain`](https://huggingface.co/datasets/zachnorton03/bart-midtrain)

Built by [Unbounded Labs](https://unboundedlab.com).

> **Filenames still say `think`/`midtraining`.** These predate the dataset rename and were left
> alone on purpose so existing manifests keep matching. The tables below map each one to what it
> actually produces.

## What midtraining is

Midtraining sits between pretraining and downstream use. Rather than continuing on the same broad
corpus, the data mixture shifts toward a smaller, higher-quality set of documents while the
learning rate decays — spending the model's final optimization steps on the text we most want it to
internalize.

## Corpus pipeline

| Notebook | Stage | Writes to `bart-midtrain` |
|---|---|---|
| `midtraining_extraction.ipynb` | 1–2 | `pipeline/stage2/` |
| `midtraining_ocroscope_filtering.ipynb` | 3 | `pipeline/stage3/` |
| `midtraining_stage4_cleaning.ipynb` | 4–5 | `pipeline/stage4/`, `pipeline/stage5/`, `corpus/` |

**Stages 1–2.** Extracts STEM documents by subject tag from
[`bart-dataset-v1`](https://huggingface.co/datasets/jbduran/bart-dataset-v1), then harvests vintage
documents from the Internet Archive, filtering by topic (Math, Science, Medicine, Technology) and
published date. De-dupes and applies an OCR filter of 0.85. → 15,075 documents

**Stage 3.** A second opinion on OCR quality using OCRoscope (0.85) plus an alphanumeric ratio of
0.65 — kept relaxed on purpose, since math documents carry many unusual symbols. Removes ~10%.
→ 13,552 documents

**Stages 4–5.** Filters OCR artifacts and strips boilerplate (headers, library stamps), mirroring
the main corpus with a log-prior filter at p2.5–p97.5. Then makes the corpus as vintage as possible
by running the footer filter and the tiered banned-words list — the same approach as
[bart-dataset-scripts](https://github.com/OwenVoorhees/bart-dataset-scripts), which we reused here
extensively. → 12,791, then **11,409 documents**

After all cleaning, 24% of the original midtraining data was removed, leaving **~608M tokens**.

## Mixtures

Training ran in three stages of decay, following HuggingFace's SmolLM. Because midtrain documents
are much shorter than pretraining documents, mixtures are measured in **tokens**, not documents,
targeting fewer than 3 epochs over the midtrain data.

| Notebook | Produces | Status |
|---|---|---|
| `midtraining_mix_ratios.ipynb` | `mixtures/v1-by-docs/` | **superseded** — ratios by document count |
| `midtraining_mix_repair.ipynb` | `mixtures/v2-by-tokens/` | current — ratios by token count |

`midtraining_mix_ratios.ipynb` set its ratios by document count, which badly under-delivered
midtrain content: a blend targeting 60% supplied 25%, one targeting 30% supplied 11.5%.
`midtraining_mix_repair.ipynb` is the rebuild that measures by token instead. Use it.

## Everything else

| Path | Purpose |
|---|---|
| `midtraining_anneal_ablation.ipynb` | anneal / mixture-ratio ablation |
| `Vintage_CORE_Eval*.ipynb` | CORE evaluation on the vintage models |
| `nanochatScripts/` | nanochat training and eval scripts (`base_train`, `tok_*`, `chat_*`, `anachronism_eval`) |
| `scripts/` | `build_list`, `filter_lib`, `footer_lib`, `structural_clean` |
| `configs/` | anneal configs (`anneal_d24_ratio0`, `ratio60`, ablation sample) |
| `running info/` | Colab runner notebook + a copy of `nanochatScripts/` |
| `cleaning info/` | vendored copy of [bart-dataset-scripts](https://github.com/OwenVoorhees/bart-dataset-scripts) |
| `d24 run data/`, `results/`, `celllogs*.txt` | run logs and eval output |

Learning rate schedule: nanochat's Warmup-Stable-Decay (WSD), starting decay at roughly 35% of
training — SmolLM starts around 75%.

## Related

[bart-dataset-v1](https://huggingface.co/datasets/jbduran/bart-dataset-v1) ·
[v2](https://huggingface.co/datasets/jbduran/bart-dataset-v2) ·
[v3](https://huggingface.co/datasets/jbduran/bart-dataset-v3) ·
[bart-midtrain](https://huggingface.co/datasets/zachnorton03/bart-midtrain) ·
[bart-dataset-scripts](https://github.com/OwenVoorhees/bart-dataset-scripts)

## License

MIT — see [LICENSE](LICENSE).
