# sft-extractor

Builds supervised fine-tuning (SFT) datasets of pre-1930s question-answer pairs. There are two independent pipelines:

- **`authentic/`** — *extraction.* Parses real Q&A out of public-domain textbooks and catechisms (Internet Archive OCR) into clean conversations. Every word is genuinely period.
- **`synth/`** — *generation.* Harvests excerpts from a pre-1930s pretraining corpus, routes them by task type, and generates new Q&A grounded in that period text. Scales far beyond the ~12k authentic pairs while keeping answers verbatim-period.
- **`encoder/`** — experiments in *register detection* (Talkie-13B embeddings, stylometry) used to probe whether generated questions can be filtered for structural anachronism. See [Register filtering](#register--anachronism-filtering-encoder).

Both pipelines run through `run.py` and share the same DeepSeek/OpenCode model backend.

## Authentic pipeline — extraction (`authentic/`)

```
authentic/data/*.txt
  → run.py extract   → authentic/output/extracted/*.json   (19,284 pairs)
  → run.py bleed     → authentic/output/bleed/*.json        (19,473 pairs)
  → run.py ocr       → authentic/output/ocr/*.json          (18,382 pairs)
  → run.py pair      → authentic/output/paired/*.json       (13,157 conversations)
  → run.py enrich    → authentic/output/enriched/*.json     (12,264 conversations)
  → run.py filter    → authentic/output/filtered/*.json     (12,257 conversations)
```

1. **Extract** — parse raw OCR text into Q&A pairs per dataset (regex-based, no model)
2. **Bleed** (pass 0) — detect and recover structural bleed via LLM; pairs where the Q field contains embedded answer text followed by more questions are split into constituent pairs (1→N), unrecoverable pairs are discarded
3. **OCR** (pass 1) — fix OCR scanning artifacts (broken hyphenation, character substitutions, encoding garbage, stray page headers) via LLM; pairs where any substitution could be an anachronism are flagged for review
4. **Pair** (pass 2) — group dependent Q&A pairs into multi-turn conversations; a sliding window model detects whether each question depends on the prior pair (pronoun reference, demonstratives, quoted terms); chains are reconstructed algorithmically
5. **Enrich** (pass 3) — detect context-bare and domain-ambiguous opening questions; bare follow-ups are attached to their predecessor; ambiguous openers are rewritten with minimal domain context
6. **Filter** (pass 4) — rule-based removal of residual garbage: music notation OCR placeholders, fingering chart artifacts, and bleed remnants (7 pairs removed, 0.06%)

## Final dataset statistics

| Metric | Count |
|--------|-------|
| Total conversations | 12,257 |
| Single-turn | 9,103 |
| Multi-turn | 3,154 |
| Enriched (question rewritten) | 147 |
| Datasets | 27 |

**Multi-turn conversation length distribution:**

| Turns | Conversations |
|-------|--------------|
| 2 | 1,887 |
| 3 | 640 |
| 4 | 282 |
| 5 | 115 |
| 6–10 | 199 |
| 11+ | 31 |

## Dataset breakdown

| Dataset | Conversations | Multi-turn | Enriched |
|---------|--------------|------------|---------|
| 1001-questions | 592 | 240 | 6 |
| advanced-questions | 547 | 71 | 1 |
| agriculture | 213 | 57 | 7 |
| astronomy | 331 | 121 | 5 |
| botany | 136 | 51 | 1 |
| brewers-guide | 1,216 | 289 | 9 |
| chemistry | 712 | 177 | 9 |
| civil-war | 1,198 | 314 | 0 |
| common-core | 2,194 | 425 | 0 |
| constitution | 116 | 52 | 3 |
| electricity | 629 | 139 | 0 |
| engineering | 659 | 229 | 0 |
| ethics | 178 | 66 | 1 |
| familiar-things | 385 | 174 | 3 |
| grammar | 251 | 163 | 0 |
| investors | 400 | 2 | 0 |
| laborers | 110 | 73 | 2 |
| logic | 145 | 82 | 15 |
| music | 128 | 38 | 16 |
| mythology | 301 | 148 | 4 |
| new-york-bar | 620 | 1 | 2 |
| patriotism | 7 | 6 | 0 |
| school-bulletin | 352 | 43 | 10 |
| seeleys | 726 | 142 | 48 |
| stokers | 30 | 10 | 0 |
| symbological | 10 | 7 | 0 |
| world-history | 71 | 34 | 5 |

## Example conversations

These examples illustrate the dataset's range, authenticity, and the value of multi-turn chaining.

---

**Natural philosophy — trade winds** *(6 turns, brewers-guide)*
```
Q: Do winds never blow regularly?
A: Yes; in those parts of the world where these obstructions do not exist; as on the
   Atlantic and Pacific Ocean, the winds blow with great regularity.

Q: What are the winds which blow over the Atlantic and Pacific Ocean called?
A: They are called "Trade Winds."

Q: Why are they called trade winds?
A: Because (as they blow uniformly in one direction) they are very convenient to those
   who carry on trade by means of these winds.

Q: In what direction do the trade winds blow?
A: That in the northern hemisphere blows from the north-east; that in the southern
   hemisphere from the south-east.

Q: Why do they not blow from the full north and south?
A: Because the polar current, combining with the equatorial current, give the wind a
   new direction.

Q: What is the cause of the equatorial current?
A: The rotation of the earth upon its axis.
```

---

**History of astronomy** *(6 turns, astronomy)*
```
Q: Who are some noted astronomers of still later times?
A: Huyghens, Roemer, Newton, and William Herschel.

Q: What is to be noted of Huyghens?
A: Huyghens (1629–1695) proposed the wave-theory of light, and made the first
   pendulum clock.

Q: Who was Roemer?
A: Roemer, a Dane (1644–1710) is the inventor of the transit instrument; he likewise
   roughly determined the velocity of light.

Q: What is to be said of Newton?
A: Newton (1642–1727) discovered the law of universal gravitation and wrote a
   monumental work called the Principia.

Q: What did William Herschel do?
A: William Herschel (1738–1822) built several large reflecting telescopes; he also
   discovered the planet Uranus.

Q: Are there noted astronomers who lived later than the ones just mentioned?
A: Yes, there are a great many noted astronomers who lived after Herschel's time, or
   who are still living; some of their discoveries will be mentioned later.
```

---

**Medieval history — the Jacquerie** *(4 turns, 1001-questions)*
```
Q: What great battle was fought?
A: Battle of Poitiers (1356), in which the French were defeated by a small army of
   English archers.

Q: What insurrection burst forth at this time?
A: An insurrection of the peasantry caused by the misery in which they had been so
   long kept by the tyrannical nobles.

Q: What is it called in history?
A: The Jacquerie, from Jacques Bonhomme, a name derisively applied to a French peasant.

Q: How did it end?
A: After sacking the feudal castles the peasants were finally defeated in an attack
   upon one of the towns and massacred by the nobles.
```

---

**Moral philosophy** *(3 turns, ethics)*
```
Q: Are there various divisions of ends or purposes?
A: Yes. For, one purpose intended may be intended for another purpose; and this for a
   third; and the third for a fourth; and so on; until we arrive at one which is
   intended for its own sake only.

Q: What is the first one intended (the nearest) called?
A: The proximate end.

Q: What is the last one intended (the farthest off) called?
A: The ultimate end. If there be but one end intended, it is of course first and last,
   or proximate and ultimate.
```

---

**Everyday science** *(singleton, brewers-guide)*
```
Q: Why is beer flat, if the cask be open too long?
A: Because too much of the carbonic acid gas (produced by fermentation) is suffered
   to escape.
```

---

**Military history — soldier wages, 1917** *(2 turns, civil-war)*
```
Q: What wages do the soldiers of the belligerents receive per day?
A: Great Britain gives 1s. 2d. (29 cents); Germany, 5 cents; France, 3 cents;
   Canada, $1.12; New Zealand, $1.25; and Australia, $1.25.

Q: How would the daily army pay-bills of the nations compare?
A: That of Great Britain probably would be about six times that of Germany, while
   Australia appears to be paying every day for its soldiers about 25 times as much
   as Germany pays.
```

---

**Constitutional law** *(singleton, constitution)*
```
Q: Why is the 4th of July kept with such public rejoicing through all parts of the
   United States?
A: Because on the 4th of July 1776 the Colonies first declared themselves free and
   independent; from that day the independence of the country is reckoned in all our
   public proceedings; though it was not until the treaty of 1783 that Great Britain
   acknowledged the fact.
```

---

## Output format

Each file is a JSON array of conversation objects:

```json
[
  {
    "conversations": [
      { "role": "user",      "content": "What is the capital of France?" },
      { "role": "assistant", "content": "Paris." }
    ]
  }
]
```

Additional fields that may appear:
- `"chained": true` — multi-turn conversation (from pair or enrich pass)
- `"enriched": true` — opening question was rewritten for context
- `"recovered": true` — pair was split from a bleed
- `"flagged": true` — OCR correction may contain an anachronism

## Usage

All pipeline operations go through `run.py`:

```bash
python3 run.py <command> [options]
```

### Extract

```bash
python3 run.py extract <name> authentic/data/<file>.txt authentic/output/extracted/<file>.json
```

Available extractors: `advanced_questions`, `common_core`, `brewers_guide`, `familiar_things`, `1001_questions`, `logic`, `seeleys`, `stokers`, `symbological`, `agriculture`, `astronomy`, `botany`, `chemistry`, `civil_war`, `constitution`, `electricity`, `engineering`, `ethics`, `grammar`, `mythology`, `new_york_bar`, `patriotism`, `school_bulletin`, `investors`, `music`, `laborers`, `world_history`

### Bleed pass

```bash
python3 run.py bleed                        # full run (resumable)
python3 run.py bleed --test --count 10      # test sample
python3 run.py bleed --retry-bleed          # second pass on authentic/output/ocr/
```

### OCR pass

```bash
python3 run.py ocr                          # full run (resumable)
python3 run.py ocr --test --count 10        # test sample
python3 run.py ocr --retry-flagged          # reprocess flagged pairs
```

### Pair pass

```bash
python3 run.py pair                         # full run (resumable)
python3 run.py pair --test --count 10       # test sample
```

### Enrich pass

```bash
python3 run.py enrich                       # full run (resumable)
python3 run.py enrich --test --count 20     # test sample
```

### Filter pass

```bash
python3 run.py filter                       # run and write authentic/output/filtered/
python3 run.py filter --sample --count 20   # preview what would be dropped
```

### Sample any stage

```bash
python3 run.py sample filtered --count 20 --seed 42
```

Stages: `extracted`, `bleed`, `ocr`, `paired`, `enriched`, `filtered`

### Environment

Model passes (bleed, ocr, pair, enrich) use DeepSeek via the Anthropic SDK compatibility layer:

```bash
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_API_KEY=<your deepseek key>
```

All model passes are resumable — progress is saved to state files (`.bleed_state.json`, `.ocr_state.json`, `.pair_state.json`, `.enrich_state.json`) and re-running skips already-processed items.

## Domains

27 source texts covering: world history, American history, natural science, chemistry, astronomy, botany, agriculture, engineering, electricity, steam engineering, grammar, logic, ethics, mythology, music theory, investment law, labor economics, constitutional law, New York bar examination, American civics, biblical interpretation, and literature.

---

# Synthetic pipeline — generation (`synth/`)

Where the authentic pipeline is limited to what a few dozen scanned books happen to contain, the synthetic pipeline generates fresh Q&A from a large pre-1930s **pretraining corpus** (`jbduran/think-dataset` on Hugging Face). It scales to hundreds of thousands of pairs across many task types, while keeping the answers period-authentic by construction.

```
pretrain corpus (HF)
  → run.py harvest        → synth/output/excerpts.jsonl      (candidate passages + metadata)
  → run.py classify       → excerpts.jsonl (+ classes/primary) (model assigns task type)
  → run.py <route>        → synth/output/<route>/*.json        (grounded Q&A per task type)
```

1. **Harvest** — stream documents from the corpus and cut out candidate passages. A cheap regex **affordance gate** (`synth/corpus.py`) is a high-recall net that tags each excerpt with a coarse form (`expository`, `argument`, `narrative`, …). Writes `excerpts.jsonl`, one JSON record per passage with metadata (LoC subject, year, title, `prose_score`, word count). `filter-pool` removes OCR-garbled passages by an `ocr_score` threshold.
2. **Classify** — a model reads each excerpt and assigns one or more **route classes** from a fixed taxonomy (recall net → precision cut). Multi-label with a conviction bar; the highest-priority label becomes `primary`. Rare, specific classes claim an excerpt before elastic catch-alls, so a scarce class is never strip-mined by a general one. Priority order:

   `stem_reasoning → how_to → verse → conversational → reasoning → narrative_grounded → narrative_fiction → opinion → knowledge → composition → drop`

3. **Routes** — content-bound generators that turn typed excerpts into Q&A. All share one engine (`synth/engine.py`): sample excerpts → batch → one model call per batch → parse → verify → resumable state → write. A route is just a `Route` config (its system prompt, which classes it sources, how a model result becomes an answer).

## The anachronism guarantee (extractive answers)

The core design rule: **answers are verbatim spans of period text, never model-composed.** Each route lifts the answer as 1–2 exact quotations from the source excerpt (verified as literal substrings; spans join with an ellipsis). An answer built only from pre-1930s prose cannot contain a modern word or fact. Only the **question** is model-written (in period-schoolbook register, self-situating so it stands alone). This shifts the anachronism risk entirely onto the question phrasing, which is short and controllable — rather than onto the answer content.

## Routes

| Command | Route(s) | Sources classes | Question framing |
|---|---|---|---|
| `knowledge-qa` | knowledge_qa | knowledge | Expository fact → focused schoolbook question |
| `reasoning-qa` | reasoning_qa | reasoning, knowledge, … | "Why/how does it follow" from a reasoning chain |
| `stem-reasoning` | stem_reasoning | stem_reasoning | Quantitative/physical reasoning, **math verbalized** (see below) |
| `narrative-qa` | narrative_grounded, narrative_fiction | narrative_* | Retell/comprehend a real episode, or read a fictional scene |
| `composition-qa` | composition_qa | composition | Questions about a nameable prose form |
| `how-to-qa` | how_to_qa | how_to | Procedure / method questions |
| `opinion-qa` | opinion_qa | opinion | Argument / stance questions |
| `verse-qa` | verse_qa | verse | Questions on poetry/verse passages |
| `multiturn-qa` | multiturn_qa | knowledge | Chained follow-up questions (conversation) |
| `calibration-qa` | calibration_qa | knowledge, reasoning | Factual non-answer when the source cannot settle the point |

Each supports `--test` (print a small sample, no write), `--sample` (write a review dump under `synth/samples/`), `--size N`, and `--seed`.

## Register & anachronism filtering (`encoder/`)

Downstream filtering was investigated but **is not part of the shipping pipeline** — the extractive-answer design plus corpus OCR-cleaning already remove the bulk of anachronism. Two approaches were tested and their limits documented:

- **Per-token loss filter** (`synth/anachronism_filter.py`) — scores each question by `delta_peak`, a vintage nanochat model's per-word bits-per-byte minus GPT-2's. Catches lexical/OOV anachronism but **falsely flags STEM** (math notation is out-of-distribution for the vintage corpus); mitigated by `synth/notation_mask.py` (`--mask`) and by **verbalizing math** in the stem prompt. On the generated dataset it failed to yield a usable threshold (the true anachronism base rate is too low). This work was split to a separate effort.
- **Structural-register detection** (`encoder/`) — Talkie-13B sentence embeddings (`talkie_encoder.py`) and topic-blind **stylometry** (`stylometry.py`, `style_oneclass.py`) were used to test whether *structural* anachronism (modern syntax, all period-legal words) is detectable. Finding: the signal is genuinely faint on the two largest routes (knowledge, multiturn; register AUC ≈ 0.75 by any method), so an automated structural filter is not achievable there — the fix belongs on the generation side.

## Synthetic usage

```bash
# 1. harvest candidate passages (resumable; --stem / --verse for targeted sweeps)
python3 run.py harvest --total 100000
python3 run.py filter-pool --threshold 0.15 --apply     # drop OCR-garbled passages

# 2. classify excerpts into route types (resumable)
python3 run.py classify
python3 run.py classify --coverage                       # progress vs per-class targets

# 3. generate a route (resumable; --test / --sample to preview)
python3 run.py knowledge-qa --size 5000
python3 run.py stem-reasoning --sample
python3 run.py narrative-qa --size 5000
```

### Pair grading

After the alignment filter has written `verified/<route>/` on Hugging Face, grade the
surviving examples for training usefulness with a holistic 0--100 score:

```bash
python3 -m synth.grade --sample 100       # stats + review file; does not write HF output
python3 -m synth.grade --sample 100 --show 25
python3 -m synth.grade                    # full resumable pass
```

Sample reviews are also written as wrapped text files under `synth/samples/grade/`;
the file contains per-route clean/flagged/hard-cap statistics, the best structurally
clean candidates first, and then every sampled grading unit sorted from lowest to
highest. Any deterministic hard cap is shown beside the affected example.

The pass writes scored copies under `graded/<route>/`. Single-turn rows receive an
integer `score`; multi-turn rows receive one holistic `score` for the complete
conversation. Judgments checkpoint to `synth/state/grade_<route>.json`, and a route is
published only after every grading unit has a valid score. The judge is constrained to
return only a bare positional integer array such as `[87, 92, 76]`; object-shaped
responses and filler are rejected.

Model passes use OpenCode Go / DeepSeek; set `OPENCODE_API_KEY` (or `DS_API_KEY`) in the environment or `ROOT/.env`.

## Synthetic output format

Each route writes `synth/output/<route>/*.json`, rows carrying the Q&A plus source metadata:

```json
{
  "doc_index": "61634", "category": "SCIENCE", "year": 1871,
  "prose_score": 0.962,
  "question": "Why are the trade winds so called?",
  "answer": "Because ... they are very convenient to those who carry on trade ..."
}
```

The full generated dataset is published at [`zachnorton03/synthetic-pre1930-sft`](https://huggingface.co/datasets/zachnorton03/synthetic-pre1930-sft), one folder per route.
