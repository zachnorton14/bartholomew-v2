---
license: cc-by-4.0
language:
- en
tags:
- vintage
- historical
- pre-1930
- sft
- robustness
size_categories:
- 1K<n<10K
pretty_name: Vintage SFT Robustness
---

# Vintage SFT Robustness

Supplementary SFT rows for a pre-1930 conversational model. **7,338 rows across five
routes.** Kept deliberately separate from the graded synthetic dataset, whose rows are
answers lifted verbatim from period prose and scored by a judge; these are constructed,
ungraded, and would muddy that provenance guarantee if mixed in.

## Why it exists

A model pretrained on pre-1930 books and finetuned on question-and-answer pairs behaves
correctly right up until the input stops looking like a question. Then it falls off the
finetuned distribution, falls back on base-corpus behaviour — book prose, which carries
almost no end-of-sequence token — and continues until the token budget runs out.

Observed failures that motivated each route:

| input | what happened | route |
|---|---|---|
| `Hello` | paragraphs of narration | `conversation_qa` |
| `xakhavjba` | same | `unparseable_qa` |
| a question with the `?` dropped | same | train-time noise (below) |
| `What is a nedle?` | answered something else | `typo_qa` |
| `What year is it?` | wrong or evasive | `era_qa` |
| any multi-turn exchange | no model of taking turns | `conversation_multiturn` |

The last one has a structural cause worth recording. The graded dataset was built by
scoring excerpts on `region_quality`, which rewards mean sentence length. Dialogue turns
are 2–6 words, so quoted speech scores near zero on a quarter of that metric and lands
below the 0.70 keep floor. Measured on the source corpus:

```
excerpts with no dialogue   0.741 mean prose score   (passes)
excerpts with dialogue      0.62–0.64                (fails)
```

The SFT data is therefore ~98% narration. The model trained on books, but on the parts
of books that are not people talking.

## Contents

| route | rows | median tokens/row | what it covers |
|---|---|---|---|
| `conversation_qa` | 3,500 | 15 | greetings, farewells, thanks, acknowledgements, odd openers (`Praise God!`), vague references (`Does he so?`) |
| `unparseable_qa` | 2,000 | 18 | gibberish and cut-off sentences |
| `typo_qa` | 1,200 | 30 | one word mangled hard → `You mean needle, I take it?` |
| `era_qa` | 220 | 17 | year (147), anachronisms (31), identity (25), place (17) |
| `conversation_multiturn` | 418 | 86 | genuine two-party dialogue, 2,044 turns |
| **total** | **7,338** | | **~142k tokens per pass** |

`unparseable_qa` at 2,000 rows: fragment 655, multi_token 365, single_nonword 349,
consonant_run 193, long_run 99, alnum_noise 93, micro 73, mixed_case 69, keyboard_mash 64,
repeated_char 32, bare_punctuation 8. The last two saturate — their distinct value spaces
are finite and rows deduplicate on the question.

**Row count and token weight diverge sharply.** These rows are short. In a curriculum where
they are 3.4% of rows they can be 0.6% of tokens, so the gradient signal is far smaller
than the row count suggests. Size the dose by tokens, not rows.

Single-turn rows carry `question` / `answer`; `conversation_multiturn` carries a
`conversations` list of alternating `{role, content}`. All carry `score: 100` — they are
constructed, not graded, and must clear any curriculum grade threshold.

## How each part was sourced

Three provenance classes, and only one carries anachronism risk.

**Corpus-verbatim** — lifted from a 988 MB pre-1930 excerpt corpus (452,591 excerpts):

- 90 deflection/social clauses, kept only when context-free (a terminator follows rather
  than a complement, so `I don't know what you mean` qualifies and
  `...by load factor` does not)
- 5,000 whole sentences, truncated at a random word boundary to make fragments
- 4,000 short standalone utterances, used as odd openers
- all `conversation_multiturn` turns, from
  [`croqaz/vintage-conversations`](https://huggingface.co/datasets/croqaz/vintage-conversations)
  (Gutenberg novels, verified letter-by-letter against the source)

**Procedural** — random strings and mechanical mangling. Carries no register at all, so no
anachronism risk: the gibberish classes, and the five typo operations (drop, swap, double,
QWERTY-neighbour, phonetic).

**Authored** — 115 distinct strings written by hand in period voice. This is the entire
anachronism surface. Everything else is corpus text or random characters.

```
 32  conversation openers ("What brings you", "What is the news with you")
 12  pronoun-specific asks ("Who is 'he'? We have not spoken of anyone.")
 12  year answers ("Nineteen hundred and thirty.", "1930, and well into it.")
 10  typo confirmations ("You mean {word}, I take it?")
  9  fallback acknowledgements, 9 identity answers, 8 closers,
  7  curious tails, 6 generic asks, 5 nature answers, 5 place answers
115  TOTAL
```

At 7,338 rows those 115 strings recur heavily — a bad one repeats thousands of times. They
are also 115 lines, readable in five minutes, and worth reading before training on them.

## Anachronism risk warnings

**1. Corpus-verbatim protects against modern leakage, not against archaism.**
The corpus spans centuries. `good morrow` was the single most frequent greeting match and
`adieu` the second most frequent farewell — both Shakespearean-to-Georgian, wrong for 1930.
Filtering by the excerpt's `year` does **not** fix this: `good morrow` is *more* frequent in
1900s-published books than in 1800s ones, because publication year does not track register
(the corpus holds reprints and historical fiction that are archaic on purpose). Curation by
hand was the only reliable filter, and the blocklist is necessarily incomplete.

**2. Frequency is not evidence of fitness.**
`come again` was the most common "invitation" match by a wide margin, precisely because the
period sense is *return*, not *repeat that* — every sampled occurrence was astronomical
recurrence or a parting pleasantry. It is blocklisted. Others of the same kind may survive.

**3. `conversation_multiturn` skews Victorian, earlier than 1930.**
Sourced from Dickens, Eliot, Tolstoy in translation. This is defensible — the pretraining
corpus is itself modal around 1850–1910 — but it is not 1930.

**4. Dialect survives in user turns.**
Phonetic dialect (`Know'd it yes'day aft'noon at tea-time`) is rejected from **assistant**
turns only. It remains in some user turns, on the reasoning that a visitor may type
anything but the model should not answer that way.

**5. Speaker attribution in the upstream dialogue set is not guaranteed.**
Its own card says so. Mechanical filters catch structural breaks, not semantic ones, so some
exchanges will not cohere even though every turn is verbatim.

**6. The user side is deliberately modern, and that is not a bug.**
`hi`, `hey`, `what's up`, `ok` appear as inputs on purpose: the visitor is a person at a
keyboard today. Only the **answers** need to hold period register.

**7. Constructed answers are not corpus-attested.**
The 115 authored strings are assembled in period voice but no period book contains them.
They pass no anachronism filter because none was run on them.

## Companion: train-time input noise

Not part of this dataset, but designed with it. The consuming trainer batters a share of
**user turns across the whole curriculum** — 16 operations in five families: punctuation
(including doubled `Hello!!`), case (`HELlo`, shout, leading-lower), spacing (`Whatis`,
`What   is`), character (transpose, dropped/doubled letter, QWERTY-neighbour), and word
(dropped, repeated, apostrophe stripped). Seeded per epoch, so a row reads clean on one pass
and battered on the next.

Answers are never touched. The reason is worth stating: *a corrupted question is context the
model reads; a corrupted answer is a target it learns to reproduce.*

`typo_qa` is the complement, not a duplicate: noise applies light damage anywhere and leaves
the answer alone, teaching the model to read through it and answer; `typo_qa` applies heavy
damage to one load-bearing word, where guessing is wrong and asking is right. Keeping the two
separate is what should stop the model querying every faint typo.

**Caveat:** the character-level operations mimic OCR artifacts — the exact class of corruption
that may have been cleaned out of the source corpus. That is intentional on the question side.
If the model begins reproducing OCR-style misspellings in its **answers**, suspect this first.

## Reproduction

Deterministic for a given `(count, seed)`; default seed 1930.

```
python -m synth.robustness.mine --scan-bytes 0        # mine the clause pools (slow)
python -m synth.robustness.mine --fragments 2500      # sentence pool
python -m synth.robustness.mine --utterances 4000     # utterance pool
python -m synth.robustness.build --dry-run            # counts only
python -m synth.robustness.build                      # build and push
```

Layout is `rows/<route>/part-*.jsonl`.
