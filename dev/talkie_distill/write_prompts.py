"""
Talkie writes the user side too: prompts seeded by random passages of the
pre-1930 pretraining corpus, so the prompt set shares nothing with the SFT data.

For each passage Talkie writes one prompt of a given kind (question, how-to,
opinion, explanation, composition request, casual remark) that must stand on its
own; the passage is then discarded. It also names the passage's subject in one to
three words, which becomes a one-word prompt ("pepper", "tell me about pepper").
A share of all prompts gets a lowercased / unpunctuated / typo'd copy.

Prompts close to a held-out eval prompt or an SFT question are dropped.

python -m dev.talkie_distill.write_prompts --corpus data/base_data_v3 --out prompts/ --passages 12000
"""

import argparse
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

from dev.talkie_distill.teacher import Teacher

KINDS = {
    "questions": ("Write one short question that a curious visitor might ask you about something in this "
                  "passage. It must make sense to someone who has not read it, so do not mention the passage.", 0.30),
    "howto": ("Write one short request, as a visitor might put it to you, asking how to do something practical "
              "that this passage touches on. Do not mention the passage.", 0.15),
    "opinion": ("Write one short question asking your opinion on a matter this passage raises. "
                "Do not mention the passage.", 0.15),
    "explain": ("Write one short question, beginning with Why or How, about a thing or event in this passage. "
                "Name the thing plainly and do not mention the passage.", 0.15),
    "casual": ("You are the visitor now, calling on a friend. Write one short sentence you might say to them: {}.",
               0.15),
}
COMPOSE_TEMPLATES = ["Write {form} about {topic}.", "Please write {form} on {topic}.", "Could you write {form} about {topic}?",
                     "Compose {form} on {topic}.", "write {form} about {topic}", "I should like {form} about {topic}."]
COMPOSE_FORMS = ["a letter", "a few lines of verse", "a short description", "a toast", "a short story", "a list",
                 "a note", "a short speech", "a riddle", "an advertisement"]
CASUAL_TOPICS = ["a greeting", "a remark on the weather", "how you are feeling today", "a small piece of news",
                 "thanks for something", "taking your leave", "something you did today", "a complaint about something",
                 "asking after their health", "something you are looking forward to"]
SUBJECT = ("Name the one thing, place, person, creature or practice this passage is mostly about, as a noun of one "
           "to three words. Write only the noun.")
META = re.compile(r"passage|extract|foregoing|excerpt|\bauthor\b|\bwriter\b|above|this book|the book|article|"
                  r"chapter|why or how|the following|something|\b(this|these|those|such|described|mentioned)\b", re.I)
DANGLING = re.compile(r"\b(on|upon|of|about|the|a|an|to|for|mrs|mr)[.?!]?$", re.I)
STYLES = [lambda t: t, lambda t: t.capitalize(), lambda t: t + "?", lambda t: f"tell me about {t}",
          lambda t: f"what about {t}", lambda t: f"thoughts on {t}", lambda t: f"{t} facts", lambda t: f"and {t}?",
          lambda t: f"what of {t}", lambda t: f"{t}..."]
KEYBOARD = "qwertyuiop asdfghjkl zxcvbnm".split()

parser = argparse.ArgumentParser(description="Talkie writes passage-seeded prompts")
parser.add_argument("--corpus", required=True, help="dir of pretraining parquet shards (column 'text')")
parser.add_argument("--out", required=True)
parser.add_argument("--passages", type=int, default=12000)
parser.add_argument("--messy-fraction", type=float, default=0.3)
parser.add_argument("--subject-fraction", type=float, default=0.25, help="passages that also yield a one-word topic")
parser.add_argument("--exclude", nargs="*", default=[], help="jsonl files of prompts/questions to stay away from")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--seed", type=int, default=1930)
args = parser.parse_args()
rng = random.Random(args.seed)


def passages(n):
    """Random ~800-character windows of English prose, cut at paragraph boundaries."""
    shards = sorted(glob.glob(os.path.join(args.corpus, "*.parquet")))
    books = [t for s in shards for t in pq.read_table(s, columns=["text"]).column("text").to_pylist()]
    out = []
    while len(out) < n:
        paragraphs = [p.strip() for p in rng.choice(books).split("\n\n") if len(p.strip()) > 200]
        if not paragraphs:
            continue
        i = rng.randrange(len(paragraphs))
        text = paragraphs[i]
        while len(text) < 600 and i + 1 < len(paragraphs):
            i += 1
            text += "\n\n" + paragraphs[i]
        text = text[:1200]
        letters = sum(c.isascii() and c.isalpha() for c in text) / max(sum(c.isalpha() for c in text), 1)
        words = text.lower().split()
        english = sum(w in ("the", "and", "of", "to", "in", "is", "was") for w in words) / max(len(words), 1)
        if letters > 0.98 and english > 0.08:
            out.append(text)
    return out


def clean(text):
    text = text.strip().strip('"').strip("“”").strip()
    text = re.sub(r"^(question|request|remark)\s*:\s*", "", text, flags=re.I)
    return text.split("\n")[0].strip()


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def messy(text, r):
    text = text.lower()
    if r.random() < 0.6:
        text = re.sub(r"[.,;:!?'\"]", "", text)
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and r.random() < 0.08:
            row = next((k for k in KEYBOARD if chars[i] in k), None)
            op = r.choice(["drop", "double", "swap"] + (["near"] if row else []))
            if op == "drop": chars[i] = ""
            elif op == "double": chars[i] *= 2
            elif op == "swap" and i + 1 < len(chars): chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == "near": chars[i] = row[min(max(row.index(chars[i]) + r.choice([-1, 1]), 0), len(row) - 1)]
    return "".join(chars).strip()


excluded = set()
for path in args.exclude:
    for line in open(path):
        row = json.loads(line)
        text = row.get("prompt") or row.get("question") or row["messages"][0]["content"]
        excluded.add(normalize(text))

teacher = Teacher()
seeds = passages(args.passages)
kinds, weights = zip(*[(k, w) for k, (_, w) in KINDS.items()])
jobs = [(i, rng.choices(kinds, weights)[0]) for i in range(len(seeds))]
jobs += [(i, "subject") for i in range(len(seeds)) if rng.random() < args.subject_fraction]
results = {}
for b0 in range(0, len(jobs), args.batch):
    batch = jobs[b0:b0 + args.batch]
    fill = {"casual": CASUAL_TOPICS}
    instructions = [SUBJECT if kind == "subject" else KINDS[kind][0].format(rng.choice(fill.get(kind, [""])))
                    for _, kind in batch]
    prompts = [teacher.encode_chat([{"role": "user", "content": f"Here is a passage from a book.\n\n{seeds[i]}\n\n{ins}"}])
               for (i, _), ins in zip(batch, instructions)]
    prompts += [prompts[-1]] * (args.batch - len(batch))
    outs = teacher.generate(prompts, max_new_tokens=60, temperature=0.8, top_p=0.95, seed=args.seed + b0)
    for (i, kind), out in zip(batch, outs):
        results[i, kind] = clean(teacher.decode(out))
    if (b0 // args.batch) % 50 == 0:
        print(f"{b0 + len(batch)}/{len(jobs)}", flush=True)

pools, seen = {}, set(excluded)


def add(route, text, **extra):
    key = normalize(text)
    if not key or key in seen:
        return
    seen.add(key)
    pools.setdefault(route, []).append({"id": f"{route}:{len(pools.get(route, []))}", "route": route,
                                        "messages": [{"role": "user", "content": text}], **extra})


for (i, kind), text in sorted(results.items()):
    words = text.split()
    if kind == "subject":
        topic = re.sub(r"^(the|a|an) ", "", re.sub(r"[^a-z' -]", "", text.lower()).strip())
        if 1 <= len(topic.split()) <= 3 and len(topic) >= 3 and not META.search(topic):
            for style in rng.sample(range(len(STYLES)), 2):
                add("fragments", STYLES[style](topic), topic=topic, source=f"passage:{i}")
            if rng.random() < 0.5:  # composition requests are templated; Talkie writes titles, not requests
                add("compose", rng.choice(COMPOSE_TEMPLATES).format(form=rng.choice(COMPOSE_FORMS), topic=topic),
                    source=f"passage:{i}")
        continue
    if (not (2 <= len(words) <= 40) or DANGLING.search(text) or text.lower().startswith("what is meant")
            or (kind != "casual" and META.search(text))):
        continue
    add(kind, text, source=f"passage:{i}")
    if kind != "casual" and rng.random() < args.messy_fraction:
        add("messy", messy(text, rng), source=f"passage:{i}:{kind}")

os.makedirs(args.out, exist_ok=True)
for route, rows in pools.items():
    with open(os.path.join(args.out, f"{route}.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
print({route: len(rows) for route, rows in pools.items()})
