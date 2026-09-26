"""
Assemble filtered Talkie rows into the dataset layout (rows/<route>/part-00000.jsonl,
heldout/part-00000.jsonl), the same layout as zachnorton03/vintage-sft-robustness.

Two fixes on top of the filter, both from looking at the trained model:
1. Bare-word prompts ("pepper"): Talkie usually greets or assents ("Yes, I am fond
   of pepper."), and a model trained on that does the same. So every prompt style of
   a topic, bare word included, is paired with one of Talkie's descriptive answers to
   that topic (>= 15 words, no first-person or assent opener), usually written for
   "tell me about pepper". Only first turns are used for this route.
2. First answers to real questions must run at least 12 words; Talkie is terse and
   the student copies it.

python -m dev.talkie_distill.build_dataset --filtered filtered/ --fragment-prompts prompts/fragments.jsonl --out talkie-distill-pre1930/
"""

import argparse
import collections
import json
import os
import random
import re

parser = argparse.ArgumentParser(description="Build the Talkie distillation dataset")
parser.add_argument("--filtered", nargs="+", required=True, help="filter.py output dirs")
parser.add_argument("--fragment-prompts", required=True, help="every prompt style per topic")
parser.add_argument("--out", required=True)
parser.add_argument("--heldout-topics", type=int, default=50)
parser.add_argument("--seed", type=int, default=11)
args = parser.parse_args()

ASSENT = re.compile(r"^(yes|no|i |i'|we |you |your |very well|indeed|certainly|thank)", re.I)
rng = random.Random(args.seed)


def read(path):
    return [json.loads(line) for line in open(path) if line.strip()]


rows = {"train": [], "heldout": []}
answers = collections.defaultdict(list)  # topic -> descriptive first answers
for d in args.filtered:
    for split in ("train", "heldout"):
        for row in read(os.path.join(d, f"{split}.jsonl")):
            answer = row["messages"][-1]["content"]
            if row["route"] == "fragments":
                if row["turn"] == 1 and len(answer.split()) >= 15 and not ASSENT.search(answer):
                    answers[row["topic"]].append(answer)
                continue
            if row["turn"] == 1 and row["route"] != "casual" and len(answer.split()) < 12:
                continue
            rows[split].append(row)

styles = collections.defaultdict(list)
for p in read(args.fragment_prompts):
    styles[p["topic"]].append(p["messages"][0]["content"])
topics = sorted(t for t in styles if answers[t])
heldout_topics = set(rng.sample(topics, args.heldout_topics))
for topic in topics:
    for prompt in sorted(set(styles[topic])):
        split = "heldout" if topic in heldout_topics else "train"
        rows[split].append({"id": f"fragments:{topic.replace(' ', '_')}:{len(rows[split])}", "family_id": f"topic:{topic}",
                            "route": "fragments", "source": "topic_fragment", "topic": topic, "turn": 1,
                            "messages": [{"role": "user", "content": prompt},
                                         {"role": "assistant", "content": rng.choice(answers[topic])}]})

by_route = collections.defaultdict(list)
for row in rows["train"]:
    by_route[row["route"]].append(row)
for route, route_rows in by_route.items():
    os.makedirs(os.path.join(args.out, "rows", route), exist_ok=True)
    with open(os.path.join(args.out, "rows", route, "part-00000.jsonl"), "w") as f:
        for row in route_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
os.makedirs(os.path.join(args.out, "heldout"), exist_ok=True)
with open(os.path.join(args.out, "heldout", "part-00000.jsonl"), "w") as f:
    for row in rows["heldout"]:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"train": {r: len(v) for r, v in sorted(by_route.items())}, "heldout": len(rows["heldout"]),
                  "fragment_topics": len(topics), "fragment_answers": sum(len(answers[t]) for t in topics)}, indent=1))
