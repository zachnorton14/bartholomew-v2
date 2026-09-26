"""
Filter Talkie's conversations and expand them into one SFT row per kept turn.

A row's messages are the conversation up to and including that assistant turn;
training targets only the last one. A turn is dropped if it was cut off at the
token cap, leaks the persona, echoes the user, is too short, copies an earlier
answer, uses modern vocabulary, is mostly non-letters, or greets/assents when
asked a question. Once a turn is dropped the rest of that conversation goes too,
since its context contains the bad turn. For topic fragments ("pepper") the first
answer must also name the topic and run at least 12 words.

The held-out split is disjoint by prompt family (all samples and turns of a prompt).

python -m dev.talkie_distill.filter --gen gen.jsonl --out filtered/
"""

import argparse
import collections
import json
import os
import random
import re

parser = argparse.ArgumentParser(description="Filter Talkie conversations")
parser.add_argument("--gen", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--min-words", type=int, default=3)
parser.add_argument("--heldout-families", type=int, default=300)
parser.add_argument("--seed", type=int, default=7)
args = parser.parse_args()

LEAK = re.compile(r"well-read|well read|\bvisitor\b|these instructions|present day|\byear 1930\b|familiar with history|"
                  r"know of events|carry (my|your) learning|^you are\b", re.I)
GREETING = re.compile(r"^(I am (very |most |right )?(glad|pleased|happy|delighted) to (see|meet|hear)|good (morning|day|evening|night)\b|"
                      r"how do you do|I thank you\.?$|thank you\.?$|I understand\.?$|very well\.?$|yes\.?$|no\.?$)", re.I)
MODERN = re.compile(r"\b(television|televisions|computer|computers|internet|nazi|nazis|hitler|world war (ii|two|2)|"
                    r"second world war|atomic bomb|nuclear|radar|penicillin|jet plane|rocket ship|nylon|"
                    r"19[3-9]\d|20\d\d)\b", re.I)
CASUAL_ROUTES = {"casual"}  # greetings are the right answer here


def ngrams(text, n=4):
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def names_topic(answer, topic):
    words = [w for w in topic.split() if len(w) >= 3 and w not in ("the", "queen", "king")]
    return any(w[:5].lower() in answer.lower() for w in words)


def reject(answer, user, turn, conv, previous):
    letters = sum(c.isalpha() or c.isspace() for c in answer) / max(len(answer), 1)
    if not conv["turns"][turn - 1]["ended"]: return "capped"
    if conv.get("topic") and turn == 1 and not names_topic(answer, conv["topic"]): return "off_topic"
    if conv.get("topic") and turn == 1 and len(answer.split()) < 12: return "short_topic"
    if LEAK.search(answer): return "leak"
    if normalize(answer) in normalize(user): return "echo"
    if len(answer.split()) < args.min_words: return "short"
    if MODERN.search(answer): return "modern"
    if letters < 0.8: return "nonalpha"
    if re.match(r"^\d+[A-Za-z]", answer): return "leading_digit"
    if not re.search(r"[.!?\"'”’)\]]$", answer): return "unterminated"
    if turn == 1 and conv["route"] not in CASUAL_ROUTES and GREETING.search(answer): return "greeting_on_question"
    grams = ngrams(answer)
    if any(len(grams & ngrams(p)) > 0.5 * max(len(grams), 1) for p in previous): return "repeat"
    return None


convs = {}
for line in open(args.gen):
    row = json.loads(line)
    convs[row["id"]] = row  # last line per id has the most turns

reasons = collections.Counter()
rows = []
for conv in convs.values():
    messages = conv["messages"]
    assistant_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    n_prior = len(assistant_idx) - len(conv["turns"])  # multi-turn prompts arrive with earlier answers
    previous = [messages[i]["content"].strip() for i in assistant_idx[:n_prior]]
    for turn, i in enumerate(assistant_idx[n_prior:], start=1):
        answer = messages[i]["content"].strip()
        reason = reject(answer, messages[i - 1]["content"].strip(), turn, conv, previous)
        reasons[reason or "kept"] += 1
        if reason:
            break
        rows.append({"id": f"{conv['id']}:t{turn}", "family_id": conv["id"].rsplit(":", 1)[0],
                     "route": conv["route"], "source": conv["source"], "topic": conv.get("topic"),
                     "turn": turn, "messages": messages[:i + 1]})
        previous.append(answer)

families = sorted({r["family_id"] for r in rows})
random.Random(args.seed).shuffle(families)
heldout = set(families[:args.heldout_families])
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "train.jsonl"), "w") as ftrain, open(os.path.join(args.out, "heldout.jsonl"), "w") as fheld:
    for row in rows:
        (fheld if row["family_id"] in heldout else ftrain).write(json.dumps(row, ensure_ascii=False) + "\n")
stats = {"conversations": len(convs), "turns": dict(reasons),
         "rows": {"train": sum(r["family_id"] not in heldout for r in rows), "heldout": sum(r["family_id"] in heldout for r in rows)},
         "by_route": dict(collections.Counter(r["route"] for r in rows))}
json.dump(stats, open(os.path.join(args.out, "filter_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
