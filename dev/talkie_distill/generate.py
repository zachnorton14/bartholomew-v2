"""
Talkie writes conversations for off-policy distillation.

Generation is stage-wise so the batch size stays fixed (the compiled int8 decode
autotunes once per batch size):
  stage 1: every conversation gets its first answer
  stage 2: a random subset gets a generic follow-up ("tell me more", "why?") and a second answer
  stage 3: a smaller subset gets a third
The persona (configs/system_prompts/talkie-1930-teacher.txt) is merged into the
first user turn for Talkie only; the saved conversations do not contain it.
Output is append-only jsonl, one line per generated turn, so a killed run resumes.

python -m dev.talkie_distill.generate --pools prompts/*.jsonl --out gen.jsonl --samples 2
"""

import argparse
import json
import os
import random
import time

import torch


FOLLOWUPS = ["tell me more", "Tell me more.", "go on", "Go on.", "and then?", "why?", "Why is that?",
             "what else", "How so?", "can you explain that", "more please", "Say more about that.",
             "is that so", "what do you mean", "and what came of it?", "Really?", "hm, tell me more about that",
             "what happened next", "why do you think that", "I don't follow.", "and?"]

parser = argparse.ArgumentParser(description="Generate Talkie conversations")
parser.add_argument("--teacher", default="talkie", choices=["talkie", "bart-sft"],
                    help="bart-sft = the released SFT answers itself (self-distillation control)")
parser.add_argument("--pools", nargs="+", required=True, help="prompt jsonl files: {id, route, messages} per line")
parser.add_argument("--out", required=True)
parser.add_argument("--persona", default="configs/system_prompts/talkie-1930-teacher.txt", help="'' = none")
parser.add_argument("--exclude", default="", help="jsonl of held-out eval prompts never to generate for")
parser.add_argument("--samples", type=int, default=2, help="independent samples per prompt")
parser.add_argument("--p-follow2", type=float, default=0.5, help="fraction of conversations that get a second turn")
parser.add_argument("--p-follow3", type=float, default=0.3, help="fraction of those that get a third")
parser.add_argument("--batch", type=int, default=16, help="16 fits a 32 GB card; 32 runs out of KV cache")
parser.add_argument("--max-new-tokens", type=int, default=300)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-p", type=float, default=0.95)
parser.add_argument("--seed", type=int, default=1930)
args = parser.parse_args()


def log(*msg):
    print(time.strftime("%H:%M:%S"), *msg, flush=True)


persona = open(args.persona).read().strip() if args.persona else ""
excluded = set()
if args.exclude:
    excluded = {json.loads(line)["prompt"].strip().lower() for line in open(args.exclude) if line.strip()}

# Prompt pool -> conversations (one per prompt per sample)
convs = []
for path in args.pools:
    for line in open(path):
        row = json.loads(line)
        messages = row["messages"]
        if messages[0]["content"].strip().lower() in excluded:
            continue
        for sample in range(args.samples):
            convs.append({"id": f"{row['id']}:{sample}", "route": row["route"], "source": row.get("source", ""),
                          "topic": row.get("topic"), "messages": [dict(m) for m in messages], "turns": []})
log(f"{len(convs)} conversations from {len(args.pools)} pools")

# Resume: the last line per id holds the most turns
if os.path.exists(args.out):
    by_id = {c["id"]: c for c in convs}
    for line in open(args.out):
        row = json.loads(line)
        if row["id"] in by_id:
            by_id[row["id"]].update(messages=row["messages"], turns=row["turns"])

if args.teacher == "talkie":
    from dev.talkie_distill.teacher import Teacher
    teacher = Teacher()
else:
    from dev.talkie_distill.bart_teacher import BartTeacher
    teacher = BartTeacher()


def teacher_messages(messages):
    messages = [dict(m) for m in messages]
    if persona:
        messages[0]["content"] = persona + "\n\n" + messages[0]["content"]
    return messages


def run_stage(stage, items):
    """Append one assistant turn to every conversation that ends in a user turn."""
    items = [c for c in items if len(c["turns"]) < stage]
    encoded = [(c, teacher.encode_chat(teacher_messages(c["messages"]))) for c in items]
    encoded = [(c, ids) for c, ids in encoded if len(ids) + args.max_new_tokens <= teacher.max_seq_len]
    encoded.sort(key=lambda x: len(x[1]))  # similar lengths per batch -> less padding
    log(f"stage {stage}: {len(encoded)} generations")
    t0, new_tokens = time.time(), 0
    with open(args.out, "a") as f:
        for b0 in range(0, len(encoded), args.batch):
            chunk = encoded[b0:b0 + args.batch]
            prompts = [ids for _, ids in chunk]
            prompts += [prompts[-1]] * (args.batch - len(chunk))  # keep the batch size fixed
            outs = teacher.generate(prompts, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                    top_p=args.top_p, seed=args.seed * 1000 + stage * 100000 + b0)
            for (conv, _), out in zip(chunk, outs):
                conv["messages"].append({"role": "assistant", "content": teacher.decode(out).strip()})
                conv["turns"].append({"tokens": len(out), "ended": len(out) < args.max_new_tokens})
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                new_tokens += len(out)
            f.flush()
            torch.cuda.empty_cache()
            if (b0 // args.batch) % 20 == 0:
                done = b0 + len(chunk)
                elapsed = time.time() - t0
                log(f"stage {stage}: {done}/{len(encoded)} | {new_tokens / elapsed:.0f} tok/s | "
                    f"eta {(len(encoded) - done) / done * elapsed / 60:.0f} min")


rng = random.Random(args.seed)
run_stage(1, convs)
for stage, p in ((2, args.p_follow2), (3, args.p_follow3)):
    followups = [c for c in convs if len(c["turns"]) == stage - 1 and rng.random() < p]
    for conv in followups:
        if conv["messages"][-1]["role"] == "assistant":
            conv["messages"].append({"role": "user", "content": rng.choice(FOLLOWUPS)})
    run_stage(stage, followups)
log("done")
