"""
dumbass-bench: does the model give a sensible reply? Not a correct one.

200 hand-written prompts in seven kinds (one-word fragments, small talk, how-to,
opinion, explanation, composition requests, misspelled messages), 64 of them
with a scripted second turn ("why?", "tell me more", "i dont understand") asked
after the model's own first answer. Facts and anachronisms are not graded.

Stages:
  generate  sample every model in the config (2 samples per prompt, chat_cli settings)
  packets   blind grading packets: per prompt, every model's first sample, shuffled A, B, C, ...
  tally     automatic metrics + blind grades (after unblinding) -> results table

Grades come from any judge following RUBRIC (an LLM or a person) and are written
as grades/<packet>.json: [{"item": ..., "scores": {"A": {"appropriate": 2, "coherent": 2,
"fluent": 2}, ...}}]. A reply is "acceptable" when appropriate = 2 and coherent,
fluent >= 1.

python -m scripts.dumbass_bench generate --config configs/dumbass_bench/v1.json --model released-sft
python -m scripts.dumbass_bench packets --config configs/dumbass_bench/v1.json
python -m scripts.dumbass_bench tally --config configs/dumbass_bench/v1.json
"""

import argparse
import collections
import json
import os
import random
import re

RUBRIC = """Score every reply 0-2 on each scale. Ignore factual accuracy, anachronisms and length;
old-fashioned phrasing is fine. For follow-up items grade only the follow-up reply.
- appropriate: 2 = responds to what the user actually said in a fitting way (answers the question or tells
  about the named topic, returns a greeting, attempts the request). 1 = loosely related or a weak attempt.
  0 = ignores or misreads the message, deflects ("I don't understand", "tell me plainly"), greets when
  asked something, or talks about something else.
- coherent: 2 = makes sense throughout. 1 = some odd or self-contradictory parts. 0 = mostly nonsense or loops.
- fluent: 2 = grammatical, well-formed sentences. 1 = some broken sentences. 0 = mostly garbled.
A reply cut off mid-sentence at the very end is a length cap; do not penalize that alone."""

DEFLECT = re.compile(r"(don'?t|do not|cannot|can't|can not)( quite)? (understand|follow|comprehend|make it out)|"
                     r"tell me plainly|what do you mean|how do you mean|say (it|that) again|put it another way|"
                     r"speak (more )?plainly|beg your pardon|i am at a loss|know what you mean", re.I)
LETTERS = "ABCDEFGH"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompts(config):
    source = config["prompts"]
    if os.path.exists(source["path"]):
        return read_jsonl(source["path"])
    from huggingface_hub import hf_hub_download
    return read_jsonl(hf_hub_download(source["repo"], source["path"], repo_type="dataset", revision=source.get("revision")))


# -----------------------------------------------------------------------------
# generate

def nanochat_sampler(model_config):
    import torch
    from nanochat.checkpoint_manager import load_model_from_checkpoint_dir
    from nanochat.common import autodetect_device_type, compute_init, get_base_dir
    from nanochat.engine import Engine
    _, _, _, _, device = compute_init(autodetect_device_type())
    # relative checkpoint dirs are under $NANOCHAT_BASE_DIR, like the harness's experiments/ tree
    checkpoint_dir = os.path.join(get_base_dir(), model_config["checkpoint_dir"])
    model, tokenizer, _ = load_model_from_checkpoint_dir(checkpoint_dir, device, phase="eval", step=model_config.get("step"))
    engine = Engine(model, tokenizer)

    def sample(messages, num_samples, max_tokens, seed):
        tokens = tokenizer.render_for_completion({"messages": messages + [{"role": "assistant", "content": ""}]})
        with torch.no_grad():
            out, _ = engine.generate_batch(tokens, num_samples=num_samples, max_tokens=max_tokens,
                                           temperature=0.6, top_k=50, seed=seed)
        return [(tokenizer.decode(o[len(tokens):]).strip(), len(o) - len(tokens) < max_tokens) for o in out]
    return sample


def talkie_sampler(model_config):
    from dev.talkie_distill.teacher import Teacher
    teacher = Teacher()
    persona = open(model_config["persona"]).read().strip() if model_config.get("persona") else None

    def sample(messages, num_samples, max_tokens, seed):
        messages = [dict(m) for m in messages]
        if persona:
            messages[0]["content"] = persona + "\n\n" + messages[0]["content"]
        prompt = teacher.encode_chat(messages)
        outs = teacher.generate([prompt] * num_samples, max_new_tokens=max_tokens, seed=seed)
        return [(teacher.decode(o).strip(), len(o) < max_tokens) for o in outs]
    return sample


def generate(config, model_id):
    model_config = next(m for m in config["models"] if m["id"] == model_id)
    sample = {"nanochat-checkpoint": nanochat_sampler, "talkie": talkie_sampler}[model_config["backend"]](model_config)
    gen = config["generation"]
    rows = []
    for p in load_prompts(config):
        first = [{"role": "user", "content": p["prompt"]}]
        for s, (text, ended) in enumerate(sample(first, gen["samples"], gen["max_tokens"], seed=gen["seed"])):
            rows.append({"id": p["id"], "turn": 1, "sample": s, "messages": first, "reply": text, "ended": ended})
            if p.get("followup"):
                second = first + [{"role": "assistant", "content": text}, {"role": "user", "content": p["followup"]}]
                (text2, ended2), = sample(second, 1, gen["max_tokens"], seed=gen["seed"] + 1 + s)
                rows.append({"id": p["id"], "turn": 2, "sample": s, "messages": second, "reply": text2, "ended": ended2})
    write_jsonl(os.path.join(config["output_dir"], "replies", f"{model_id}.jsonl"), rows)
    print(f"{model_id}: {len(rows)} replies")


# -----------------------------------------------------------------------------
# packets

def packets(config, num_packets):
    replies = {m["id"]: {(r["id"], r["turn"]): r for r in read_jsonl(os.path.join(config["output_dir"], "replies", f"{m['id']}.jsonl"))
                         if r["sample"] == 0} for m in config["models"]}
    models = sorted(replies)
    items = sorted(next(iter(replies.values())))
    rng = random.Random(1930)
    rng.shuffle(items)
    key = {}
    for k in range(num_packets):
        lines = [RUBRIC, ""]
        for pid, turn in items[k::num_packets]:
            item = f"{pid}-t{turn}"
            order = rng.sample(models, len(models))
            key[item] = dict(zip(LETTERS, order))
            first = replies[order[0]][pid, turn]["messages"]
            lines.append(f"### ITEM {item}")
            if turn == 1:
                lines.append(f"USER: {first[0]['content']}")
            else:
                lines.append(f"First user message: {first[0]['content']}\nFollow-up: {first[2]['content']}")
            for letter, model in zip(LETTERS, order):
                r = replies[model][pid, turn]
                context = f"[own first answer, context only] {r['messages'][1]['content']}\n" if turn == 2 else ""
                lines.append(f"--- reply {letter}:\n{context}{r['reply'] or '(empty)'}")
            lines.append("")
        os.makedirs(os.path.join(config["output_dir"], "packets"), exist_ok=True)
        with open(os.path.join(config["output_dir"], "packets", f"p{k}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    with open(os.path.join(config["output_dir"], "packets", "key.json"), "w") as f:
        json.dump(key, f, indent=1)
    print(f"{len(items)} items in {num_packets} packets")


# -----------------------------------------------------------------------------
# tally

def words(text):
    return re.findall(r"[A-Za-z']+", text)


def overlap(a, b):
    grams = lambda t: [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
    ga, gb = grams([w.lower() for w in words(a)]), set(grams([w.lower() for w in words(b)]))
    return sum(g in gb for g in ga) / len(ga) if ga else 0.0


def bootstrap(diffs, n=2000):
    rng = random.Random(0)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def tally(config):
    out = config["output_dir"]
    print("| model | words | capped | deflect | follow-up repeats | acceptable, first | acceptable, follow-up |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    key_path = os.path.join(out, "packets", "key.json")
    grades = collections.defaultdict(dict)  # model -> item -> acceptable
    if os.path.exists(key_path):
        key = json.load(open(key_path))
        for name in sorted(os.listdir(os.path.join(out, "grades"))):
            for g in json.load(open(os.path.join(out, "grades", name))):
                for letter, s in g["scores"].items():
                    grades[key[g["item"]][letter]][g["item"]] = int(s["appropriate"] == 2 and s["coherent"] >= 1 and s["fluent"] >= 1)
    for m in config["models"]:
        rows = read_jsonl(os.path.join(out, "replies", f"{m['id']}.jsonl"))
        first = {(r["id"], r["sample"]): r["reply"] for r in rows if r["turn"] == 1}
        second = [r for r in rows if r["turn"] == 2]
        acc = grades.get(m["id"], {})
        rate = lambda t: (f"{sum(v for k, v in acc.items() if k.endswith(t)) / max(sum(k.endswith(t) for k in acc), 1):.0%}" if acc else "-")
        print(f"| {m['id']} | {sum(len(words(r['reply'])) for r in rows) / len(rows):.0f} "
              f"| {sum(not r['ended'] for r in rows) / len(rows):.0%} "
              f"| {sum(bool(DEFLECT.search(r['reply'])) for r in rows) / len(rows):.0%} "
              f"| {sum(overlap(r['reply'], first[r['id'], r['sample']]) > 0.5 for r in second)}/{len(second)} "
              f"| {rate('-t1')} | {rate('-t2')} |")
    for a, b in config.get("compare", []):
        if a in grades and b in grades:
            diffs = [grades[a][i] - grades[b][i] for i in grades[a]]
            lo, hi = bootstrap(diffs)
            print(f"{a} vs {b}: acceptable {sum(diffs) / len(diffs):+.3f} [95% {lo:+.3f}, {hi:+.3f}], "
                  f"{sum(d > 0 for d in diffs)} wins / {sum(d < 0 for d in diffs)} losses")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dumbass-bench")
    parser.add_argument("stage", choices=["generate", "packets", "tally"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="model id to generate (default: all)")
    parser.add_argument("--packets", type=int, default=6)
    args = parser.parse_args()
    config = json.load(open(args.config))
    if args.stage == "generate":
        for m in config["models"]:
            if args.model in (None, m["id"]):
                generate(config, m["id"])
    elif args.stage == "packets":
        packets(config, args.packets)
    else:
        tally(config)
