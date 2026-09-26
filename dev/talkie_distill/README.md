# Talkie distillation data

Builds [talkie-distill-pre1930](https://huggingface.co/datasets/greenpolo/talkie-distill-pre1930):
conversations written by [Talkie-1930-13B-IT](https://huggingface.co/talkie-lm/talkie-1930-13b-it) for
off-policy distillation into Bartholomew. Training is `scripts/chat_distill.py`, driven by
`configs/posttrain/pre1930-curriculum-c3-robust-v2-talkie-distill-v1.json`.

Everything runs on one 32 GB GPU (tested on an RTX 5090). Talkie runs as int8, which needs
`uv pip install torchao==0.15.0` (not in the lockfile).

```sh
# 1. Talkie answers every prompt twice; half the conversations get a follow-up, 30% of those a third (~3.5 h)
python -m dev.talkie_distill.generate --pools prompts/*.jsonl --out gen.jsonl \
    --exclude dumbass-bench/data/prompts.jsonl

# 2. drop bad turns, one row per kept turn, held-out split by prompt family
python -m dev.talkie_distill.filter --gen gen.jsonl --out filtered/

# 3. fragment answer swap + length floor, write rows/<route>/ and heldout/
python -m dev.talkie_distill.build_dataset --filtered filtered/ --fragment-prompts prompts/fragments.jsonl \
    --out talkie-distill-pre1930/
```

The prompt pools are in the dataset's `prompts/` folder. `teacher.py` and `talkie_model.py` are a
minimal copy of Talkie's inference code (Apache-2.0) with a KV cache and left-padded batching.

## What mattered

- **Off-policy, not on-policy.** We first tried on-policy distillation (Bart samples, Talkie scores).
  It removed "I don't understand" deflections but plateaued: Bart almost never samples a good answer to
  "pepper" or "tell me more", so there is nothing for the teacher's score to reinforce.
- **Talkie greets on bare words.** Asked "pepper", it answers "I am glad to see you" or "Yes, I am fond
  of pepper" about a third of the time, and a model trained on that does the same. `build_dataset.py`
  pairs every phrasing of a topic with Talkie's descriptive answers to it instead.
- **The persona goes to the teacher only.** It is merged into Talkie's first user turn and stripped from
  the saved conversations, so Bart learns the behaviour without needing the prompt.
