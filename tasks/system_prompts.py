"""
Put system prompts into SFT rows, so serving with one is no longer off-distribution.

No SFT source carries a system turn, yet the deployment prepends a persona to
every conversation. Tokenizer.render_conversation already handles a leading
system message (merged into the first user turn, masked out of the loss), so
all that is missing is rows that have one. This wraps a Task and gives a
deterministic share of its rows a system message drawn from
configs/system_prompts/.

Configured under data.system_prompts in an SFT experiment config:

    "system_prompts": {
      "seed": 1930,
      "mix": {"pre1930-companion": 0.5, "pre1930-companion-short": 0.3, "none": 0.2},
      "max_reply_words": 120
    }

`mix` weights are relative; "none" leaves the row as it was, so the model keeps
answering well without a prompt. `max_reply_words` keeps the persona off rows
whose replies break it: a persona that says "a few sentences" attached to a
300-word answer teaches the model that the prompt does not matter. Such rows
are left without a prompt (null disables the check).

`"depth_cue": true` is for personas that add "unless your visitor asks you to
explain, to go at length, or for a number of things" (the v2 family). A long
reply then keeps the persona when the visitor's first turn asks for depth, so
the model sees both halves of that rule instead of learning that a persona
always means a short answer.
"""

import random
import re
from pathlib import Path

from tasks.common import Task

PROMPT_DIR = Path(__file__).resolve().parents[1] / "configs" / "system_prompts"
# A first turn that asks for an explanation, length, or a number of things.
DEPTH_CUE_RE = re.compile(
    r"\b(?:explain|describe|in detail|at length|tell me (?:all )?about|why|how (?:does|do|did|is|are|was|were|can"
    r"|could|would|should)|give me|list|reasons|ways|compare|discuss|write (?:a|an|me)|account of)\b",
    re.IGNORECASE)


def _reply_words(conversation):
    """Word count of the longest assistant turn, counting only text parts."""
    longest = 0
    for message in conversation["messages"]:
        if message["role"] != "assistant":
            continue
        content = message["content"]
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if part.get("type") == "text")
        longest = max(longest, len(content.split()))
    return longest


class SystemPromptMixture(Task):
    """A view of `task` in which a deterministic share of rows open with a system message."""

    def __init__(self, task, spec, **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.seed = spec.get("seed", 1930)
        self.max_reply_words = spec.get("max_reply_words")
        self.depth_cue = spec.get("depth_cue", False)
        mix = spec["mix"]
        assert mix and all(w > 0 for w in mix.values()), f"system_prompts.mix needs positive weights: {mix}"
        self.names = list(mix)
        self.weights = [mix[name] for name in self.names]
        self.prompts = {}
        for name in self.names:
            if name == "none":
                continue
            path = PROMPT_DIR / f"{name}.txt"
            assert path.exists(), f"system_prompts.mix names {name!r}, but {path} does not exist"
            self.prompts[name] = path.read_text(encoding="utf-8").strip()

    def num_examples(self):
        return len(self.task)

    def choose(self, index):
        """The prompt name for row `index`: fixed by (seed, index), so reruns and resumes agree."""
        return random.Random(f"{self.seed}:{index}").choices(self.names, weights=self.weights)[0]

    def get_example(self, index):
        conversation = self.task[index]
        messages = conversation["messages"]
        name = self.choose(index)
        if name == "none" or messages[0]["role"] == "system":
            return conversation
        if self.max_reply_words is not None and _reply_words(conversation) > self.max_reply_words:
            first_user = next(m["content"] for m in messages if m["role"] == "user")
            if not (self.depth_cue and DEPTH_CUE_RE.search(first_user)):
                return conversation
        return {**conversation, "messages": [{"role": "system", "content": self.prompts[name]}, *messages]}

    def summary(self, sample=2000):
        """Share of rows that actually get each prompt, estimated from an even sample."""
        n = len(self)
        indices = range(0, n, max(1, n // sample))
        counts = {}
        for index in indices:
            messages = self[index]["messages"]
            name = "none"
            if messages[0]["role"] == "system":
                name = next((k for k, v in self.prompts.items() if v == messages[0]["content"]), "other")
            counts[name] = counts.get(name, 0) + 1
        total = sum(counts.values())
        return {name: round(count / total, 3) for name, count in sorted(counts.items())}
