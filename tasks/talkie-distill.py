"""
Conversations written by Talkie-1930-13B-IT for off-policy distillation.
https://huggingface.co/datasets/greenpolo/talkie-distill-pre1930

Rows are {"messages": [...]} where only the last assistant turn is a training
target; earlier assistant turns are Talkie's own context. Routes live under
rows/<route>/part-*.jsonl, the held-out rows under heldout/part-*.jsonl.
"""

import json
import os
from functools import lru_cache

from huggingface_hub import HfApi, hf_hub_download

from tasks.common import Task

DATASET = "greenpolo/talkie-distill-pre1930"


@lru_cache(maxsize=None)
def _load_rows(dataset, prefix):
    """dataset is a Hub repo id or a local directory with the same layout."""
    if os.path.isdir(dataset):
        root = os.path.join(dataset, prefix)
        files = sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".jsonl"))
    else:
        names = sorted(f for f in HfApi().list_repo_files(dataset, repo_type="dataset")
                       if f.startswith(prefix + "/") and f.endswith(".jsonl"))
        files = [hf_hub_download(dataset, f, repo_type="dataset") for f in names]
    if not files:
        raise FileNotFoundError(f"no shards under {prefix!r} in {dataset}")
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


class TalkieDistill(Task):

    def __init__(self, route=None, split="train", dataset=DATASET, **kwargs):
        super().__init__(**kwargs)
        if split == "train":
            self.rows = _load_rows(dataset, f"rows/{route}")
        else:
            self.rows = _load_rows(dataset, "heldout")

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        return {"messages": self.rows[index]["messages"]}
