"""
Serve a d32 checkpoint on a small consumer GPU (6-8 GB, e.g. an RTX 2060).

The deployed bf16 export is 5.4 GB, which does not fit beside a CUDA context
and a KV cache on a 6 GB card. But 2.15 GB of it is the 16 value-embedding
tables (padded_vocab x kv_dim each, see GPT.value_embeds), and those are
lookup tables, not matmuls: a decode step reads one row from each. So they can
live in CPU RAM and ship ~64 KB per token to the GPU, while everything that is
actually multiplied (the transformer matrices, wte, lm_head -- ~3.5 GB in bf16)
stays on the card.

Compute stays in bf16 so the numerics match the A10/L4 deployment. Turing
(SM 7.5) has no bf16 tensor cores; PyTorch emulates bf16 there, which costs
little on the decode GEMVs that dominate generation and more on prefill. The
caller must set NANOCHAT_DTYPE=bfloat16 before nanochat.common is imported,
because nanochat.common picks fp32 on pre-Ampere cards and the Engine's KV
cache is bf16 on CUDA regardless, so fp32 compute would fail in attention.

This is an inference loader. Post-training the 2.8B model does not fit on a
card this size (full SFT holds fp32 weights plus optimizer state, ~45 GB).
"""

import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.checkpoint_manager import _patch_missing_config_keys, _patch_missing_keys, find_last_step
from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


class _TokenIdsOnCPU:
    """Shared by every offloaded table so a forward pass copies `idx` to the host once, not 16 times.

    The copy synchronises the stream; doing it per table would drain the GPU
    queue at every value-embedding layer.
    """

    def __init__(self):
        self._idx = None
        self._idx_cpu = None

    def get(self, idx):
        if idx is not self._idx:
            self._idx = idx  # holding the reference keeps `is` from matching a recycled tensor
            self._idx_cpu = idx.to("cpu")
        return self._idx_cpu


class OffloadedEmbedding(nn.Module):
    """nn.Embedding whose table stays in CPU RAM; only the looked-up rows go to the GPU.

    The table is a plain attribute rather than a Parameter, so model.to(device)
    leaves it where it is.
    """

    def __init__(self, weight, ids):
        super().__init__()
        # clone(): the checkpoint is mmapped, and a lookup should not page-fault to disk.
        self.table = weight.detach().to("cpu").clone()
        self.ids = ids

    def forward(self, idx):
        rows = F.embedding(self.ids.get(idx), self.table)
        return rows.to(idx.device, non_blocking=True)


def load_model_small_gpu(checkpoint_dir, step=None, tokenizer_dir=None, device="cuda", verbose=True):
    """Build an eval-mode GPT with value embeddings offloaded to CPU.

    Returns (model, tokenizer, meta_data), like load_model_fast.
    """
    assert COMPUTE_DTYPE == torch.bfloat16, (
        f"COMPUTE_DTYPE is {COMPUTE_DTYPE}; set NANOCHAT_DTYPE=bfloat16 before importing nanochat "
        "so compute matches the deployment and the Engine's bf16 KV cache")
    device = torch.device(device)
    if step is None:
        step = find_last_step(checkpoint_dir)

    t0 = time.time()
    # mmap: the 5.4 GB file is paged in as tensors are touched, instead of read into RAM up front.
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location="cpu", mmap=True)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"), "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)

    with torch.device("meta"):
        model = GPT(model_config)
    model.load_state_dict(model_data, strict=True, assign=True)

    # cos/sin are persistent=False, so they are not in the checkpoint and are still
    # meta tensors here (see fast_load.py). Build them before model.to(), which
    # cannot copy out of meta.
    head_dim = model_config.n_embd // model_config.n_head
    model.cos, model.sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device="cpu")

    # Detach the tables before moving, so model.to() never stages them on the GPU.
    value_embeds = model.value_embeds
    model.value_embeds = nn.ModuleDict()
    model.to(device)
    ids = _TokenIdsOnCPU()
    model.value_embeds = nn.ModuleDict({k: OffloadedEmbedding(ve.weight, ids) for k, ve in value_embeds.items()})
    del value_embeds, model_data
    assert not any(p.is_meta for p in model.parameters()), "a parameter was left on meta"
    model.eval()

    if tokenizer_dir is None and os.path.isdir(os.path.join(checkpoint_dir, "tokenizer")):
        tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")  # the bf16 export ships its tokenizer beside the weights
    tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config "
        f"vocab size {model_config_kwargs['vocab_size']}")

    if verbose:
        on_gpu = sum(p.numel() for p in model.parameters()) / 1e9
        on_cpu = sum(m.table.numel() for m in model.value_embeds.values()) / 1e9
        resident = torch.cuda.memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        print(f"[small_gpu] step {step}: {on_gpu:.2f}B params on {device}, {on_cpu:.2f}B value-embedding "
              f"params in CPU RAM, {resident:.2f} GiB resident, loaded in {time.time() - t0:.1f}s")
    return model, tokenizer, meta_data
