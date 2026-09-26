"""
Opt-in LoRA for the nanochat GPT. Nothing else in the repo imports this; it exists
so that post-training fits on a single 32 GB card (e.g. an RTX 5090), where a
full fine-tune of d32 does not (params + grads + optimizer state ~45 GB).

    inject_lora(model, r=64, alpha=128)   # freezes the base, wraps the target Linears
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-5)
    ...
    save_checkpoint(..., merged_state_dict(model), ...)   # plain nanochat checkpoint

Targets are the attention q/k/v/o and MLP projections. Embeddings, value
embeddings, gates and lm_head stay frozen. A/B are kept in fp32 and cast to the
activation dtype in forward. B starts at zero, so a fresh adapter is exactly the
parent model.
"""

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULT_TARGETS = ("attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        A, B = self.lora_A.to(x.dtype), self.lora_B.to(x.dtype)
        return self.base(x) + F.linear(F.linear(x, A), B) * self.scaling

    def delta_weight(self):
        return (self.lora_B @ self.lora_A) * self.scaling


def inject_lora(model, r=64, alpha=None, targets=DEFAULT_TARGETS):
    """Freeze the model and wrap every Linear whose dotted name ends with a target."""
    alpha = alpha if alpha is not None else 2 * r
    model.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(name == t or name.endswith("." + t) for t in targets):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(module, r, alpha))
            wrapped.append(name)
    assert wrapped, f"no Linear matched {targets}"
    model.lora_config = {"r": r, "alpha": alpha, "targets": list(targets)}
    return wrapped


def lora_modules(model):
    return [(name, m) for name, m in model.named_modules() if isinstance(m, LoRALinear)]


def lora_parameters(model):
    return [p for name, p in model.named_parameters() if "lora_" in name]


def lora_state_dict(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters() if "lora_" in name}


def load_lora_state_dict(model, state_dict):
    params = dict(model.named_parameters())
    missing = [name for name in params if "lora_" in name and name not in state_dict]
    assert not missing, f"adapter is missing {len(missing)} tensors, e.g. {missing[:3]}"
    with torch.no_grad():
        for name, value in state_dict.items():
            params[name].copy_(value.to(params[name].device, params[name].dtype))


@torch.no_grad()
def merged_state_dict(model, dtype=torch.bfloat16):
    """State dict of the plain GPT with every LoRA delta folded in (merge in fp32),
    so the result loads with the normal checkpoint_manager and runs anywhere."""
    deltas = {name + ".weight": m for name, m in lora_modules(model)}
    out = {}
    for key, value in model.state_dict().items():
        if "lora_" in key:
            continue
        key = key.replace(".base.", ".")
        if key in deltas:
            value = value.float() + deltas[key].delta_weight().float()
        if value.is_floating_point():
            value = value.to(dtype)
        out[key] = value.cpu()
    return out


def enable_activation_checkpointing(model):
    """Recompute each block in backward. Needed for d32 LoRA on 32 GB; inference
    (kv_cache set, or eval mode) is untouched."""
    for block in model.transformer.h:
        if hasattr(block, "_forward_no_ckpt"):
            continue
        block._forward_no_ckpt = block.forward

        def forward(self, x, ve, cos_sin, window_size, kv_cache):
            if self.training and kv_cache is None and torch.is_grad_enabled():
                return checkpoint(self._forward_no_ckpt, x, ve, cos_sin, window_size, kv_cache, use_reentrant=False)
            return self._forward_no_ckpt(x, ve, cos_sin, window_size, kv_cache)

        block.forward = types.MethodType(forward, block)
