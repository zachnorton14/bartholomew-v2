"""
The released Bartholomew SFT as a teacher, for the self-distillation control: same
interface as teacher.Teacher, so generate.py runs the identical pipeline with either.

nanochat's Engine samples copies of one prompt; this samples many different prompts
at once (left-padded, static KV cache, per-row stop), which is ~50x faster for data
generation. It re-implements GPT.forward with a per-row key mask using the model's own
submodules. Exact because RoPE is relative (shifting a row by its padding leaves q.k
unchanged), smear is zeroed at each row's first real token, and rows stay under the
1024-token short window where every layer is full-context.
"""

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.gpt import apply_rotary_emb, norm


def _attn(attn, x, ve, cos, sin, kc, vc, pos, mask):
    B, T, _ = x.shape
    H, D = attn.n_head, attn.head_dim
    q = attn.c_q(x).view(B, T, H, D)
    k = attn.c_k(x).view(B, T, attn.n_kv_head, D)
    v = attn.c_v(x).view(B, T, attn.n_kv_head, D)
    if ve is not None:
        ve = ve.view(B, T, attn.n_kv_head, D)
        gate = 3 * torch.sigmoid(attn.ve_gate(x[..., : attn.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * ve
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q) * 1.2, norm(k) * 1.2
    kc[:, pos: pos + T] = k
    vc[:, pos: pos + T] = v
    y = F.scaled_dot_product_attention(q.transpose(1, 2), kc[:, : pos + T].transpose(1, 2),
                                       vc[:, : pos + T].transpose(1, 2), attn_mask=mask,
                                       enable_gqa=attn.n_head != attn.n_kv_head)
    return attn.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


class _Cache:
    def __init__(self, model, B, T, device):
        c = model.config
        D = c.n_embd // c.n_head
        dt = model.cos.dtype
        self.k = torch.empty(c.n_layer, B, T, c.n_kv_head, D, device=device, dtype=dt)
        self.v = torch.empty_like(self.k)
        self.key_valid = torch.zeros(B, T, dtype=torch.bool, device=device)
        self.prev_emb = None
        self.pos = 0


def _forward(model, idx, cache: _Cache, valid_prev=None):
    """idx (B,T) at cache.pos. valid_prev (B,T-1) bool: whether position t-1 is a real token (prefill only).
    Returns last-position logits (B, V) fp32, softcapped exactly like GPT.forward."""
    B, T = idx.shape
    pos = cache.pos
    cos, sin = model.cos[:, pos: pos + T], model.sin[:, pos: pos + T]
    x = norm(model.transformer.wte(idx).to(model.cos.dtype))
    prev = cache.prev_emb
    cache.prev_emb = x[:, -1:, :]
    if T > 1:
        gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
        if valid_prev is not None:
            gate = gate * valid_prev[:, :, None].to(x.dtype)
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    elif prev is not None:
        gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, :, :24]))
        x = x + gate * prev
    kv = cache.key_valid[:, : pos + T]
    if T > 1:
        causal = torch.ones(T, pos + T, dtype=torch.bool, device=idx.device).tril(diagonal=pos)
        mask = causal[None] & kv[:, None, :]
        eye = torch.zeros(T, pos + T, dtype=torch.bool, device=idx.device)
        eye[:, pos:] = torch.eye(T, dtype=torch.bool, device=idx.device)
        mask |= eye[None]  # pad queries attend to themselves (avoids all-masked softmax rows)
        mask = mask[:, None]
    else:
        mask = kv[:, None, None, :]
    x0 = x
    n_layer = model.config.n_layer
    x_back = None
    for i, blk in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
        x = x + _attn(blk.attn, norm(x), ve, cos, sin, cache.k[i], cache.v[i], pos, mask)
        x = x + blk.mlp(norm(x))
        if i == n_layer // 2:
            x_back = x
    x = x - model.backout_lambda.to(x.dtype) * x_back
    x = norm(x[:, -1])
    logits = model.lm_head(x)[..., : model.config.vocab_size].float()
    cache.pos += T
    return 15 * torch.tanh(logits / 15)


@torch.no_grad()
def sample_batch(model, prompts, max_new_tokens=256, temperature=1.0, top_k=None, stop_ids=(), seed=0,
                 pad_id=0, max_total=1024):
    """prompts: list of token-id lists (ragged). Returns list of generated id lists; each row is cut at (and
    includes) its first stop token if one was produced. Rows that hit max_new_tokens have no stop token."""
    dev = model.get_device()
    B, P = len(prompts), max(len(p) for p in prompts)
    assert P + max_new_tokens <= max_total, (P, max_new_tokens)
    ids = torch.full((B, P), pad_id, dtype=torch.long)
    valid = torch.zeros(B, P, dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, P - len(p):] = torch.as_tensor(p)
        valid[i, P - len(p):] = True
    ids, valid = ids.to(dev), valid.to(dev)
    cache = _Cache(model, B, P + max_new_tokens, dev)
    cache.key_valid[:, :P] = valid
    gen = torch.Generator(device=dev).manual_seed(seed)
    stop = torch.tensor(sorted(stop_ids), device=dev, dtype=torch.long) if stop_ids else None

    def sample(lg):
        if temperature <= 0:
            return lg.argmax(-1)
        lg = lg / temperature
        if top_k:
            v, _ = torch.topk(lg, top_k)
            lg = lg.masked_fill(lg < v[:, -1:], float("-inf"))
        return torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).squeeze(-1)

    tok = sample(_forward(model, ids, cache, valid_prev=valid[:, :-1]))
    out = torch.empty(B, max_new_tokens, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    n = 0
    for step in range(max_new_tokens):
        out[:, step] = tok
        if stop is not None:
            done |= torch.isin(tok, stop)
        n = step + 1
        if step == max_new_tokens - 1 or (step % 16 == 15 and bool(done.all())):
            break
        cache.key_valid[:, cache.pos] = True
        tok = sample(_forward(model, tok[:, None], cache))
        tok = torch.where(done, torch.full_like(tok, pad_id), tok)
    stop_set = set(stop_ids)
    res = []
    for row in out[:, :n].tolist():
        cut = next((j + 1 for j, t in enumerate(row) if t in stop_set), len(row))
        res.append(row[:cut])
    return res


class BartTeacher:
    max_seq_len = 1024  # the SSSL short window; sample_batch asserts it

    def __init__(self, device="cuda", top_k=50):
        self.model, self.tokenizer, _ = load_model("sft", torch.device(device), phase="eval")
        self.model.to(torch.bfloat16).eval()
        self.assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        self.top_k = top_k

    def encode_chat(self, messages):
        return self.tokenizer.render_for_completion({"messages": list(messages) + [{"role": "assistant", "content": ""}]})

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def generate(self, prompts, max_new_tokens=256, temperature=0.6, top_p=None, seed=0):
        """top_p is ignored: Bartholomew's chat default is top-k 50."""
        outs = sample_batch(self.model, prompts, max_new_tokens=max_new_tokens, temperature=temperature,
                            top_k=self.top_k, stop_ids=[self.assistant_end], seed=seed)
        return [o[:-1] if o and o[-1] == self.assistant_end else o for o in outs]
