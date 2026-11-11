"""
Engram: hashed n-gram lookup tables injected into attention values.

Port of the model side of https://github.com/karpathy/nanochat/pull/854, with
the table kept on the model's device instead of in host RAM. See
dev/ENGRAM_ABLATION_PLAN.md for why. Notable features:
- one bf16 table (rows x n_engram_layers*per_slot), a persistent buffer on the model device
- each position hashes its trailing n-grams (several salted slots) into rows of the table
- every engram layer reads its slice of the gathered rows, gates it on a few residual
  channels, projects it up to the model dim and adds it to that layer's attention values
- the gathered rows are an autograd leaf; step() applies a row-wise RMSprop update to
  the rows touched since the last step (stochastic rounding back to bf16)
Hashes roll across document boundaries within a packed sequence (known approximation).

Deliberately not ported from PR 854: the /dev/shm table shared by all ranks, the
prefetch worker thread, and the side CUDA streams. Unlike PR 854 the table and its
second-moment accumulator are persistent buffers, so checkpoints hold them and a
resumed run continues with the same table.

The smaller "lite" variant (hashed bigram embedding added to the residual stream,
https://github.com/karpathy/nanochat/pull/747) lives directly in nanochat/gpt.py, as it
does upstream. Both are off by default; with both off the model is exactly the
nanochat_current architecture (same state-dict keys, same logits).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_PRIMES = (1_000_003, 998_244_353, 911_382_323, 805_306_457)


def _fused_update_impl(grad, vals, accum_rows, lr, n_layer, n_embd, eps, decay, accum_beta):
    """Row-wise RMSprop + stochastic rounding to bf16, one kernel."""
    g = grad.view(-1, n_layer, n_embd)
    accum_rows = accum_beta * accum_rows + (1.0 - accum_beta) * g.pow(2).mean(dim=-1)
    scale = lr / (accum_rows.sqrt() + eps)
    x = vals.float() * (1.0 - decay) - (scale.unsqueeze(-1) * g).view(-1, n_layer * n_embd)
    # stochastic rounding: round-to-nearest would drop every update below the bf16 ulp
    noise = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32)
    out = ((x.view(torch.int32) + noise) & -65536).view(torch.float32).to(torch.bfloat16)
    return out, accum_rows


_FUSED_UPDATE = None


def _fused_update(*args):
    """Compiled on CUDA (row count changes every step, hence dynamic=True); eager elsewhere."""
    global _FUSED_UPDATE
    if not args[0].is_cuda:
        return _fused_update_impl(*args)
    if _FUSED_UPDATE is None:
        try:
            _FUSED_UPDATE = torch.compile(_fused_update_impl, dynamic=True)
        except Exception:
            _FUSED_UPDATE = _fused_update_impl
    try:
        return _FUSED_UPDATE(*args)
    except Exception:
        _FUSED_UPDATE = _fused_update_impl
        return _fused_update_impl(*args)


def largest_prime_at_most(n):
    """Prime table sizes spread multiplicative hashes much better than powers of two."""
    def is_prime(m):
        if m < 2 or m % 2 == 0:
            return m == 2
        f = 3
        while f * f <= m:
            if m % f == 0:
                return False
            f += 2
        return True
    n = int(n) | 1
    while n > 2 and not is_prime(n):
        n -= 2
    return n


def ngram_hash(idx, table_size, ngram, salt=0):
    """Hash each position's trailing `ngram` token ids (B, T) into bucket ids in [0, table_size)."""
    assert 1 <= ngram <= len(_PRIMES)
    acc = idx * (_PRIMES[0] + salt * 2_654_435_761)
    for j in range(1, ngram):
        prev = torch.zeros_like(idx) # missing history (row start) hashes as token 0
        if j < idx.size(1):
            prev[:, j:] = idx[:, :-j]
        acc = acc + prev * (_PRIMES[j] + salt * 40_503)
    return acc.remainder(table_size)


class EngramBank(nn.Module):
    """The device-resident table (rows x n_layer*n_embd) and its row-wise optimizer.

    n_layer here is the number of engram layers and n_embd the per-slot width, as in PR 854.
    """

    def __init__(self, table_size, n_layer, n_embd, decay, accum_beta):
        super().__init__()
        self.table_size = table_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.decay = decay
        self.accum_beta = accum_beta
        # Persistent (PR 854 keeps them out of the checkpoint): a checkpoint must carry the
        # table to be runnable, and the accumulator to be resumable.
        self.register_buffer("weight", torch.zeros(table_size, n_layer * n_embd, dtype=torch.bfloat16))
        self.register_buffer("adagrad_accum", torch.zeros(table_size, n_layer, dtype=torch.float32))
        self._pending = [] # (uniq_idx, leaf) per micro-step, consumed by step()

    @torch.no_grad()
    def init_weights(self):
        """Zero table => the engram path is a no-op at step 0."""
        self.weight.zero_()
        self.adagrad_accum.zero_()
        self._pending = []

    def table_bytes(self):
        return self.weight.numel() * self.weight.element_size() + self.adagrad_accum.numel() * self.adagrad_accum.element_size()

    @torch._dynamo.disable # torch.unique has a data-dependent output shape
    def lookup(self, buckets, dtype):
        """buckets (G, B, T) -> rows (G, B, T, n_layer*n_embd); a grad-tracked leaf when training."""
        training = self.training and torch.is_grad_enabled()
        if not training:
            return self.weight[buckets].to(dtype)
        uniq, inv = torch.unique(buckets.reshape(-1), return_inverse=True)
        with torch.no_grad():
            rows = self.weight.index_select(0, uniq).to(dtype)
        leaf = rows.requires_grad_(True)
        self._pending.append((uniq, leaf))
        return leaf[inv].view(*buckets.shape, self.n_layer * self.n_embd)

    @torch.no_grad()
    def step(self, lr, eps=1e-8):
        """Update the rows touched since the last step (grads summed over micro-steps)."""
        if not self._pending:
            return
        device = self.weight.device
        LD = self.n_layer * self.n_embd
        pending, self._pending = self._pending, []
        merged, inv = torch.unique(torch.cat([u for u, _ in pending]), return_inverse=True)
        grad = torch.zeros(merged.numel(), LD, device=device)
        vals = torch.empty(merged.numel(), LD, dtype=pending[0][1].dtype, device=device) # the leaves hold the current row values
        off = 0
        for uniq, leaf in pending:
            sl = inv[off:off + uniq.numel()]
            if leaf.grad is not None:
                grad.index_add_(0, sl, leaf.grad.float())
            vals[sl] = leaf.detach()
            off += uniq.numel()
        accum_rows = self.adagrad_accum[merged]
        new_rows, accum_new = _fused_update(grad, vals, accum_rows, lr, self.n_layer, self.n_embd, eps, self.decay, self.accum_beta)
        self.adagrad_accum[merged] = accum_new.to(self.adagrad_accum.dtype)
        self.weight[merged] = new_rows.to(self.weight.dtype)


class EngramLayer(nn.Module):
    """One layer's view of the bank: gate on the token's residual channels, then project the embedding up."""

    def __init__(self, layer_idx, rank, n_embd, gate_channels):
        super().__init__()
        self.layer_idx = layer_idx # slot index into the bank's per-row layer slices
        self.gate_channels = gate_channels
        self.gate = nn.Linear(gate_channels, 1, bias=False)
        # Called through F.linear below, as in PR 854. base_train.py also excludes
        # `engrams.*` from FP8 conversion by name, so this stays a bf16 matmul.
        self.up = nn.Linear(rank, n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.uniform_(self.gate.weight, 0.0, 0.02)
        torch.nn.init.orthogonal_(self.up.weight) # the embeddings start at zero and first learn through this basis

    def forward(self, x, bank_out):
        # x: (B, T, n_embd), bank_out: (B, T, n_layer, rank)
        e = bank_out[:, :, self.layer_idx]
        gin = x[..., :self.gate_channels]
        g = torch.sigmoid(F.linear(gin, self.gate.weight.to(gin.dtype)) / self.gate_channels) # fan-in norm: step size independent of width
        return F.linear(g * e, self.up.weight.to(e.dtype)) # gate the narrow embedding, then project
