"""
Talkie-1930-13B-IT (https://huggingface.co/talkie-lm/talkie-1930-13b-it, Apache-2.0)
as a batched generator on one 32 GB GPU.

The 280 block Linears are quantized to torchao int8 weight-only (~14 GB resident,
+0.005 nats of loss vs bf16); embeddings, lm_head and the gain parameters stay
bf16. Blocks are moved and quantized one at a time, so the bf16 model never has
to fit on the GPU. Decode-sized matmuls go through a compiled F.linear that fuses
the dequant (~4x faster decode than eager torchao).

The tokenizer, chat template and top-p filter are small copies of talkie-lm/talkie
@ 35317ba (Apache-2.0); that package requires Python 3.11, this repo pins 3.10.
"""

import base64
import time
import types

import tiktoken
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from dev.talkie_distill.talkie_model import GPTConfig, KVCache, TalkieModel

REPO_ID = "talkie-lm/talkie-1930-13b-it"

# talkie/tokenizer.py
_PAT_STR = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])
BASE_VOCAB_SIZE = 65536
SPECIAL_TOKENS = {
    "<|endoftext|>": BASE_VOCAB_SIZE - 1,
    "<|end|>": BASE_VOCAB_SIZE,
    "<|user|>": BASE_VOCAB_SIZE + 1,
    "<|assistant|>": BASE_VOCAB_SIZE + 2,
    "<|system|>": BASE_VOCAB_SIZE + 3,
}


def build_tokenizer(vocab_path):
    ranks = {}
    with open(vocab_path, "rb") as f:
        for line in f:
            if line.strip():
                token, rank = line.split()
                if int(rank) < BASE_VOCAB_SIZE - 1:
                    ranks[base64.b64decode(token)] = int(rank)
    return tiktoken.Encoding(name="talkie-it", pat_str=_PAT_STR, mergeable_ranks=ranks,
                             special_tokens=SPECIAL_TOKENS)


def format_chat(messages):
    """talkie/chat.py: <|user|>...<|end|><|assistant|>...<|end|>...<|assistant|>"""
    parts = [f"<|{m['role']}|>{m['content']}<|end|>" for m in messages]
    return "".join(parts) + "<|assistant|>"


def top_p_filter(logits, top_p):
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
    drop = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[drop] = float("-inf")
    return torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)


def load_model(checkpoint_path, device, max_seq_len=4096):
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    config = GPTConfig(vocab_size=state_dict["embed.weight"].shape[0])
    with torch.device("meta"):
        model = TalkieModel(config, max_seq_len)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.cos, model.sin = TalkieModel.rotary(max_seq_len, config.head_dim, device="cpu")
    model = model.to(torch.bfloat16).eval().requires_grad_(False)
    for block in model.blocks:
        block.to(device)
        quantize_(block, Int8WeightOnlyConfig())
        torch.cuda.empty_cache()
    model.to(device)
    # fused dequant+matmul for decode steps; prefill stays eager (compute bound anyway)
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    small_linear = torch.compile(lambda x, w: F.linear(x, w), mode="max-autotune-no-cudagraphs", dynamic=False)

    def forward(self, x):
        return small_linear(x, self.weight) if x.shape[:-1].numel() <= 64 else F.linear(x, self.weight)

    for module in model.blocks.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = types.MethodType(forward, module)
    return model


class Teacher:
    def __init__(self, device="cuda"):
        t0 = time.time()
        self.device = torch.device(device)
        self.model = load_model(hf_hub_download(REPO_ID, "rl-refined.pt"), self.device)
        self.tok = build_tokenizer(hf_hub_download(REPO_ID, "vocab.txt"))
        self.pad_id = SPECIAL_TOKENS["<|endoftext|>"]
        self.stop_ids = sorted(SPECIAL_TOKENS.values())
        print(f"Talkie loaded in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def encode_chat(self, messages):
        return self.tok.encode(format_chat(messages), allowed_special="all")

    def decode(self, ids):
        return self.tok.decode(ids)

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=256, temperature=0.8, top_p=0.95, seed=0):
        """Left-padded batched sampling with a static KV cache. Returns the generated
        ids of each row, stop token excluded."""
        device, model = self.device, self.model
        stop = torch.tensor(self.stop_ids, device=device)
        B, P = len(prompts), max(len(p) for p in prompts)
        ids = torch.full((B, P), self.pad_id, dtype=torch.long)
        valid = torch.zeros(B, P, dtype=torch.bool)
        for i, p in enumerate(prompts):
            ids[i, P - len(p):] = torch.as_tensor(p)
            valid[i, P - len(p):] = True
        ids, valid = ids.to(device), valid.to(device)
        cfg = model.config
        cache = KVCache(cfg.n_layer, B, cfg.n_head, P + max_new_tokens, cfg.head_dim, device)
        cache.key_valid[:, :P] = valid
        generator = torch.Generator(device=device).manual_seed(seed)

        def sample(logits):
            logits = logits / temperature
            if top_p < 1:
                logits = top_p_filter(logits, top_p)
            return torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(-1)

        pos = (valid.long().cumsum(1) - 1).clamp(min=0)
        causal = torch.ones(P, P, dtype=torch.bool, device=device).tril()
        mask = (causal[None] & valid[:, None, :]) | torch.eye(P, dtype=torch.bool, device=device)[None]
        h = model(ids, pos_ids=pos, attn_mask=mask[:, None], kv_cache=cache)
        next_pos = pos[:, -1] + 1
        token = sample(model.logits(h[:, -1]))
        out = torch.empty(B, max_new_tokens, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        n = 0
        for step in range(max_new_tokens):
            out[:, step] = token
            done |= torch.isin(token, stop)
            n = step + 1
            if step == max_new_tokens - 1 or (step % 16 == 15 and bool(done.all())):
                break
            cache.key_valid[:, cache.pos] = True
            h = model(token[:, None], pos_ids=next_pos[:, None],
                      attn_mask=cache.key_valid[:, None, None, : cache.pos + 1], kv_cache=cache)
            next_pos = next_pos + 1
            token = torch.where(done, torch.full_like(token, self.pad_id), sample(model.logits(h[:, -1])))
        stop_set = set(self.stop_ids)
        results = []
        for row in out[:, :n].tolist():
            cut = next((j for j, t in enumerate(row) if t in stop_set), len(row))
            results.append(row[:cut])
        return results
