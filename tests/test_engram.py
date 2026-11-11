"""
CPU-only checks for the Engram variants (issue #11, dev/ENGRAM_ABLATION_PLAN.md).

    uv run pytest tests/test_engram.py -v

1. Flags off: same state-dict keys/shapes and same-seed logits as master's nanochat/gpt.py.
2. Each variant: one full forward == prefill + token-by-token KV-cache decode, including
   after KVCache.prefill into a larger batch.
3. Each variant: a few dozen optimisation steps reduce loss, table rows become non-zero,
   and save -> build_model reload gives identical logits.
4. Real d12 shape on the meta device: transformer_matrices + lm_head == 110,100,912 with
   every variant on, the parameter-count asserts pass, estimate_flops ignores the tables.
5. Every ablation config's base_train command carries every Engram key in the config.
"""

import glob
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# CPU only, in fp32, even on a GPU box (must be set before torch / nanochat.common load).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("NANOCHAT_DTYPE", "float32")

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Helpers

TINY = dict(sequence_len=64, vocab_size=256, n_layer=4, n_head=2, n_kv_head=2, n_embd=64, window_pattern="SSSL")
LITE = dict(bigram_embed_factor=3, bigram_lambda_init=0.05)
VALUES = dict(
    engram_table_size=1009, engram_layers="1,3", engram_rank=16, engram_orders="2,3",
    engram_order_weights="2,2", engram_gate_channels=12, engram_decay=0.001, engram_accum_beta=0.99,
)
VARIANTS = {"lite": LITE, "values": VALUES, "both": {**LITE, **VALUES}}

D12 = dict(sequence_len=4096, vocab_size=32768, n_layer=12, n_head=6, n_kv_head=6, n_embd=768, window_pattern="SSSL")
D12_VALUES = dict(
    engram_layers="1,3,5,7", engram_rank=128, engram_orders="2,3", engram_order_weights="2,2",
    engram_gate_channels=12, engram_decay=0.001, engram_accum_beta=0.99,
)
D12_SCALING_PARAMS = 110_100_912

ABLATION_CONFIGS = sorted(glob.glob(str(REPO_ROOT / "configs/base/clean1930s-d12-r12-ctx4096-engram-*-v1.json")))


def build(model_class, config, seed=0, device="cpu"):
    """Same construction path as scripts/base_train.py: meta -> to_empty -> init_weights."""
    torch.manual_seed(seed)
    with torch.device("meta"):
        model = model_class(config)
    model.to_empty(device=device)
    model.init_weights()
    return model


@torch.no_grad()
def randomize(model, seed=1):
    """Make every parameter and table non-trivial (c_proj and the tables start at zero)."""
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.add_((torch.randn(p.shape, generator=g) * 0.05).to(p.dtype))
    if model.engram_bank is not None:
        w = model.engram_bank.weight
        w.copy_((torch.randn(w.shape, generator=g) * 0.5).to(w.dtype))
    if model.bigram_embed is not None:
        w = model.bigram_embed.weight
        w.copy_((torch.randn(w.shape, generator=g) * 0.5).to(w.dtype))


def make_kv_cache(model, batch_size, seq_len):
    c = model.config
    return KVCache(
        batch_size=batch_size, num_heads=c.n_kv_head, seq_len=seq_len,
        head_dim=c.n_embd // c.n_head, num_layers=c.n_layer, device="cpu", dtype=COMPUTE_DTYPE,
    )


def load_master_gpt(tmp_path):
    """master's nanochat/gpt.py as a standalone module."""
    source = subprocess.check_output(["git", "show", "master:nanochat/gpt.py"], cwd=REPO_ROOT, text=True)
    path = tmp_path / "master_gpt.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("master_gpt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# 1. Flags off == master


def test_flags_off_matches_master(tmp_path):
    master = load_master_gpt(tmp_path)
    config = GPTConfig(**TINY)
    ours = build(GPT, config, seed=123)
    theirs = build(master.GPT, master.GPTConfig(**TINY), seed=123)

    ours_sd, theirs_sd = ours.state_dict(), theirs.state_dict()
    assert list(ours_sd) == list(theirs_sd)
    for key in theirs_sd:
        assert ours_sd[key].shape == theirs_sd[key].shape, key
        assert ours_sd[key].dtype == theirs_sd[key].dtype, key
        assert torch.equal(ours_sd[key], theirs_sd[key]), f"same-seed init differs at {key}"
    assert [n for n, _ in ours.named_buffers()] == [n for n, _ in theirs.named_buffers()]

    # master's weights load strictly into the flags-off model (released checkpoints keep loading)
    reloaded = build(GPT, config, seed=999)
    reloaded.load_state_dict(theirs_sd, strict=True)

    idx = torch.randint(0, TINY["vocab_size"], (2, 24), generator=torch.Generator().manual_seed(0))
    ours.eval(), theirs.eval(), reloaded.eval()
    with torch.no_grad():
        ref = theirs(idx)
        assert torch.equal(ours(idx), ref)
        assert torch.equal(reloaded(idx), ref)

    # accounting and optimizer grouping are unchanged too
    assert ours.num_scaling_params() == theirs.num_scaling_params()
    assert ours.estimate_flops() == theirs.estimate_flops()
    describe = lambda opt: [(g["kind"], g["lr"], len(g["params"]), tuple(p.shape for p in g["params"])) for g in opt.param_groups]
    assert describe(ours.setup_optimizer()) == describe(theirs.setup_optimizer())


# -----------------------------------------------------------------------------
# 2. Full forward == KV-cache decode


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_kv_cache_decode_matches_full_forward(variant):
    model = build(GPT, GPTConfig(**TINY, **VARIANTS[variant]))
    randomize(model)
    model.eval()
    V = TINY["vocab_size"]
    g = torch.Generator().manual_seed(7)
    P, N, B = 7, 6, 3
    prompt = torch.randint(0, V, (1, P), generator=g)
    gen = torch.randint(0, V, (B, N), generator=g)

    def decode(break_history=False):
        """Engine.generate_batch's path: batch-1 prefill, prefill() into batch B, then 1-token steps."""
        kv_prefill = make_kv_cache(model, 1, P)
        prefill_logits = model(prompt, kv_cache=kv_prefill)
        kv = make_kv_cache(model, B, P + N)
        kv.prefill(kv_prefill)
        steps = []
        for t in range(N):
            if break_history:
                kv.prev_token, kv.engram_hist = None, None
            steps.append(model(gen[:, t:t + 1], kv_cache=kv)[:, 0])
        return prefill_logits, steps

    with torch.inference_mode():
        prefill_logits, steps = decode()
        torch.testing.assert_close(prefill_logits, model(prompt), rtol=1e-5, atol=1e-5)
        for t in range(N):
            seq = torch.cat([prompt.expand(B, P), gen[:, :t + 1]], dim=1)
            torch.testing.assert_close(steps[t], model(seq)[:, -1], rtol=1e-5, atol=1e-5, msg=f"decode step {t}")

        # The check is not vacuous: losing the n-gram history in the cache changes the logits.
        _, broken = decode(break_history=True)
        assert any(not torch.allclose(a, b, rtol=1e-5, atol=1e-5) for a, b in zip(steps, broken))
        # ...and the variant really feeds the logits.
        seq = torch.cat([prompt.expand(B, P), gen], dim=1)
        full = model(seq)
        if model.engram_bank is not None:
            model.engram_bank.weight.zero_()
        if model.bigram_embed is not None:
            model.bigram_embed.weight.zero_()
        assert not torch.allclose(full, model(seq))


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_chunked_prefill_matches_full_forward(variant):
    """Two multi-token forwards into one cache == one full forward (the PR 747 fix).
    Smear is zeroed here: its KV path ignores prev_embedding for a multi-token chunk,
    a pre-existing limitation unrelated to Engram (Engine never prefills in chunks)."""
    model = build(GPT, GPTConfig(**TINY, **VARIANTS[variant]))
    randomize(model)
    with torch.no_grad():
        model.smear_lambda.zero_()
    model.eval()
    idx = torch.randint(0, TINY["vocab_size"], (2, 20), generator=torch.Generator().manual_seed(5))
    k = 9
    with torch.inference_mode():
        kv = make_kv_cache(model, 2, 20)
        first = model(idx[:, :k], kv_cache=kv)
        second = model(idx[:, k:], kv_cache=kv)
        torch.testing.assert_close(torch.cat([first, second], dim=1), model(idx), rtol=1e-5, atol=1e-5)


def test_kv_cache_reset_and_prefill_carry_history():
    kv1 = make_kv_cache(GPT(GPTConfig(**TINY)), 1, 8)
    kv1.prev_token = torch.tensor([[5]])
    kv1.engram_hist = torch.tensor([[3, 4]])
    kv1.advance(4)
    kv3 = make_kv_cache(GPT(GPTConfig(**TINY)), 3, 16)
    kv3.prefill(kv1)
    assert kv3.prev_token.tolist() == [[5]] * 3
    assert kv3.engram_hist.tolist() == [[3, 4]] * 3
    kv3.prev_token[0, 0] = 9 # rows own their storage
    assert kv3.prev_token[1, 0].item() == 5
    kv3.reset()
    assert kv3.prev_token is None and kv3.engram_hist is None and kv3.get_pos() == 0


def test_chunked_hashes_match_full_sequence():
    """A multi-token forward into a non-empty cache hashes against the cached history
    (PR 747 used the sentinel here)."""
    model = build(GPT, GPTConfig(**TINY, **LITE, **VALUES))
    idx = torch.randint(0, TINY["vocab_size"], (2, 20), generator=torch.Generator().manual_seed(3))
    k = 9
    full = model._bigram_hash(idx)
    assert torch.equal(model._bigram_hash(idx[:, k:], idx[:, k - 1:k]), full[:, k:])
    assert (full[:, 0] == model.bigram_vocab_size - 1).all() # no history: sentinel row

    kv = make_kv_cache(model, 2, 20)
    first = model.engram_buckets(idx[:, :k], kv)
    second = model.engram_buckets(idx[:, k:], kv)
    assert torch.equal(torch.cat([first, second], dim=-1), model.engram_buckets(idx))


# -----------------------------------------------------------------------------
# 3. Training reduces loss, tables fill in, save -> build_model reload is exact


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_training_and_checkpoint_roundtrip(variant, tmp_path, monkeypatch):
    from nanochat import checkpoint_manager
    from nanochat.checkpoint_manager import build_model, save_checkpoint

    config = GPTConfig(**TINY, **VARIANTS[variant])
    model = build(GPT, config)
    optimizer = model.setup_optimizer()
    g = torch.Generator().manual_seed(11)
    x = torch.randint(0, TINY["vocab_size"], (4, 32), generator=g)
    y = torch.randint(0, TINY["vocab_size"], (4, 32), generator=g)

    model.train()
    losses = []
    for _ in range(30):
        loss = model(x, y)
        loss.backward()
        if model.engram_bank is not None:
            model.engram_bank.step(0.2)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5, losses

    if model.engram_bank is not None:
        bank = model.engram_bank
        assert not bank._pending
        written = (bank.weight != 0).any(dim=1).nonzero().flatten()
        assert written.numel() > 0, "no table rows were updated"
        assert (bank.adagrad_accum[written] > 0).all()
        # sparse: only rows hashed from this batch were ever written
        touched = torch.unique(model.engram_buckets(x))
        assert torch.isin(written, touched).all()
    if model.bigram_embed is not None:
        assert (model.bigram_embed.weight != 0).any(dim=1).sum() > 0
        assert not torch.allclose(model.bigram_lambdas, torch.full_like(model.bigram_lambdas, config.bigram_lambda_init))

    model.eval()
    with torch.no_grad():
        ref = model(x)
    save_checkpoint(str(tmp_path), 30, model.state_dict(), None, {"step": 30, "model_config": asdict(config)})

    class StubTokenizer:
        def get_vocab_size(self):
            return config.vocab_size
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda tokenizer_dir=None: StubTokenizer())
    reloaded, _, meta = build_model(str(tmp_path), 30, torch.device("cpu"), "eval")
    assert meta["model_architecture"] == "nanochat_current"
    assert set(reloaded.state_dict()) == set(model.state_dict()) # table and accumulator are in the checkpoint
    with torch.no_grad():
        assert torch.equal(reloaded(x), ref)


# -----------------------------------------------------------------------------
# 4. Real d12 shape: scaling params, count asserts, FLOPs


def meta_model(**overrides):
    with torch.device("meta"):
        return GPT(GPTConfig(**D12, **overrides))


D12_ARMS = {
    "baseline": {},
    "lite": dict(bigram_embed_factor=5, bigram_lambda_init=0.05),
    "values-8m": dict(engram_table_size=8_000_000, **D12_VALUES),
    "values-32m": dict(engram_table_size=32_000_000, **D12_VALUES),
    "values-128m": dict(engram_table_size=128_000_000, **D12_VALUES),
    "values-32m-bigram": dict(engram_table_size=32_000_000, **{**D12_VALUES, "engram_orders": "2", "engram_order_weights": "2", "engram_rank": 64}),
    "lite+values-32m": dict(bigram_embed_factor=5, engram_table_size=32_000_000, **D12_VALUES),
}


def test_d12_scaling_params_and_flops(tmp_path):
    master = load_master_gpt(tmp_path)
    with torch.device("meta"):
        master_model = master.GPT(master.GPTConfig(**D12))
    base = meta_model()
    base_flops = base.estimate_flops()
    assert base_flops == master_model.estimate_flops()

    for arm, overrides in D12_ARMS.items():
        model = meta_model(**overrides)
        counts = model.num_scaling_params() # asserts the per-group counts cover every parameter
        assert counts["transformer_matrices"] + counts["lm_head"] == D12_SCALING_PARAMS, arm
        model.setup_optimizer() # asserts every parameter is in exactly one group
        # The tables never enter the FLOPs estimate; only the engram gate/up matmuls do.
        assert model.estimate_flops() == base_flops + 6 * counts.get("engram", 0), arm

    # table size is invisible to the FLOPs estimate
    assert meta_model(**D12_ARMS["values-8m"]).estimate_flops() == meta_model(**D12_ARMS["values-128m"]).estimate_flops()
    # the values table is a buffer of the documented size: 32M rows x 128 bf16 ~= 8.2 GB
    bank = meta_model(**D12_ARMS["values-32m"]).engram_bank
    assert bank.weight.shape == (bank.table_size, 128) and bank.weight.dtype == torch.bfloat16
    assert bank.table_size <= 32_000_000 and 31_999_000 < bank.table_size
    bigram_bank = meta_model(**D12_ARMS["values-32m-bigram"]).engram_bank
    assert bigram_bank.weight.shape == bank.weight.shape # same table bytes as values-32m
    assert not any(n.startswith("transformer.h.") and ("engram" in n or "bigram" in n)
                   for n, _ in meta_model(**D12_ARMS["lite+values-32m"]).named_parameters())


# -----------------------------------------------------------------------------
# 5. Every config's train command forwards its Engram keys


def test_ablation_configs_exist():
    names = {Path(p).name for p in ABLATION_CONFIGS}
    for arm in ("baseline", "baseline-s43", "lite", "values-32m", "values-8m", "values-128m", "values-32m-bigram"):
        assert f"clean1930s-d12-r12-ctx4096-engram-{arm}-v1.json" in names


@pytest.mark.parametrize("config_path", ABLATION_CONFIGS, ids=lambda p: Path(p).stem)
def test_train_command_has_engram_flags(config_path, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_EXPERIMENT_ROOT", str(tmp_path))
    from scripts.experiment import ENGRAM_TRAINING_ARGS, Experiment

    experiment = Experiment(config_path)
    experiment.validate_config()
    training = experiment.config["training"]
    cmd = experiment._base_train_command({"wandb_run_id": "test"})
    engram_keys = [k for k in training if k.startswith(("engram_", "bigram_"))]
    for key in engram_keys:
        assert f"--{ENGRAM_TRAINING_ARGS[key]}={training[key]}" in cmd, key
    if "baseline" in Path(config_path).stem:
        assert not engram_keys
        assert not any(a.startswith(("--engram-", "--bigram-")) for a in cmd)
    else:
        assert engram_keys

    # shared recipe: d12, ratio 12, 4K SSSL, FP8, 524,288-token batch
    assert training["depth"] == 12 and training["target_param_data_ratio"] == 12
    assert training["scaling_params"] == D12_SCALING_PARAMS
    assert training["max_seq_len"] == 4096 and training["window_pattern"] == "SSSL"
    assert training["fp8"] is True and training["total_batch_size"] == 524_288
    assert experiment.wandb["group"] == "clean1930s-d12-engram-ablation-v1"
    assert "--target-param-data-ratio=12" in cmd and "--fp8" in cmd


def test_base_train_accepts_every_forwarded_flag():
    from scripts.experiment import ENGRAM_TRAINING_ARGS
    source = (REPO_ROOT / "scripts/base_train.py").read_text()
    for cli in ENGRAM_TRAINING_ARGS.values():
        assert f'parser.add_argument("--{cli}"' in source, cli


def test_unknown_engram_key_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_EXPERIMENT_ROOT", str(tmp_path))
    from scripts.experiment import Experiment
    config = json.loads(Path(ABLATION_CONFIGS[0]).read_text())
    config["training"]["engram_table_rows"] = 1000 # typo of engram_table_size
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unknown Engram training keys"):
        Experiment(path).validate_config()
