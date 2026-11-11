# Engram ablations at d12: implementation plan

Brief for the agent implementing [issue #11](https://github.com/zachnorton14/bartholomew-v2/issues/11), "Pre-train Engram ablations".
Written 2026-11-10. Work on the existing `engram-ablations` branch (currently identical to `master`).

## Goal

Find out whether an Engram-style hashed n-gram memory is worth carrying to the d32 Bartholomew model.
The issue asks for a controlled ablation at d12 with a 12:1 token-to-parameter ratio, which is the d32 recipe scaled down.

Your job is the code, configs, tests and runbook that make those runs a one-line launch.
You cannot run the GPU training from this Mac. Tyler launches the runs on a rented H100 and brings back the numbers.

Done means:

1. Two Engram variants exist behind `GPTConfig` flags that default to off.
2. With the flags off, the model is byte-for-byte the current architecture (same state-dict keys, same logits).
3. A matched set of d12 configs and one run script exist for the arms listed below.
4. A CPU-only test file passes and covers the checks under "Acceptance checks".
5. A short runbook section at the bottom of this file says exactly what Tyler runs and what to record.

## Background you need

Engram ([paper](https://arxiv.org/abs/2601.07372)) hashes each position's trailing 2-gram and 3-gram of token ids into rows of a large embedding table, then adds the looked-up vectors into the network.
The idea is that the transformer stops spending early layers reconstructing fixed local patterns.

Upstream nanochat tried a small version in January 2026 and reverted it a day later.
See `dev/LOG.md` lines 406 to 527: at d25 the gain was tiny, vanished on wall clock, and the table bloated VRAM.

Three later implementations are the reference points. The arms below reproduce the first two.

| Reference | Table | Where it is injected | How the table is trained | Reported result |
|---|---|---|---|---|
| [nanochat PR 747](https://github.com/karpathy/nanochat/pull/747) | 163,840 rows x full model width, in VRAM | Added to the residual stream before every block, scaled by a learned per-layer lambda | Dense AdamW | About 0.0003 val bpb better at d22. Karpathy declined to merge the bigram part |
| [nanochat PR 854](https://github.com/karpathy/nanochat/pull/854) | 300M rows x 128, in host RAM (72 GiB in `/dev/shm`) | Gated, projected to model width, added to attention values on 4 layers | Sparse row-wise RMSprop on only the rows a batch touches | 0.7037 vs 0.7180 val bpb at d24, self-reported, PR still open |
| [modded-nanogpt PR 360](https://github.com/KellerJordan/modded-nanogpt/pull/360) | 84.6M rows, bigram plus trigram, sharded over 8 GPUs | Embedding input | Adam on touched rows only | Merged record; author attributes about 7 of 34 seconds saved to the table |

The pattern across them: small dense tables barely help, and the claimed wins come from very large tables that are only affordable with sparse, touched-row updates.

The diffs for PR 747 and PR 854 are small and are your starting point. Fetch them with `gh pr diff 747 --repo karpathy/nanochat` and `gh pr diff 854 --repo karpathy/nanochat`.
Keep upstream's flag and attribute names so the code can be diffed against the PRs, and credit the PRs in the new file's docstring.

## Facts about this repo that constrain the design

- `nanochat/gpt.py` is the `nanochat_current` architecture. `checkpoint_manager.checkpoint_architecture` picks the model class from state-dict keys, and `build_model` loads with `strict=True`. The released d32 checkpoint and `dev/hosting/beam/fast_load.py` both go through `GPTConfig(**model_config)`. New config fields with off defaults are safe. New parameters or persistent buffers that exist when the feature is off are not.
- The training horizon comes from `scaling_params` (transformer matrices plus `lm_head`), which is hardcoded as 110,100,912 in the d12 configs. Keep every Engram module outside `self.transformer.h` so this count, and therefore the horizon, is identical across arms.
- `scripts/experiment.py::_base_train_command` forwards a fixed whitelist of `training` keys (`optional_args`). `validate_config` does not reject unknown keys. An `engram_*` key that is not added to the whitelist is silently dropped and the run trains the baseline.
- `GPT.setup_optimizer`, `num_scaling_params` and `estimate_flops` each assert or assume a complete list of parameter groups. All three need updating.
- Inference uses `nanochat/engine.py::KVCache`, which already carries `prev_embedding` for smear. Any n-gram feature needs the previous token ids at decode time, including through `KVCache.prefill`, which expands a batch-1 cache to `num_samples`. PR 854 never touched `engine.py`, so its decode path is incomplete.
- With `--fp8`, `base_train.py` converts every `nn.Linear` whose dims are multiples of 16 and at least 128. PR 854's `EngramLayer.up` is `nn.Linear(128, 768)` and will be converted. Either build it from the repo's `Linear` class and confirm the filter's behaviour, or exclude Engram modules by name. Check which and say what you did.
- `DistMuonAdamW` reduce-scatters large AdamW params and requires `shape[0]` divisible by world size. The d12 ablations are single-GPU, so this only matters later. Assert single-process for the PR 854 style arm rather than letting per-rank tables diverge.
- The model is compiled with `torch.compile(model, dynamic=False)`. The table lookup uses `torch.unique`, whose output shape varies, so keep it behind `torch._dynamo.disable` as PR 854 does.
- `tests/` was deleted in commit `99437e8`. `pytest` is still in the dev dependency group. Create `tests/test_engram.py`.

## What to build

### Variant A: `lite` (PR 747)

Port the bigram part of PR 747 as is: `bigram_embed_factor` and `bigram_lambda_init` on `GPTConfig`, a zero-initialised `nn.Embedding(vocab_size * factor, n_embd)`, per-layer `bigram_lambdas`, the `prev_token` field on `KVCache`.
Skip the Muon parts of that PR; this repo already has them hardcoded in `nanochat/optim.py`.

One thing to fix while porting: in PR 747, a multi-token forward into a non-empty KV cache hashes its first position against the sentinel instead of `kv_cache.prev_token`. Use the cached token when one exists.

### Variant B: `values` (PR 854 design, device-resident table)

Port the model side of PR 854: `ngram_hash`, prime table sizing, `EngramLayer` (gate on the first 12 residual channels, project rank 128 to model width), injection into attention values, and the row-wise RMSprop update with stochastic rounding to bf16.

Do not port the host-RAM machinery (the `/dev/shm` file, prefetch thread, side CUDA streams).
Keep the table as an ordinary buffer on the model's device. Reasons: it runs on CPU so it can be tested here, the table fits in VRAM at d12, and the first question is whether the mechanism helps at all at equal steps.
The host-RAM path is a follow-up if this arm wins, because that is what d32 would need (d32 trains at device batch size 2, so VRAM is already tight).

Differences from PR 854 that follow from that choice:

- Make the table and its second-moment accumulator persistent buffers, so checkpoints are runnable and resumable. PR 854 does not save the table at all. Note the checkpoint size in the runbook: 32M rows x 128 x 2 bytes is about 8.2 GB.
- Add the n-gram history to `KVCache` properly (`__init__`, `reset`, `prefill`) instead of PR 854's `getattr`.
- Add `--engram-lr` and call the bank's `step()` in `base_train.py` where PR 854 does, scaled by the LR schedule multiplier.

### Wiring

- `scripts/base_train.py`: add the flags, pass them into `GPTConfig` in `build_model_meta`. The d12 reference model built for scaling-law constants must not allocate a table (PR 854 handles this with `if depth == args.depth`; on the meta device nothing is allocated anyway, but keep the reference model Engram-free).
- `scripts/experiment.py`: add the new keys to `optional_args` in `_base_train_command`.
- `nanochat/engram.py`: new file for the hashing, bank and layer.

## Arms

All arms share: dataset `jbduran/think-dataset-clean-1930s`, the reused `clean1930s-d24-r12-ctx4096-fulltok-v1` tokenizer, depth 12, ratio 12, context 4096, `SSSL`, FP8 tensorwise, batch 524,288 tokens, one GPU.
That is 1,321,210,944 tokens, 2,520 steps.

Copy `configs/base/clean1930s-d12-r11.25-ctx4096-sssl-fulltok-ablation-v1.json` and change only the ratio, ids, W&B fields and Engram keys.
Use W&B group `clean1930s-d12-engram-ablation-v1`. Experiment ids are immutable and fingerprinted, so pick final names once.

Wave 1, decides whether anything is worth pursuing:

| Arm | Change from baseline |
|---|---|
| `baseline` | none, seed 42 |
| `baseline-s43` | seed 43, to measure the seed-to-seed gap |
| `lite` | `bigram_embed_factor=5`, `bigram_lambda_init=0.05` |
| `values-32m` | 32M rows, orders `2,3`, two salted hashes per order, rank 128, layers `1,3,5,7`, row LR 0.2, accumulator beta 0.99, decay 0.001 |

Layers `1,3,5,7` are PR 854's `3,7,11,15` halved for d12. They coincide with value-embedding layers, as they do upstream.

Wave 2, only for the variant that beats the seed gap in wave 1:

| Arm | Purpose |
|---|---|
| `values-8m`, `values-128m` | Table size. PR 854 claims gains continue until each row is written only a few dozen times; 128M rows is about 41 writes per row here, 8M is about 660 |
| `values-32m-bigram` | Orders `2` only. Upstream's January test found trigrams did not help |

The 128M arm is a 33 GB table. Save only the final checkpoint for it and mark it as a stretch arm.

Add one script, `runs/clean1930s-d12-r12-ctx4096-engram-ablation.sh <arm>`, modelled on `runs/clean1930s-d12-r11.25-ctx4096-ablation.sh`.

## Acceptance checks

Put these in `tests/test_engram.py`, CPU only, tiny synthetic models and random token ids.

1. Flags off: state-dict keys and shapes equal those of `master`'s `nanochat/gpt.py` (load it with `git show master:nanochat/gpt.py`), and same-seed logits are equal.
2. For each variant: logits from one full forward equal logits from prefill plus token-by-token decode through `KVCache`, including after `prefill` into a larger batch.
3. For each variant: a few dozen optimisation steps reduce loss, table rows become non-zero, and save then `build_model` reload gives identical logits.
4. At the real d12 shape built on the meta device: `transformer_matrices + lm_head` is 110,100,912 with every variant on, the parameter-count asserts pass, and `estimate_flops` is unchanged by the tables.
5. For each config: the command from `_base_train_command` contains every Engram flag in the config.

Run `python -m scripts.experiment plan --config <config>` for each config if it works without network credentials. If it needs `HF_TOKEN`, say so rather than working around it.

## What to record per run (for the runbook)

Final and per-eval val bpb, total training time, tokens per second, peak VRAM, table size in bytes, checkpoint size.

How to read the results:

- A variant counts as helping only if its final val bpb beats `baseline` by more than the gap between `baseline` and `baseline-s43`.
- Then compare at equal wall clock, not equal steps: find the baseline's val bpb at the time the Engram arm finished. This is the test upstream's January attempt failed.
- For d32, report the table's memory per row and what the same writes-per-row would imply at 12 x 1,677,724,672 tokens.

## Out of scope

- The paper's tokenizer compression, context-aware dot-product gate, and causal convolution. Upstream found the gate did not help at this scale, and DeepSeek's later model dropped the convolution.
- Multi-GPU support, the host-RAM table, any d24 or d32 run.
- Changes to SFT, hosting, or the frozen `legacy_gpt.py` and `resformer_gpt.py`.

## Rules

- Do not push, open a PR, or comment on the issue. Commit only if Tyler asks.
- Do not start GPU jobs or spend money.
- If a check fails and you cannot fix it, report it with the output. Do not weaken the check.

## What to hand back

A summary of the changes by file, the test output, the runbook section appended below, and a list of what you could not verify on CPU (FP8 conversion, `torch.compile`, VRAM use, throughput).

## Open items for Tyler

- Zach's engram notes (supplied 2026-11-11) are a walkthrough of the DeepSeek paper. They do not change the arms. How they map onto what is built:
  - Multiple hash heads (notes section 8.1): `values` does this with two salted hashes per n-gram order, each 32 wide, concatenated. They share one table instead of using one table per head. `lite` uses a single hash.
  - Context-aware gating (notes section 8.2): not tested. The `values` gate reads the first 12 residual channels of the current position; it never compares the retrieved vector with the hidden state, which is what the paper's gate does. Karpathy reported the paper's layer as "ok but not great" at d12 ([discussion 472](https://github.com/karpathy/nanochat/discussions/472), from the notes' resource list). A paper-style gate is the obvious extra wave 2 arm if `values` helps.
  - Table size (notes section 5): a collision-free bigram table at this vocabulary would be about 1.07B rows; the arms use 8M to 128M rows shared by bigrams and trigrams, so collisions are the normal case.
  - The notes' last section, "Engrams in nanochat?", is empty. This document is a candidate for it.
- The hosted pretokenised cache (`jbduran/clean1930s-d24-r12-ctx4096-fulltok-v1-pretok`) is not visible to Tyler's Hugging Face account. The Modal launcher falls back to building the cache from the public dataset, as the Colab notebook does.

## Runbook (added with the implementation)

### What was built

- `nanochat/engram.py`: `ngram_hash`, `largest_prime_at_most`, `EngramBank` (device-resident table + row-wise RMSprop with stochastic rounding) and `EngramLayer` (gate on 12 residual channels, rank-to-width projection), ported from PR 854 without the `/dev/shm`, prefetch-thread and side-stream machinery.
- `nanochat/gpt.py`: `lite` (PR 747 bigram residual, `bigram_*` fields) and `values` (PR 854, `engram_*` fields). Both default off and register nothing when off. Engram modules live in `bigram_embed`, `bigram_lambdas`, `engrams.*` and `engram_bank.*`, never under `transformer.h`.
  - `lite` fix over PR 747: a multi-token forward into a non-empty KV cache hashes its first position against `kv_cache.prev_token`, not the sentinel.
  - `values`: the table (`engram_bank.weight`, bf16) and accumulator (`engram_bank.adagrad_accum`, fp32) are persistent buffers, so checkpoints are runnable and resumable. Gates train with the smear group (AdamW); up-projections train with Muon, as in PR 854.
  - Engram inits run after every baseline init, so the baseline weights get the same RNG draws in every arm at a given seed.
- `nanochat/engine.py`: `KVCache.prev_token` and `KVCache.engram_hist`, set in `__init__`, cleared in `reset`, expanded in `prefill`.
- `scripts/base_train.py`: the `--bigram-*` and `--engram-*` flags; the d12 scaling-law reference model is always built Engram-free; `engrams.*` is excluded from FP8 conversion by name (PR 854 calls `F.linear` on those weights, so a `Float8Linear` there would be bypassed anyway; without the exclusion the four 128->768 `up` layers would be converted); `engram_bank.step(engram_lr * lrm)` runs before `optimizer.step()`; single-process and no-fp16 are asserted for `values`.
- `scripts/experiment.py`: `ENGRAM_TRAINING_ARGS` is merged into `optional_args`, and `validate_config` now rejects any `engram_*` or `bigram_*` training key it does not know, so a typo cannot silently train the baseline.
- `configs/base/clean1930s-d12-r12-ctx4096-engram-<arm>-v1.json` for all seven arms, and `runs/clean1930s-d12-r12-ctx4096-engram-ablation.sh <arm>`.
- `tests/test_engram.py`: the five acceptance checks plus chunked-prefill, KV-cache field and config-validation tests.

`values-32m-bigram` drops the trigram slots and halves `engram_rank` to 64. Its two bigram slots then hash to exactly the same buckets as in `values-32m`, the per-slot width stays 32, and the table bytes are identical, so the arm isolates the trigram contribution.

### Running on Modal (added 2026-11-11)

`runs/modal_experiment.py` is the Modal counterpart of `dev/colab_nanochat_experiment.ipynb`: it runs the notebook's base path (prepare, validate, train, optional eval) through `scripts.experiment`. Use it instead of the shell script below when there is no rented box.

What it does on `launch`:

1. Uploads the local working tree as the run's code and deploys the app `bartholomew-experiments`. Runs continue after the terminal is closed.
2. Prepares the token cache once on a CPU container and stores it on the Volume `bartholomew-nanochat`. All arms read that one cache.
3. Starts one container with one pinned H100 (`H100!`, never upgraded to an H200) per arm. Each runs `scripts.gpu_preflight`, then `scripts.experiment train`.

One-time setup (handles credentials, so do it yourself): two Modal secrets, `huggingface-secret` holding `HF_TOKEN` and `wandb-secret` holding `WANDB_API_KEY`. These are the names the Modal dashboard's Hugging Face and Weights & Biases templates create. Both exist in the `tilesos` workspace as of 2026-11-11.

The HF token must be able to write to `jbduran/bart-experiments` and the W&B key must belong to the `jbduran-thinkingmachinesncsu` team. If either is missing the run stops at its first upload.

Commit before launching: the launcher refuses a dirty tree unless `--allow-dirty` is passed, and records the commit in W&B.

Order of runs:

```bash
python3 runs/modal_experiment.py check
```

```bash
python3 runs/modal_experiment.py launch --arm smoke,smoke-values
```

```bash
python3 runs/modal_experiment.py launch --arm baseline,baseline-s43,lite,values-32m
```

`check` needs no credentials and no GPU. The two smoke arms are 20-step runs of the plain model and of the Engram `values` path with a 1M-row table; they exercise FP8, `torch.compile`, checkpoint upload and W&B before the full arms are paid for, and they build the token cache the full arms reuse.

Watching and stopping:

```bash
modal app logs bartholomew-experiments
```

```bash
modal app stop bartholomew-experiments
```

Things that differ from the shell path:

- The image runs Python 3.12, not 3.10, because Modal's image builder no longer offers 3.10. Dependencies still come from `uv.lock`.
- Modal can preempt a GPU container. It restarts the run on the same input and the harness resumes from its last checkpoint (every 500 steps). A resumed run's wall-clock number is less clean, so note any arm that was preempted.
- `--eval bpb` or `--eval full` adds the notebook's eval cell after training. The default is none, because training already logs val bpb at every eval step.

### Before launching (rented box)

1. Commit and push `engram-ablations` (nothing is committed yet), then check that branch out on the H100 box.
2. Put `HF_TOKEN` and `WANDB_API_KEY` in `.env`. The run script needs both.
3. Optional sanity check on the box (forces CPU and fp32, takes seconds):

```bash
uv run pytest tests/test_engram.py -q
```

### Launch

One arm per command, one GPU each. Arms are independent and can run back to back on one box (they share the hosted token cache).

Wave 1:

```bash
bash runs/clean1930s-d12-r12-ctx4096-engram-ablation.sh baseline
```

```bash
bash runs/clean1930s-d12-r12-ctx4096-engram-ablation.sh baseline-s43
```

```bash
bash runs/clean1930s-d12-r12-ctx4096-engram-ablation.sh lite
```

```bash
bash runs/clean1930s-d12-r12-ctx4096-engram-ablation.sh values-32m
```

Wave 2, only if `values-32m` beats the seed gap:

```bash
bash runs/clean1930s-d12-r12-ctx4096-engram-ablation.sh values-8m
```

```bash
bash runs/clean1930s-d12-r12-ctx4096-engram-ablation.sh values-32m-bigram
```

```bash
bash runs/clean1930s-d12-r12-ctx4096-engram-ablation.sh values-128m
```

Every arm should log `Calculated number of iterations from target data:param ratio: 2,520` and `Total number of training tokens: 1,321,205,760` (2,520 x 524,288). If an arm logs a different step count, or a `values` arm does not print `Engram values enabled`, stop it: the config did not reach `base_train`.

### Sizes to expect

| Arm | Table rows (prime) | Table | Accumulator | Writes per row (4 slots, upper bound) | Checkpoints saved |
|---|---|---|---|---|---|
| `lite` | 163,840 x 768 | 0.25 GB | none (AdamW state in the optimizer file) | n/a | every 500 steps |
| `values-8m` | 7,999,993 x 128 | 2.05 GB | 0.13 GB | 661 | every 500 steps |
| `values-32m` | 31,999,939 x 128 | 8.19 GB | 0.51 GB | 165 | every 500 steps |
| `values-32m-bigram` | 31,999,939 x 128 | 8.19 GB | 0.51 GB | 83 (2 slots) | every 500 steps |
| `values-128m` (stretch) | 127,999,981 x 128 | 32.77 GB | 2.05 GB | 41 | final only (`save_every=-1`) |

Each `values` checkpoint is the baseline checkpoint plus table plus accumulator, and the harness uploads every saved step to `jbduran/bart-experiments`. For `values-32m` that is six checkpoints of about 8.7 GB each, about 52 GB uploaded; for `values-128m` one of about 35 GB. Local disk keeps one checkpoint (`keep_local_checkpoints: 1`). The run script asks for 80 GiB free, 150 GiB for `values-128m`.

VRAM: the table and accumulator sit on the GPU on top of the baseline's footprint, so `values-128m` adds about 35 GB. If it runs out of memory, lowering `device_batch_size` needs a new experiment id, because ids are immutable.

### What to record per run

| Field | Where |
|---|---|
| Final val bpb | W&B summary `final_val_bpb`, log `Step 02520 \| Validation bpb` |
| Val bpb at every eval (every 250 steps) | W&B `val/bpb` with `total_training_time` on the same row |
| Total training time | W&B summary `training_time_seconds`, log `Total training time` (excludes the first 10 steps) |
| Tokens per second | W&B `train/tok_per_sec` (take the median after step 10) and summary `final_tok_per_sec` |
| Peak VRAM | W&B summary `peak_memory_mib`, log `Peak memory usage` |
| Table size in bytes | `values`: W&B summary `engram_table_bytes` (table + accumulator) and log `Engram values enabled`; `lite`: log `Engram lite enabled` |
| Checkpoint size | `ls -l` of `model_002520.pt` under the experiment's `base_checkpoints/` |

### How to read the results

- `seed_gap = |bpb(baseline) - bpb(baseline-s43)|` at step 2,520. A variant helps only if `bpb(baseline) - bpb(variant) > seed_gap`.
- Equal wall clock: take the variant's `training_time_seconds`, find the baseline's `val/bpb` at that `total_training_time` (interpolate between its 250-step evals), and compare. A variant slower per step must still win here.
- d32 projection: a row costs 256 bytes of table plus 16 bytes of accumulator. d32 trains on 12 x 1,677,724,672 = 20,132,696,064 tokens, so matching a d12 arm's writes per row with 4 slots needs `rows = 4 x 20.13e9 / writes_per_row`: about 0.49B rows (125 GB table) for `values-32m`'s 165 writes per row, about 1.96B rows (503 GB) for `values-128m`'s 41. Both need the host-RAM path, which is not built.

### Not verified on CPU

- FP8: only checked that the filter leaves `engrams.*` unconverted at the d12 shape (73 layers converted, the same as the baseline). No FP8 matmul ran.
- `torch.compile` on CUDA with the graph break at `EngramBank.lookup`, and the compiled `_fused_update` (it runs eagerly off CUDA). CPU inductor compile of the full model did run in a tiny `base_train` smoke run.
- VRAM use and throughput for every arm.
- Whether the hosted pretokenised cache passes the harness's validation for these ids (unchanged open item).

### Things to know when reading the code

- PR 854's RMSprop has no bias correction. With `engram_accum_beta=0.99` a row's first update is about 10x `engram_lr` in normalised units. This is ported as is.
- `ngram_hash` pads missing history with token id 0, so a real token 0 collides with "no history". Also as upstream.
- Unrelated to Engram: smear's KV path ignores `prev_embedding` for a multi-token chunk, so chunked prefill does not match a full forward for any model. `Engine` never prefills in chunks, so generation is unaffected; the chunked-prefill test zeroes `smear_lambda` to isolate the Engram path.
- Unrelated to Engram: on `--resume-from-step` with the pretokenised loader, the resumed run starts one micro-batch later than the original run did (the saved loader state is taken after the next batch was prefetched). Seen in the CPU smoke run; it only matters for a run that was resumed.
