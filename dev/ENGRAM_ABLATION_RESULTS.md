# Engram ablations at d12: wave 1 results

Results for [issue #11](https://github.com/zachnorton14/bartholomew-v2/issues/11). Arms, configs and the reading rules are in `dev/ENGRAM_ABLATION_PLAN.md`.

- Code: `c52502dce820625be4ad406e7709ba2532eed4c0` on `engram-ablations` (W&B `git_commit_sha` matches for all four runs).
- Launched 2026-11-12 15:06 EST with `python3 runs/modal_experiment.py launch --arm baseline,baseline-s43,lite,values-32m`, one pinned H100 80GB HBM3 per arm on Modal.
- W&B: entity `jbduran-thinkingmachinesncsu`, project `think.nano`, group `clean1930s-d12-engram-ablation-v1`.

| Arm | W&B run | Modal function call | Container(s) |
|---|---|---|---|
| `baseline` | `5a2c9314` | `fc-01M78NX6JCFBS4S7HFP1YAXM9T` | `ta-01M78NX82G7QF5NN1SK0AKB9NR` |
| `baseline-s43` | `3e14cd8d` | `fc-01M78NX6NVYWQ54AH4XZT3D7JF` | `ta-01M78NXF7Y62JWZ8W8FT076A5R` |
| `lite` | `fb21d349` | `fc-01M78NX6S3V01THHGMY13H33Y8` | `ta-01M78NXXS6Z4RYZT9KQAE20G5R` |
| `values-32m` | `331a3e01` | `fc-01M78NX6W1JGW16Z78CQ73VKSB` | `ta-01M78NXYGRG9DTWSQNZSC9JXRR` (preempted), then `ta-01M78QSZHPW3X5PKS59TJ0KN1R` |

Every arm logged `Calculated number of iterations from target data:param ratio: 2,520` and `Total number of training tokens: 1,321,205,760`. `lite` logged `Engram lite enabled: bigram table 163,840 x 768 (0.252 GB)`. `values-32m` logged `Engram values enabled: 31,999,939 rows x 128 (8.70 GB table + accumulator on cuda), row lr=0.2`. The baselines logged no Engram line.

## Results

Numbers come from W&B (final summary and `val/bpb` history). The finals, training times and peak VRAM match the Modal logs (`Step 02520 | Validation bpb`, `Total training time`, `Peak memory usage`) to the printed precision.

| Arm | Final val bpb | Baseline minus arm | Training time | Median tok/s after step 10 | Peak VRAM | Engram table bytes | Final checkpoint `model_002520.pt` (+ `optim_002520_rank0.pt`) | Preempted / resumed |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 1.088355 | 0 | 2,717.0 s (45.28 min) | 486,252 | 18,738.5 MiB | none | 792,761,690 B (+ 1,246,165,357 B) | no / no |
| `baseline-s43` | 1.087895 | +0.000460 | 2,760.4 s (46.01 min) | 479,277 | 18,738.5 MiB | none | 792,761,690 B (+ 1,246,165,357 B) | no / no |
| `lite` | 1.082562 | **+0.005793** | 2,828.4 s (47.14 min) | 466,130 | 19,746.0 MiB | 251,658,240 (bf16 table; its AdamW state adds 503,317,932 B to the optimizer file) | 1,044,420,644 B (+ 1,749,483,289 B) | no / no |
| `values-32m` | 1.070675 | **+0.017680** | 2,828.9 s (47.15 min), **preempted run, see below** | 468,099 | 28,543.8 MiB | 8,703,983,408 (table + accumulator) | 9,498,322,093 B (+ 1,247,753,691 B) | **yes, at step 1,580** / no: restarted from step 0 |

- Training time is `training_time_seconds`, which excludes the first 10 steps (compile) and eval time.
- Median tok/s is over every step from 11 to 2,519 in the Modal log. W&B only records `train/tok_per_sec` every 100 steps. The median of those 25 samples is 486,241 / 479,953 / 466,274 / 467,942, within 0.2% of the per-step medians.
- Checkpoint sizes are the files on `jbduran/bartholomew-experiments`. The `values-32m` model file is 8,705,560,403 B larger than the baseline's, which is the 8,703,983,408 B table and accumulator plus the gates and up-projections. Its total upload was 10.7 GB in about 20 minutes after training ended at 16:30:49 EST (the container stayed up, and billed, for that time). `values-32m` saved only the final step (`save_every=-1`); the other three also uploaded checkpoints at 500, 1000, 1500, 2000 and 2500. Saving did not slow training: the 60 steps after each save averaged within 0.5% of each run's median step time.

### Validation bpb at every eval, with training time at that eval

| Step | `baseline` | `baseline-s43` | `lite` | `values-32m` |
|---|---|---|---|---|
| 0 | 3.413367 (0 s) | 3.413610 (0 s) | 3.413367 (0 s) | 3.413367 (0 s) |
| 250 | 1.514041 (259.2 s) | 1.517335 (261.8 s) | 1.483294 (269.0 s) | 1.503445 (271.4 s) |
| 500 | 1.339712 (530.6 s) | 1.346789 (536.5 s) | 1.328176 (551.1 s) | 1.321545 (552.2 s) |
| 750 | 1.260008 (800.9 s) | 1.258164 (810.8 s) | 1.251080 (833.4 s) | 1.239534 (833.5 s) |
| 1000 | 1.237468 (1,072.1 s) | 1.236057 (1,091.0 s) | 1.231244 (1,114.8 s) | 1.218195 (1,117.4 s) |
| 1250 | 1.194680 (1,341.9 s) | 1.193880 (1,364.7 s) | 1.190159 (1,398.2 s) | 1.178391 (1,397.7 s) |
| 1500 | 1.156017 (1,612.0 s) | 1.154198 (1,639.8 s) | 1.149827 (1,680.9 s) | 1.136897 (1,680.2 s) |
| 1750 | 1.142278 (1,883.1 s) | 1.141738 (1,914.1 s) | 1.137651 (1,963.9 s) | 1.122909 (1,962.4 s) |
| 2000 | 1.120758 (2,153.8 s) | 1.119250 (2,189.3 s) | 1.115216 (2,244.9 s) | 1.101996 (2,243.9 s) |
| 2250 | 1.103058 (2,424.2 s) | 1.102238 (2,464.0 s) | 1.097362 (2,525.3 s) | 1.084970 (2,525.3 s) |
| 2500 | 1.088850 (2,695.4 s) | 1.088198 (2,738.6 s) | 1.083120 (2,806.1 s) | 1.071276 (2,806.5 s) |
| 2520 | 1.088355 (2,717.0 s) | 1.087895 (2,760.4 s) | 1.082562 (2,828.4 s) | 1.070675 (2,828.9 s) |

`values-32m` rows are from the second attempt (W&B history holds only that attempt; its first eval is at 20:41:06 UTC, after the restart).

### The `values-32m` preemption

Modal's system log, 15:39:52 EST: `Container terminated due to preemption. Your Function will be restarted with the same input.` The first container had reached step 1,580. Because this arm saves only a final checkpoint, there was nothing to resume from, and the second container trained from step 0. Its 2,828.9 s is therefore one uninterrupted run, not a stitched resume, so it is comparable to the other arms. It did cost about 30 extra H100 minutes, and the arm finished training at 16:30 EST instead of about 16:00. It was not relaunched by hand. The Modal retry was automatic.

The two attempts used the same seed, code and data order, so they show run-to-run nondeterminism on different hosts:

| Step | 250 | 500 | 750 | 1000 | 1250 | 1500 |
|---|---|---|---|---|---|---|
| Attempt 1 (log only) | 1.501604 | 1.321797 | 1.240626 | 1.218182 | 1.178907 | 1.138554 |
| Attempt 2 (W&B) | 1.503445 | 1.321545 | 1.239534 | 1.218195 | 1.178391 | 1.136897 |
| Difference | 0.00184 | 0.00025 | 0.00109 | 0.00001 | 0.00052 | 0.00166 |

## Seed gap

`seed_gap = |1.088355 - 1.087895| = 0.000460`.

Measured against that gap, at equal steps (2,520):

- `lite` beats `baseline` by 0.005793, 12.6 times the gap. **Helps.**
- `values-32m` beats `baseline` by 0.017680, 38.4 times the gap. **Helps.**

The single final-step gap is a thin noise estimate from two runs. Two other measures are larger. The two baselines differ by 0.0005 to 0.0018 at every eval from step 750 on, and same-seed `values-32m` attempts differ by up to 0.0018 mid-run. The conclusion survives a noise floor of 0.002: `values-32m` clears it about 9 times, and `lite` about 3 times.

## Equal wall clock

Both Engram arms took longer than `baseline`: 2,828.4 s (`lite`) and 2,828.9 s (`values-32m`), against 2,717.0 s, about 4.1% longer. `baseline`'s curve ends at 2,717.0 s, so **its val bpb at the time either Engram arm finished cannot be interpolated between its evals**. It finished first, and its last value is 1.088355. Extending it would need a longer-horizon baseline run, whose LR schedule would differ.

Turned around, the comparison has an answer. Each Engram arm's val bpb at 2,717.0 s, interpolated linearly between its step-2,250 and step-2,500 evals, is:

| Arm | Val bpb at 2,717.0 s | `baseline` final (same time) | Margin |
|---|---|---|---|
| `lite` | 1.087637 | 1.088355 | +0.000718 |
| `values-32m` | 1.075633 | 1.088355 | **+0.012722** |

This is a conservative test for the Engram arms. At 2,717 s they are still in LR warmdown. A run with a horizon matched to that time would finish its warmdown and do better than a truncated curve.

- `values-32m` wins at equal wall clock by 0.0127, far above any noise measure here. This is the test upstream's January attempt failed.
- `lite`'s equal-time margin, 0.0007, is above the final seed gap but inside the 0.002 mid-run noise. It is not shown to help at equal wall clock.

Against `baseline-s43`, the slower baseline (finished at 2,760.4 s, 1.087895), the margins are `lite` +0.0025 and `values-32m` +0.0144.

## Throughput and the wall-clock noise floor

`baseline` and `baseline-s43` run identical code. On different H100 hosts their median throughput differs by **1.43%** (486,252 vs 479,277 tok/s) and their training time by 1.60% (2,717.0 vs 2,760.4 s). That is the floor for any wall-clock difference between arms.

| Arm | Median tok/s | vs `baseline` | vs `baseline-s43` | Median step time |
|---|---|---|---|---|
| `baseline` | 486,252 | 0 | +1.46% | 1,078.2 ms |
| `baseline-s43` | 479,277 | -1.43% | 0 | 1,093.9 ms |
| `lite` | 466,130 | -4.14% | -2.74% | 1,124.8 ms |
| `values-32m` | 468,099 | -3.73% | -2.33% | 1,120.0 ms |

The Engram overhead is about 2.3% to 4.1% per step at d12, measured against a 1.4% host-to-host floor. Both variants cost about the same per step. This is in line with the smoke runs (about 1.08 vs 1.14 s/step).

## VRAM and table cost

| Arm | Peak VRAM | Over `baseline` | Table | Per-row cost | Writes per row |
|---|---|---|---|---|---|
| `lite` | 19,746.0 MiB | +1,007.5 MiB | 163,840 x 768 bf16 = 251.7 MB; AdamW state +503.3 MB (in the optimizer file) | 1,536 B table | dense (every row is touched by the optimizer every step) |
| `values-32m` | 28,543.8 MiB | +9,805.3 MiB (10.28 GB) | 31,999,939 x 128 bf16 = 8.19 GB, plus 0.51 GB fp32 accumulator = 8,703,983,408 B | 256 B table + 16 B accumulator = 272 B | 4 slots x 1,321,205,760 / 31,999,939 = 165.2 (upper bound) |

`values-32m` uses about 1.6 GB of VRAM beyond its 8.70 GB table and accumulator, presumably lookup and update buffers; not broken down. On an 80 GB H100, d12 uses 28.5 GB, so `values-128m` (34.8 GB table + accumulator) should fit at about 55 GB if the overhead scales the same way. Not verified.

### d32 projection

d32 trains on 12 x 1,677,724,672 = 20,132,696,064 tokens. Matching `values-32m`'s 165.2 writes per row with 4 slots needs:

- rows = 4 x 20,132,696,064 / 165.15 = **487.6M rows**
- table 487.6M x 256 B = **124.8 GB** bf16, accumulator 487.6M x 16 B = 7.8 GB, **132.6 GB in all**

That does not fit on one 80 GB H100, and d32 already runs at device batch size 2. It needs PR 854's host-RAM path, or a table sharded across GPUs, and neither is built. At the d12 cost the projected table is also a large checkpoint: about 133 GB per save unless the table is saved separately or less often. Keeping the d12 table size (32M rows, 8.7 GB) at d32 is affordable in VRAM terms, but it means about 2,500 writes per row, which is the regime PR 854 reports as weaker.

`lite` at d32 width (2,048) would be 163,840 x 2,048 bf16 = 0.67 GB plus its AdamW state. That is cheap, but it showed no equal-wall-clock win here.

## Recommendation

**Run wave 2 for `values`: `values-8m`, `values-32m-bigram`, and `values-128m` as the stretch arm.**

- `values-32m` is the only variant that wins both tests. At equal steps it is 0.0177 bpb better, 38 times the seed gap and about 9 times the noisiest gap seen. At equal wall clock it is at least 0.0127 better, while costing about 4% throughput. Its lead is steady at 0.016 to 0.020 from step 500 on, rather than appearing only at the end.
- Wave 2 answers what d32 needs to know. `values-8m` vs `values-32m` (vs `values-128m`) shows whether gains keep growing with rows, i.e. whether writes per row is the right scaling rule for sizing the d32 table. `values-32m-bigram` shows whether the trigram slots earn their place.
- Do not carry `lite` forward on this evidence. It helps at equal steps (0.0058), but its 0.0007 equal-wall-clock margin is inside the noise, which matches upstream's January finding.
- Before treating any wave-2 difference under about 0.002 as real, consider a `values-32m` seed-43 rerun. A new config and id would be needed. Same-seed reruns here differed by up to 0.0018 mid-run.
- Cost note for `values-128m`: it is configured with `save_every=-1`, so a preemption, as happened here, restarts it from step 0 (about 50 minutes on an H100, plus a 35 GB final upload). Changing that to `save_every: 1000` before its first launch trades about 70 GB of extra uploads for preemption safety. It has never been launched, so its id can still change.

## Not verified

- Equal wall clock at the Engram arms' finish times: not measurable, because `baseline` finished earlier. The reverse comparison above is used instead.
- The source of the roughly 1.6 GB of `values-32m` VRAM beyond its table and accumulator.
- Whether attempt 1 of `values-32m` and the restart ran on the same kind of host (both report `NVIDIA H100 80GB HBM3`).
- The plan's "Arms" section says 1,321,210,944 tokens. Every run logged 1,321,205,760 (2,520 x 524,288), as the runbook says. This is a typo in the plan; nothing was affected.

## Draft comment for issue #11 (not posted)

> **Engram wave 1 at d12 (ratio 12, 2,520 steps, one H100 each, commit `c52502d`)**
>
> | Arm | Final val bpb | vs baseline | Train time | tok/s | Peak VRAM |
> |---|---|---|---|---|---|
> | baseline (seed 42) | 1.088355 | n/a | 45.3 min | 486k | 18.7 GiB |
> | baseline (seed 43) | 1.087895 | -0.0005 | 46.0 min | 479k | 18.7 GiB |
> | lite (PR 747 bigram) | 1.082562 | **-0.0058** | 47.1 min | 466k | 19.7 GiB |
> | values-32m (PR 854 style, 32M x 128 table on GPU) | 1.070675 | **-0.0177** | 47.2 min* | 468k | 28.5 GiB |
>
> \*values-32m was preempted by Modal at step 1,580 and restarted from step 0. The time shown is the clean second run.
>
> - Seed gap: 0.00046. Mid-run gaps between the baselines, and between two same-seed values-32m attempts, reach about 0.0018.
> - Equal wall clock: both Engram arms took about 4% longer than baseline. Cut off at baseline's finish time (still in warmdown), values-32m is at 1.0756, 0.0127 better than baseline's final. lite is at 1.0876, only 0.0007 better, inside the noise.
> - Cost: values-32m adds 8.7 GB of table and accumulator and about 10.3 GB of VRAM at d12. Matching its 165 writes per row at d32 would need about 488M rows, about 133 GB, so it needs the host-RAM path.
>
> Proposal: run wave 2 for values (8M and 128M rows, and bigram only) to see how the gain scales with table size, and drop lite. Full numbers are in `dev/ENGRAM_ABLATION_RESULTS.md`.
