"""
Off-policy distillation: supervised fine-tuning of an SFT model on conversations
written by a stronger teacher (Talkie-1930-13B-IT, see dev/talkie_distill/),
mixed with authentic pre-1930 replay so the parent's behaviour is kept.

LoRA only for now: it is what fits a d32 on one 32 GB card. Each optimizer step
draws a fixed number of rows, a fixed fraction of them teacher rows (sampled with
replacement, weighted per route), the rest replay. The loss is the mean per-row
token NLL of each source, mixed by that fraction, so long answers do not dominate.
Only the last assistant turn of a row is a target.

Intermediate saves are the adapter + optimizer (adapter_<step>.pt, for resume). The
final save is also the merged model as a plain nanochat checkpoint (bf16), so
chat_eval / chat_cli / the harness load it like any other.

1 GPU:
python -m scripts.chat_distill --sft-checkpoint-dir=... --sft-step=42
"""

import argparse
import json
import os
import random
import time
from importlib import import_module

import torch
import wandb

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_model_from_checkpoint_dir
from nanochat.experiment_metrics import (
    compute_log_fields,
    configure_wandb_metrics,
    cumulative_pipeline_flops,
    training_flops,
    update_wandb_compute_summary,
    update_wandb_lineage_summary,
)
from nanochat.lora import (
    inject_lora,
    lora_parameters,
    lora_state_dict,
    load_lora_state_dict,
    merged_state_dict,
    enable_activation_checkpointing,
)

TalkieDistill = import_module("tasks.talkie-distill").TalkieDistill
AuthenticPre1930 = import_module("tasks.authentic-pre1930").AuthenticPre1930

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Off-policy distillation (LoRA SFT on teacher conversations)")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--wandb-run-id", type=str, default="")
parser.add_argument("--wandb-group", type=str, default="")
parser.add_argument("--wandb-tags", type=str, default="")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--sft-checkpoint-dir", type=str, default=None)
parser.add_argument("--sft-step", type=int, default=None)
parser.add_argument("--checkpoint-dir", type=str, default=None)
parser.add_argument("--tokenizer-dir", type=str, default=None)
parser.add_argument("--resume-from-step", type=int, default=None)
parser.add_argument("--experiment-id", type=str, default="")
parser.add_argument("--experiment-config", type=str, default="")
parser.add_argument("--parent-cumulative-flops", type=float, default=0.0)
parser.add_argument("--tokenizer-fingerprint", type=str, default="")
parser.add_argument("--git-commit-sha", type=str, default="")
# Data
parser.add_argument("--dataset", type=str, default="greenpolo/talkie-distill-pre1930", help="Hub repo id or local dir")
parser.add_argument("--route-weights", type=str, default="questions=1,messy=1,casual=1,multiturn=1,sft_questions=1,fragments=3",
                    help="relative sampling weight per route (0 excludes a route)")
parser.add_argument("--teacher-fraction", type=float, default=0.6, help="fraction of each step's rows from the teacher; the rest is replay")
parser.add_argument("--replay-substantive", type=int, default=1, help="replay only conversations with real answers (see substantive())")
parser.add_argument("--max-seq-len", type=int, default=640, help="rows longer than this are dropped, never truncated")
parser.add_argument("--max-answer-tokens", type=int, default=256)
# LoRA
parser.add_argument("--lora-rank", type=int, default=64)
parser.add_argument("--lora-alpha", type=int, default=128)
parser.add_argument("--activation-checkpointing", type=int, default=1, help="recompute blocks in backward (needed for d32 on 32 GB)")
# Optimization
parser.add_argument("--num-iterations", type=int, default=800)
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--rows-per-step", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=17)
# Evaluation / saving
parser.add_argument("--eval-every", type=int, default=100, help="held-out teacher-row CE every N steps")
parser.add_argument("--save-every", type=int, default=100)
args = parser.parse_args()
user_config = vars(args).copy()
if args.experiment_config and os.path.exists(args.experiment_config):
    with open(args.experiment_config, "r", encoding="utf-8") as f:
        user_config["resolved_experiment_config"] = json.load(f)
    experiment = user_config["resolved_experiment_config"]
    user_config.update({
        "stage": "posttrain",
        "base_experiment_id": experiment.get("parent", {}).get("base_experiment_id"),
        "parent_experiment_id": experiment.get("parent", {}).get("sft_experiment_id"),
        "parent_checkpoint_step": experiment.get("parent", {}).get("checkpoint_step"),
        "config_fingerprint": experiment.get("config_fingerprint"),
    })
    for key, value in experiment.get("data", {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
assert not ddp, "chat_distill is single-GPU"
random.seed(args.seed)
torch.manual_seed(args.seed)

# wandb logging init
use_dummy_wandb = args.run == "dummy"
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    entity=os.environ.get("WANDB_ENTITY"),
    project=os.environ.get("WANDB_PROJECT", "think.nano"),
    name=args.run,
    id=args.wandb_run_id or None,
    resume="allow",
    group=args.wandb_group or None,
    tags=[tag for tag in args.wandb_tags.split(",") if tag],
    config=user_config,
)
if not use_dummy_wandb:
    configure_wandb_metrics(wandb_run)
    update_wandb_lineage_summary(wandb_run, user_config, args.experiment_id or args.run)

# Init model from the exact SFT parent. The frozen base runs in bf16; LoRA A/B stay fp32.
if args.sft_checkpoint_dir:
    model, tokenizer, meta = load_model_from_checkpoint_dir(
        args.sft_checkpoint_dir, device, phase="train", step=args.sft_step, tokenizer_dir=args.tokenizer_dir,
    )
else:
    model, tokenizer, meta = load_model("sft", device, phase="train", model_tag=args.model_tag, step=args.model_step)
model.to(torch.bfloat16)
num_flops_per_token = model.estimate_flops()
inject_lora(model, r=args.lora_rank, alpha=args.lora_alpha)
if args.activation_checkpointing:
    enable_activation_checkpointing(model)
optimizer = torch.optim.AdamW(lora_parameters(model), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
print0(f"LoRA r={args.lora_rank} alpha={args.lora_alpha}: {sum(p.numel() for p in lora_parameters(model)):,} trainable parameters")

base_dir = get_base_dir()
checkpoint_dir = args.checkpoint_dir or os.path.join(base_dir, "chatdistill_checkpoints", args.model_tag or f"d{model.config.n_layer}")
start_step = 0
if args.resume_from_step is not None:
    blob = torch.load(os.path.join(checkpoint_dir, f"adapter_{args.resume_from_step:06d}.pt"), map_location=device)
    load_lora_state_dict(model, blob["lora"])
    optimizer.load_state_dict(blob["optimizer"])
    random.setstate(blob["python_rng"])
    start_step = args.resume_from_step
    print0(f"Resumed at step {start_step}")

# -----------------------------------------------------------------------------
# Data: (inputs, targets) per row; targets are -1 everywhere but the last answer + <|assistant_end|>

assistant_end = tokenizer.encode_special("<|assistant_end|>")


def encode_row(conversation):
    if conversation["messages"][-1]["role"] != "assistant":
        return None
    prompt = tokenizer.render_for_completion(conversation)
    answer = tokenizer.encode(conversation["messages"][-1]["content"]) + [assistant_end]
    if len(answer) > args.max_answer_tokens or len(prompt) + len(answer) > args.max_seq_len:
        return None
    ids = prompt + answer
    targets = [-1] * (len(prompt) - 1) + answer
    return torch.tensor(ids[:-1]), torch.tensor(targets)


def encode_task(task, keep=lambda conversation: True):
    rows = [encode_row(c) if keep(c) else None for c in (task[i] for i in range(len(task)))]
    return [r for r in rows if r is not None], sum(r is None for r in rows)


def substantive(conversation):
    """Most authentic-pre1930 turns are a few words of dialogue. Replaying all of them
    shortens every answer, so keep conversations whose user turns have >= 4 words,
    assistant turns >= 6, and >= 24 assistant words in total."""
    messages = [m for m in conversation["messages"] if m["role"] != "system"]
    counts = [(m["role"], len(m["content"].split())) for m in messages]
    return (all(n >= (4 if role == "user" else 6) for role, n in counts)
            and sum(n for role, n in counts if role == "assistant") >= 24)


weights = {route: int(w) for route, w in (kv.split("=") for kv in args.route_weights.split(","))}
teacher_rows = []
for route, weight in weights.items():
    rows, dropped = encode_task(TalkieDistill(route, dataset=args.dataset))
    teacher_rows += rows * weight
    print0(f"route {route}: {len(rows):,} rows x{weight} ({dropped} over length)")
replay_rows, dropped = encode_task(AuthenticPre1930(split="train"), keep=substantive if args.replay_substantive else (lambda c: True))
print0(f"replay authentic-pre1930: {len(replay_rows):,} rows ({dropped} filtered or over length)")
heldout_rows, _ = encode_task(TalkieDistill(split="heldout", dataset=args.dataset))


def collate(rows):
    width = max(len(x) for x, _ in rows)
    inputs = torch.zeros(len(rows), width, dtype=torch.long)
    targets = torch.full((len(rows), width), -1, dtype=torch.long)
    for i, (x, y) in enumerate(rows):
        inputs[i, :len(x)], targets[i, :len(y)] = x, y
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def heldout_ce():
    model.eval()
    total, count = 0.0, 0
    for b0 in range(0, len(heldout_rows), args.device_batch_size):
        inputs, targets = collate(heldout_rows[b0:b0 + args.device_batch_size])
        loss = model(inputs, targets, loss_reduction="none").view_as(targets)
        valid = targets != -1
        total += float(loss[valid].sum())
        count += int(valid.sum())
    model.train()
    return total / count


def save(step, final=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {"lora": lora_state_dict(model), "lora_config": model.lora_config, "optimizer": optimizer.state_dict(),
             "python_rng": random.getstate()}
    torch.save(state, os.path.join(checkpoint_dir, f"adapter_{step:06d}.pt"))
    if final:
        # the adapter + optimizer double as the rank-0 optimizer shard the harness expects
        save_checkpoint(checkpoint_dir, step, merged_state_dict(model), state, {
            "model_config": model.config.__dict__,
            "user_config": user_config,
            "lora": model.lora_config,
            "loop_state": {
                "step": step,
                "stage_training_flops": stage_training_flops,
                "inherited_parent_flops": args.parent_cumulative_flops,
                "cumulative_pipeline_training_flops": cumulative_pipeline_flops(stage_training_flops, args.parent_cumulative_flops),
            },
        })
    print0(f"Saved step {step} to {checkpoint_dir}")


# -----------------------------------------------------------------------------
# Training loop
assert args.rows_per_step % args.device_batch_size == 0
grad_accum_steps = args.rows_per_step // args.device_batch_size
stage_training_flops = 0.0
model.train()
if start_step == 0:
    ce = heldout_ce()
    print0(f"Step 0 | heldout CE: {ce:.4f}")
    wandb_run.log({**compute_log_fields(0, 0.0, args.parent_cumulative_flops), "heldout_ce": ce})
t0 = time.time()
for step in range(start_step + 1, args.num_iterations + 1):
    n_teacher = round(step * args.rows_per_step * args.teacher_fraction) - round((step - 1) * args.rows_per_step * args.teacher_fraction)
    n_replay = args.rows_per_step - n_teacher
    batch = [(random.choice(teacher_rows), True) for _ in range(n_teacher)] + [(random.choice(replay_rows), False) for _ in range(n_replay)]
    random.shuffle(batch)
    step_loss = 0.0
    for micro in range(grad_accum_steps):
        chunk = batch[micro * args.device_batch_size:(micro + 1) * args.device_batch_size]
        inputs, targets = collate([row for row, _ in chunk])
        token_loss = model(inputs, targets, loss_reduction="none").view_as(targets)
        valid = targets != -1
        row_loss = token_loss.masked_fill(~valid, 0).sum(1) / valid.sum(1)
        is_teacher = torch.tensor([t for _, t in chunk], device=device)
        loss = (args.teacher_fraction * row_loss[is_teacher].sum() / max(n_teacher, 1)
                + (1 - args.teacher_fraction) * row_loss[~is_teacher].sum() / max(n_replay, 1))
        loss.backward()
        step_loss += loss.item()
        stage_training_flops += training_flops(inputs.numel(), num_flops_per_token)
    torch.nn.utils.clip_grad_norm_(lora_parameters(model), args.grad_clip)
    lr = args.lr * min(step / max(args.warmup_steps, 1), 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if step % 10 == 0:
        print0(f"Step {step}/{args.num_iterations} | loss: {step_loss:.4f} | lr: {lr:.2e} | {(time.time() - t0) / (step - start_step):.2f}s/step")
    log = {**compute_log_fields(step, stage_training_flops, args.parent_cumulative_flops), "loss": step_loss, "lr": lr}
    if step % args.eval_every == 0 or step == args.num_iterations:
        log["heldout_ce"] = heldout_ce()
        print0(f"Step {step} | heldout CE: {log['heldout_ce']:.4f}")
    wandb_run.log(log)
    if step == args.num_iterations:
        save(step, final=True)
    elif step % args.save_every == 0:
        save(step)

if not use_dummy_wandb:
    update_wandb_compute_summary(wandb_run, compute_log_fields(args.num_iterations, stage_training_flops, args.parent_cumulative_flops))
wandb_run.finish()
compute_cleanup()
