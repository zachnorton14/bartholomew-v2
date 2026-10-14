"""Retrain Bartholomew's SFT stage on Modal, e.g. with system prompts in the data.

    modal run --detach modal_train.py --iterations 5    # smoke test: every stage, a few dollars
    modal run --detach modal_train.py
    modal run --detach modal_train.py --config configs/sft/<name>.json --run <run-name>

Always pass --detach: the entrypoint spawns the run and returns at once, and
without --detach the App would stop with it.

This runs scripts.chat_sft directly instead of the experiment harness, so
nothing is uploaded to Hugging Face or W&B. The base checkpoint is read from the
public experiments repo and cached in the private `bartholomew-checkpoints`
Volume; every output lands there too, under /vol/runs/<run>/:

    config.json    the config the run used, with its parent recorded
    checkpoints/   chat_sft output (model_XXXXXX.pt, meta_XXXXXX.json, metrics)
    bf16/          serving export, testable with
                   modal run modal_eval.py --checkpoint-dir /vol/runs/<run>/bf16 --tag <run>
    train.log

A run name is never reused once it has checkpoints: pick a new one rather than
overwrite a finished run.
"""

import json
import os
from pathlib import Path

import modal


APP_NAME = "bartholomew-sft"
# Override with MODAL_TRAIN_GPU=H100 (faster, and FA3 kernels load on Hopper).
TRAIN_GPU = os.environ.get("MODAL_TRAIN_GPU", "A100-80GB")
MODEL_REPO = "jbduran/bart-experiments"
BASE_EXPERIMENT_ID = "Think.Unbounded-d32-v2mix-cont"
BASE_STEP = 9600  # the step the deployed C3-robust-v2 SFT started from (its run.json)
DEFAULT_CONFIG = "configs/sft/pre1930-curriculum-c3-robust-v2-sysprompt.json"

repo_root = Path(__file__).parent
_ignore = ["**/__pycache__", "**/*.pyc"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject(str(repo_root / "pyproject.toml"))
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
          "TOKENIZERS_PARALLELISM": "false", "NANOCHAT_BASE_DIR": "/root/.cache/nanochat"})
    .add_local_dir(repo_root / "nanochat", "/root/nanochat", ignore=_ignore)
    .add_local_dir(repo_root / "scripts", "/root/scripts", ignore=_ignore)
    .add_local_dir(repo_root / "tasks", "/root/tasks", ignore=_ignore)
    .add_local_dir(repo_root / "configs", "/root/configs", ignore=_ignore)
    .add_local_file(repo_root / "dev" / "hosting" / "beam" / "export_bf16.py", "/root/dev/hosting/beam/export_bf16.py")
)

checkpoints = modal.Volume.from_name("bartholomew-checkpoints", create_if_missing=True)

app = modal.App(APP_NAME)


def _sft_command(config, run_dir, base_dir, run):
    """The chat_sft invocation the harness builds (experiment.py _train_downstream), minus HF/W&B."""
    import sys

    data, training = config.get("data", {}), config["training"]
    cmd = [
        sys.executable, "-u", "-m", "scripts.chat_sft",
        f"--base-checkpoint-dir={base_dir / 'base_checkpoints'}",
        f"--base-step={BASE_STEP}",
        f"--recipe={data.get('recipe', 'nanochat-default')}",
        f"--num-iterations={training.get('num_iterations', -1)}",
        f"--num-epochs={training.get('num_epochs', 1)}",
        f"--device-batch-size={training.get('device_batch_size', 8)}",
        f"--eval-every={training.get('eval_every', -1)}",
        f"--chatcore-every={training.get('chatcore_every', -1)}",
        f"--save-every={training.get('save_every', 200)}",
        f"--checkpoint-dir={run_dir / 'checkpoints'}",
        f"--tokenizer-dir={base_dir / 'tokenizer'}",
        f"--experiment-id={run}",
        f"--experiment-config={run_dir / 'config.json'}",
        "--run=dummy",
    ]
    optional = {
        "warmup_ratio": "--warmup-ratio", "warmdown_ratio": "--warmdown-ratio",
        "init_lr_frac": "--init-lr-frac", "final_lr_frac": "--final-lr-frac",
        "matrix_lr": "--matrix-lr", "embedding_lr": "--embedding-lr",
        "unembedding_lr": "--unembedding-lr", "max_seq_len": "--max-seq-len",
        "total_batch_size": "--total-batch-size", "eval_tokens": "--eval-tokens",
        "chatcore_max_cat": "--chatcore-max-cat", "chatcore_max_sample": "--chatcore-max-sample",
        "load_optimizer": "--load-optimizer", "target_examples_per_step": "--target-examples-per-step",
    }
    cmd += [f"{flag}={training[key]}" for key, flag in optional.items() if training.get(key) is not None]
    return cmd


@app.function(image=image, gpu=TRAIN_GPU, cpu=8, memory=65536, timeout=23 * 60 * 60,
              volumes={"/vol": checkpoints})
def train(config_text: str, run: str):
    import shutil
    import subprocess
    from huggingface_hub import snapshot_download

    config = json.loads(config_text)
    run_dir = Path("/vol/runs") / run
    if (run_dir / "checkpoints").exists() and any((run_dir / "checkpoints").glob("model_*.pt")):
        raise RuntimeError(f"{run_dir} already holds checkpoints; choose a new --run")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Base weights (no optimizer: these configs set load_optimizer=0) and tokenizer,
    # cached in the Volume so a second run skips the 8.4 GiB download.
    prefix = f"experiments/{BASE_EXPERIMENT_ID}"
    base_dir = Path("/vol/base") / prefix
    if not (base_dir / "base_checkpoints" / f"model_{BASE_STEP:06d}.pt").exists():
        snapshot_download(MODEL_REPO, local_dir="/vol/base", allow_patterns=[
            f"{prefix}/base_checkpoints/model_{BASE_STEP:06d}.pt",
            f"{prefix}/base_checkpoints/meta_{BASE_STEP:06d}.json",
            f"{prefix}/tokenizer/*",
        ])
        checkpoints.commit()

    config["parent"] = {"base_experiment_id": BASE_EXPERIMENT_ID, "checkpoint_step": BASE_STEP}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    cmd = _sft_command(config, run_dir, base_dir, run)
    # As the harness's environment(): chat_sft reads token_bytes.pt via NANOCHAT_TOKENIZER_DIR,
    # not --tokenizer-dir.
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
           "NANOCHAT_TOKENIZER_DIR": str(base_dir / "tokenizer"),
           "NANOCHAT_EXPERIMENT_STAGE": "sft", "NANOCHAT_EXPERIMENT_ID": run}
    print(" ".join(cmd), flush=True)
    with open(run_dir / "train.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd="/root", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if process.wait() != 0:
            checkpoints.commit()  # keep the log of a failed run
            raise RuntimeError(f"chat_sft exited with {process.returncode}; see {run_dir / 'train.log'}")
    checkpoints.commit()  # the trained weights survive even if the export below fails

    subprocess.run([
        "python", "dev/hosting/beam/export_bf16.py",
        f"--in-dir={run_dir / 'checkpoints'}", f"--out-dir={run_dir / 'bf16'}",
        f"--tokenizer-dir={base_dir / 'tokenizer'}",
    ], cwd="/root", check=True)
    shutil.copy(run_dir / "config.json", run_dir / "bf16" / "config.json")
    checkpoints.commit()
    print(f"\nDone. Test it with:\n  modal run modal_eval.py --checkpoint-dir {run_dir / 'bf16'} --tag {run}")


@app.local_entrypoint()
def main(config: str = DEFAULT_CONFIG, run: str = "", iterations: int = 0):
    """`--iterations N` is a smoke test: N optimizer steps, no ChatCORE, its own run name.

    It checks the pipeline, not the model: with --num-iterations set, chat_sft
    measures progress in micro-batches (`it / num_iterations`), so under gradient
    accumulation the LR schedule overshoots into negative multipliers. Full runs
    (num_iterations=-1) measure progress by data consumed and are unaffected.
    """
    config_path = repo_root / config
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    run = run or spec["experiment_suffix"]
    if iterations > 0:
        spec["training"].update(num_iterations=iterations, chatcore_every=-1, eval_every=-1)
        run = f"{run}-smoke{iterations}"
    # spawn, not remote: a run outlives this client. With .remote() the job is tied
    # to the local process, and when its auth token expired ~1h45m into a run Modal
    # cancelled the input and the unsaved run was lost. Needs --detach, or the App
    # stops as soon as this entrypoint returns.
    call = train.spawn(json.dumps(spec), run)
    print(f"Spawned {run} as {call.object_id} in App {app.app_id}.\n"
          f"  logs:   modal app logs {app.app_id}\n"
          f"  output: modal volume ls bartholomew-checkpoints runs/{run}")
