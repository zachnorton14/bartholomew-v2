"""Run the system prompt eval (scripts/system_prompt_eval.py) on a Modal GPU.

    modal run modal_eval.py
    modal run modal_eval.py --variants none,pre1930-companion --samples 8
    modal run modal_eval.py --checkpoint-dir /vol/<run>/checkpoints --tag retrained

This is a separate App from modal_app.py, so running it never touches the
deployment. It uses the same GPU classes and the same bf16 export, so a prompt
is measured on the hardware and weights visitors get. The table streams to the
terminal; samples.jsonl, summary.json and transcripts.md are copied back to
dev-ignore/system_prompt_eval/<timestamp>-<tag>/.

Retrained checkpoints go in the `bartholomew-checkpoints` Volume, mounted at
/vol; point --checkpoint-dir at one to compare it with the deployed model.
From Git Bash on Windows, prefix MSYS_NO_PATHCONV=1, or the shell rewrites
/vol/... into a Windows path before Modal sees it.
"""

import time
from pathlib import Path

import modal


APP_NAME = "bartholomew-system-prompt-eval"
# Same weights and GPU fallbacks as modal_app.py: never "any", which can pick a
# T4 without bf16.
GPU_FALLBACKS = ["A10", "L4", "L40S"]
MODEL_REPO = "jbduran/bart-experiments"
BASE_EXPERIMENT_ID = "Think.Unbounded-d32-v2mix-cont"
MODEL_TAG = "Think.Unbounded-d32-v2mix-cont-pre1930-curriculum-c3-robust-v2"
HF_PREFIX = f"experiments/{BASE_EXPERIMENT_ID}/sft/{MODEL_TAG}/bf16"

repo_root = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.9.1",
        "tiktoken>=0.11.0",
        "tokenizers>=0.22.0",
        "rustbpe>=0.1.0",
        "filelock",
        "numpy",
        "huggingface_hub[cli]",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .run_commands(
        f"hf download {MODEL_REPO} --repo-type model "
        f"--include '{HF_PREFIX}/*' --local-dir /tmp/bart-weights",
        f"mkdir -p /img-model && mv /tmp/bart-weights/{HF_PREFIX}/* /img-model/ "
        "&& rm -rf /tmp/bart-weights && ls -lah /img-model",
    )
    # Source after the 5+ GiB weights layer, so code and prompt edits never rebuild it.
    .add_local_dir(repo_root / "nanochat", "/root/nanochat")
    .add_local_dir(repo_root / "configs" / "system_prompts", "/root/configs/system_prompts")
    .add_local_file(repo_root / "scripts" / "system_prompt_eval.py", "/root/scripts/system_prompt_eval.py")
    .add_local_file(repo_root / "dev" / "hosting" / "beam" / "conversation.py", "/root/dev/hosting/beam/conversation.py")
    .add_local_file(repo_root / "dev" / "hosting" / "beam" / "fast_load.py", "/root/dev/hosting/beam/fast_load.py")
)

checkpoints = modal.Volume.from_name("bartholomew-checkpoints", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(image=image, gpu=GPU_FALLBACKS, cpu=4, memory=32768, timeout=2 * 60 * 60,
              volumes={"/vol": checkpoints})
def evaluate(argv: list[str]) -> dict[str, str]:
    """Run the eval in the container and return its output files as text."""
    import subprocess
    import sys

    out = Path("/tmp/system_prompt_eval")
    subprocess.run([sys.executable, "-u", "-m", "scripts.system_prompt_eval", "--out", str(out), *argv],
                   cwd="/root", check=True)
    return {p.name: p.read_text(encoding="utf-8") for p in out.iterdir() if p.is_file()}


@app.local_entrypoint()
def main(variants: str = "all", probes: str = "", samples: int = 4, max_tokens: int = 256,
         checkpoint_dir: str = "/img-model", tag: str = "modal", priming: str = "default"):
    """`--priming none` drops the hidden opening exchange the deployment prepends."""
    argv = ["--checkpoint-dir", checkpoint_dir, "--variants", variants,
            "--samples", str(samples), "--max-tokens", str(max_tokens),
            f"--priming-turns={'' if priming == 'none' else priming}"]
    if probes:
        argv += ["--probes", probes]
    files = evaluate.remote(argv)
    out = repo_root / "dev-ignore" / "system_prompt_eval" / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    print(f"\nResults copied to {out}")
