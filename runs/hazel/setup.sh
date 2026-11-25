#!/usr/bin/env bash
# One-time (and after every dependency change) setup on a Hazel LOGIN node.
#
#   cd ~/bartholomew-v2 && bash runs/hazel/setup.sh
#
# Installs uv and the exact uv.lock environment (CUDA 12.8 torch), pre-downloads the
# FA3 kernel the trainer loads from the Hub, creates the scratch directories and
# checks that the configs parse. Installing packages is allowed on login nodes;
# training and tokenizing are not, so those go through sbatch.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BART_REPO"

echo "Project group: $HAZEL_GROUP"
echo "Software:      $HAZEL_APPS"
echo "Scratch:       $HAZEL_SHARE"
echo

if [ ! -d "/share/$HAZEL_GROUP" ]; then
    echo "No /share/$HAZEL_GROUP. Either you are not in an HPC project yet (ask the PI to add" >&2
    echo "your Unity ID to the NAIL project), or your login group is not the project group:" >&2
    echo "run 'groups' and rerun with HAZEL_GROUP=<project group> bash runs/hazel/setup.sh" >&2
    exit 1
fi
case "$HAZEL_APPS" in
    "$HAZEL_SHARE"/*)
        echo "NOTE: /usr/local/usrapps/$HAZEL_GROUP does not exist, so the venv goes on scratch,"
        echo "      where the 30-day purge can delete it. Ask the PI to request usrapps space for"
        echo "      the project (https://hpc.ncsu.edu/Software/Software.php#user)."
        echo
        ;;
esac

mkdir -p "$HAZEL_APPS/bin" "$HAZEL_SHARE"/{nanochat,wandb,logs,cache} "$(dirname "$BART_SECRETS")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv into $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv --version

echo "Syncing the uv.lock environment into $UV_PROJECT_ENVIRONMENT"
uv python install "$UV_PYTHON"
uv sync --frozen --extra gpu

bart_python - <<'EOF'
import sys, torch
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, CUDA build {torch.version.cuda}")
EOF

# The trainer fetches FA3 with kernels.get_kernel at import time. Cache it now so a
# compute node without internet access still finds it.
echo "Pre-downloading the FA3 kernel into $HF_HOME"
bart_python - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("varunneal/flash-attention-3"))
EOF

echo "Checking that the Engram and smoke configs parse"
bart_python - <<'EOF'
import glob
from scripts.experiment import Experiment
paths = sorted(glob.glob("configs/base/clean1930s-d12-r12-ctx4096-engram-*.json"))
paths += sorted(glob.glob("configs/base/clean1930s-d12-r12-ctx4096-modal-smoke*.json"))
for path in paths:
    e = Experiment(path)
    e.validate_config()
    print("config ok:", e.experiment_id)
EOF

if [ -f "$BART_SECRETS" ]; then
    chmod 600 "$BART_SECRETS"
    for key in HF_TOKEN WANDB_API_KEY; do
        if [ -z "${!key:-}" ]; then
            echo "WARNING: $key is not set in $BART_SECRETS" >&2
        fi
    done
else
    cat <<EOF

No credentials file yet. Create it yourself (it is read by every job):

    umask 077
    cat > $BART_SECRETS <<'KEYS'
HF_TOKEN=hf_...        # write access to the config's artifacts.repo
WANDB_API_KEY=...      # access to the config's wandb.entity
KEYS
EOF
fi

cat <<EOF

Setup done. Next, check a GPU node (about 10 minutes on one H100):

    runs/hazel/submit.sh check
EOF
