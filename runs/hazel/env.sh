# Shared environment for every Hazel script. Source it; do not run it.
#
#   source runs/hazel/env.sh
#
# Where things live (see runs/hazel/README.md for why):
#   repo        wherever you cloned it, normally $HOME/bartholomew-v2
#   venv, uv    /usr/local/usrapps/$GROUP/$USER   backed up, never purged, read-only on compute nodes
#   data, runs  /share/$GROUP/$USER               20 TB scratch, NOT backed up, purged after 30 days unused
#
# $GROUP is your HPC project group (Ditto's NAIL project once you have been added).
# It is detected with `id -gn`; set HAZEL_GROUP before sourcing to override it.

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _hazel_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    _hazel_here="$(cd "$(dirname "$0")" && pwd)"
fi
export BART_REPO="${BART_REPO:-$(cd "$_hazel_here/../.." && pwd)}"
unset _hazel_here

export HAZEL_GROUP="${HAZEL_GROUP:-$(id -gn)}"
export HAZEL_SHARE="${HAZEL_SHARE:-/share/$HAZEL_GROUP/$USER}"

# Software goes in usrapps when the project has it; otherwise on scratch, which the
# 30-day purge can remove (rerun setup.sh if the venv disappears).
if [ -z "${HAZEL_APPS:-}" ]; then
    if [ -d "/usr/local/usrapps/$HAZEL_GROUP" ]; then
        HAZEL_APPS="/usr/local/usrapps/$HAZEL_GROUP/$USER"
    else
        HAZEL_APPS="$HAZEL_SHARE/apps"
    fi
fi
export HAZEL_APPS

# uv: the tool, its Python builds and the project venv all live under HAZEL_APPS.
export PATH="$HAZEL_APPS/bin:$PATH"
export UV_INSTALL_DIR="$HAZEL_APPS/bin"
export UV_PYTHON_INSTALL_DIR="$HAZEL_APPS/python"
export UV_PROJECT_ENVIRONMENT="$HAZEL_APPS/venv-bartholomew"
export UV_CACHE_DIR="$HAZEL_SHARE/cache/uv"
# The cache and the venv are on different filesystems, so hardlinks are impossible.
export UV_LINK_MODE=copy
# The Modal runs used Python 3.12 (Modal no longer offers 3.10); match them.
export UV_PYTHON=3.12
export UV_PYTHON_PREFERENCE=only-managed

# Everything a run writes goes on scratch. /home is only 15 GB.
export NANOCHAT_BASE_DIR="$HAZEL_SHARE/nanochat"
export HF_HOME="$HAZEL_SHARE/cache/huggingface"
export WANDB_DIR="$HAZEL_SHARE/wandb"
export WANDB_CACHE_DIR="$HAZEL_SHARE/cache/wandb"
export TRITON_CACHE_DIR="$HAZEL_SHARE/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$HAZEL_SHARE/cache/torchinductor"
export PIP_CACHE_DIR="$HAZEL_SHARE/cache/pip"
export XDG_CACHE_HOME="$HAZEL_SHARE/cache/xdg"
export PYTHONUNBUFFERED=1

# Credentials, written by you (setup.sh explains how). Never commit this file.
export BART_SECRETS="${BART_SECRETS:-$HOME/.config/bartholomew/secrets.env}"
if [ -f "$BART_SECRETS" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$BART_SECRETS"
    set +a
fi

# Same hosted token cache the Modal launcher restores (private; built from the dataset if unreadable).
export HOSTED_PRETOK_REPO="${HOSTED_PRETOK_REPO:-jbduran/clean1930s-d24-r12-ctx4096-fulltok-v1-pretok}"

bart_python() {
    "$UV_PROJECT_ENVIRONMENT/bin/python" "$@"
}
