#!/usr/bin/env bash
# Step 0: create the Python environment.
#
#   bash scripts/00_create_env.sh            # venv at ./.venv (or $RLPROF_VENV)
#   source .venv/bin/activate
#
# Needs: an NVIDIA GPU with a driver for CUDA 13 (driver >= 580), and `uv`
# (https://docs.astral.sh/uv/; installed into ~/.local/bin if missing).
# Installs the exact versions in requirements.lock.txt (vLLM 0.34.0, torch 2.14.0 cu130,
# cupti-python 13.0.1, matching the CUPTI 13.0 that torch pins). Use REQS=requirements.txt to resolve fresh versions instead.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${RLPROF_VENV:-.venv}"
REQS="${REQS:-requirements.lock.txt}"
PY="${PYTHON_VERSION:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing it to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "warning: nvidia-smi not found; a CUDA GPU is required to run anything past step 3" >&2
fi

uv venv --clear --python "$PY" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -r "$REQS"
uv pip install -e . --no-deps

python - <<'EOF'
import torch, vllm, transformers
from rlprof.cupti_profiler import cupti_available
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), vllm {vllm.__version__}, "
      f"transformers {transformers.__version__}; {cupti_available()[1]}")
EOF
echo
echo "Environment ready. Next:  source $VENV/bin/activate && python scripts/01_check_env.py"
