#!/bin/bash
# tuned/build.sh <variant> — build/install vllm_flash_attn from source for a
# single GPU variant (gb10/rtx40/rtx50) only, single-arch, into a
# per-variant editable venv.
#
# Requires a matching torch already be installed in the venv before this
# runs -- flash-attention builds AGAINST an existing torch, it doesn't
# manage installing one for you. For gb10 specifically, that means zbrad/
# pytorch's own tuned-builds gb10 wheel (no official aarch64/sm_121 torch
# exists upstream); for rtx40/rtx50, a normal PyPI CUDA torch works fine.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found on PATH." >&2; exit 1; }

VENV_DIR="${REPO_ROOT}/.venv-${GPU_TUNED_VARIANT}"
[[ -d "${VENV_DIR}" ]] || python3 -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "=========================================="
echo "Building vllm_flash_attn for ${GPU_TUNED_HW_LABEL} only"
echo "=========================================="
echo "FA2_TUNED_ARCH: ${FA2_TUNED_ARCH}"
echo "CUDA_HOME:      ${CUDA_HOME:-<unset>}"
echo "Python:         $(python3 --version)"
echo "Git commit:     $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

if ! python3 -c "import torch" &>/dev/null; then
    if [[ "${GPU_TUNED_VARIANT}" == "gb10" ]]; then
        echo "ERROR: no torch installed in ${VENV_DIR}, and gb10 has no official" >&2
        echo "       upstream aarch64/sm_121 torch wheel. Install zbrad/pytorch's" >&2
        echo "       own tuned-builds gb10 wheel into this venv first:" >&2
        echo "       ${VENV_DIR}/bin/pip install <path-or-URL-to-zbrad-pytorch-gb10-wheel>" >&2
        exit 1
    else
        echo "No torch found -- installing the version this repo's setup.py declares..."
        pip install --upgrade pip
        pip install "torch==$(python3 -c "
import ast
with open('setup.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(t.id == 'PYTORCH_VERSION' for t in node.targets if isinstance(t, ast.Name)):
        print(ast.literal_eval(node.value))
")"
    fi
fi
python3 -c "import torch; print(f'Using torch {torch.__version__} (CUDA {torch.version.cuda})')"

echo "Installing other build-time requirements (cmake, ninja, packaging, setuptools, wheel, jinja2)..."
pip install "cmake>=3.26" ninja packaging "setuptools>=49.4.0" wheel jinja2

echo "Building vllm_flash_attn (this will take a long time)..."
pip install --no-build-isolation -v -e .

echo ""
echo "Smoke test: import vllm_flash_attn, confirm it loads..."
python3 - <<'PYEOF'
import torch
import vllm_flash_attn

print(f"vllm_flash_attn imported OK from {vllm_flash_attn.__file__}")
assert torch.cuda.is_available(), "CUDA device not available -- cannot exercise a real kernel call here"
PYEOF

echo ""
echo "=========================================="
echo "Build complete (${GPU_TUNED_HW_LABEL})."
echo "=========================================="
echo "Venv: ${VENV_DIR}"
echo "Next: bash tuned/wheel.sh ${GPU_TUNED_VARIANT}"
