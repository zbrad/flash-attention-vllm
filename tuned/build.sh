#!/bin/bash
# tuned/build.sh <variant> — build the vllm_flash_attn wheel for one GPU variant (gb10/rtx40/rtx50) and one CUDA
# toolkit, single-arch. It prepares the build venv in the per-toolkit tree tuned/build/<cuda-tag>/ (see env.sh),
# installs the pinned torch and the build tools, then makes the unstamped wheel with tuned/wheel.sh. The compile
# is the long step.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10     (or CUDA_VER=13.4)
#
# Stage order (docs/ARCHITECTURE.md B3): build.sh -> full_test.sh -> package.sh -> smoke_test.sh ->
# (review and commit the release notes) -> release.sh -> publish.sh.
#
# flash-attention builds AGAINST an existing torch, so the torch is part of the build's identity:
# tuned/torch.pins names the zbrad/pytorch release for this variant and toolkit, this script
# installs exactly that wheel into the build venv, and it refuses to compile if the installed
# torch is anything else. A variant with no pin falls back to a PyPI CUDA torch (rtx40/rtx50).
#
# Compiler caching comes from upstream's setup.py, which hands ccache to CMake as the C++ and
# CUDA compiler launcher when it is on PATH; env.sh points CCACHE_DIR at the per-toolkit cache.
# wheel.sh reads per-target timings back from ninja's own log into tuned/build/<cuda-tag>/timing.tsv.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/build.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found on PATH." >&2; exit 1; }

VENV_DIR="${GPU_TUNED_VENV_DIR}"
if [[ "${GPU_TUNED_CLEAN_VENV:-}" == "1" && -d "${VENV_DIR}" ]]; then
    echo "GPU_TUNED_CLEAN_VENV=1: removing ${VENV_DIR} for a clean rebuild"
    rm -rf "${VENV_DIR}"
fi
if [[ -d "${VENV_DIR}" ]]; then
    # Reusing an existing venv by default -- verify it wasn't copied/contaminated from another
    # repo before building on top of it. Set GPU_TUNED_CLEAN_VENV=1 to force a fresh venv instead.
    gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
else
    mkdir -p "${GPU_TUNED_BUILD_ROOT}"
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# A CMake tree configured for another toolkit or another venv's Python would silently keep them.
gpu_tuned_assert_cmake_cache_consistent

echo "=========================================="
echo "Building vllm_flash_attn for ${GPU_TUNED_HW_LABEL} only"
echo "=========================================="
echo "FA2_TUNED_ARCH: ${FA2_TUNED_ARCH}"
echo "Toolkit:        ${CUDA_TAG} (${CUDA_HOME})"
echo "Build tree:     ${GPU_TUNED_BUILD_ROOT}"
echo "ccache:         ${CCACHE_DIR} (max ${CCACHE_MAXSIZE})"
echo "Python:         $(python3 --version)"
echo "Git commit:     $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

if [[ -n "${GPU_TUNED_TORCH_URL}" ]]; then
    # Pinned torch: fetch the exact wheel once, install it, then assert the version.
    TORCH_WHEEL="${GPU_TUNED_WHEELS_DIR}/$(basename "${GPU_TUNED_TORCH_URL//%2B/+}")"
    if [[ ! -f "${TORCH_WHEEL}" ]]; then
        mkdir -p "${GPU_TUNED_WHEELS_DIR}"
        echo "Downloading the pinned torch ${GPU_TUNED_TORCH_VERSION} ..."
        curl -fL --retry 3 -o "${TORCH_WHEEL}.part" "${GPU_TUNED_TORCH_URL}"
        mv "${TORCH_WHEEL}.part" "${TORCH_WHEEL}"
    fi
    if [[ "$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || true)" != "${GPU_TUNED_TORCH_VERSION}" ]]; then
        echo "Installing torch ${GPU_TUNED_TORCH_VERSION} into the build venv ..."
        pip install --upgrade pip
        pip install --force-reinstall "${TORCH_WHEEL}"
    fi
    INSTALLED_TORCH="$(python3 -c 'import torch; print(torch.__version__)')"
    [[ "${INSTALLED_TORCH}" == "${GPU_TUNED_TORCH_VERSION}" ]] || {
        echo "ERROR: the build venv has torch ${INSTALLED_TORCH}, but tuned/torch.pins requires ${GPU_TUNED_TORCH_VERSION}." >&2
        exit 1
    }
    # The extensions are compiled with this toolkit, so torch must be built for the same CUDA.
    TORCH_CUDA="$(python3 -c 'import torch; print(torch.version.cuda)')"
    [[ "${TORCH_CUDA}" == "${CUDA_VER}" ]] || {
        echo "ERROR: torch reports CUDA ${TORCH_CUDA} but this build uses ${CUDA_TAG} (CUDA ${CUDA_VER})." >&2
        exit 1
    }
elif ! python3 -c "import torch" &>/dev/null; then
    if [[ "${GPU_TUNED_VARIANT}" == "gb10" ]]; then
        echo "ERROR: gb10 has no torch pin for ${CUDA_TAG} in tuned/torch.pins, no torch is installed in" >&2
        echo "       ${VENV_DIR}, and there is no official upstream aarch64/sm_121 torch wheel." >&2
        exit 1
    fi
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
python3 -c "import torch; print(f'Using torch {torch.__version__} (CUDA {torch.version.cuda})')"

echo "Installing other build-time requirements (build, cmake, ninja, packaging, setuptools, wheel, jinja2)..."
pip install build "cmake>=3.26" ninja packaging "setuptools>=49.4.0" wheel jinja2

echo ""
bash "${REPO_ROOT}/tuned/wheel.sh" "${GPU_TUNED_VARIANT}"

echo ""
echo "=========================================="
echo "Build complete (${GPU_TUNED_HW_LABEL})."
echo "=========================================="
echo "Venv:         ${VENV_DIR}"
echo "Label:        $(cat "${GPU_TUNED_BUILD_ROOT}/release-label")"
echo "Build commit: $(cat "${GPU_TUNED_BUILD_ROOT}/build-commit")"
echo "Next: CUDA_TAG=${CUDA_TAG} bash tuned/full_test.sh ${GPU_TUNED_VARIANT}"
