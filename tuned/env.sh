#!/bin/bash
# tuned/env.sh <variant> — device config + build-env setup for a tuned
# single-arch flash-attention (vllm_flash_attn) build (gb10/rtx40/rtx50).
# Source this file with the variant as $1; do not execute it directly.
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/FA2_ARCH/HW_LABEL (from
# tuned/devices/<variant>.conf), CUDA_HOME (autodetected highest installed
# toolkit), CUDA_VERSION_COMPACT, CUDA_ARCHS, FA2_TUNED_ARCH.
#
# CUDA_ARCHS/FA2_TUNED_ARCH are plain shell env vars, forwarded into cmake
# by setup.py's own cmake_build_ext (see setup.py's
# `if os.environ.get('CUDA_ARCHS')` block, added alongside this tooling --
# confirmed nothing forwarded -DCUDA_ARCHS before that).

GPU_TUNED_ARG_VARIANT="$1"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10/rtx40/rtx50)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=devices/rtx50.conf
source "${GPU_TUNED_SELF_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_FA2_ARCH GPU_TUNED_HW_LABEL

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (pinned commit --
# see common.sh's own header/sync instructions to update). Provides
# gpu_tuned_verify_arch/assert_platform/installed_cuda_toolkits, shared
# verbatim across the fleet instead of hand-copied-and-edited per repo.
# NOT used for embed_build_info here: kept local on purpose (same
# reasoning as zbrad/raft's/zbrad/cuvs's/zbrad/faiss's/zbrad/pytorch's
# tuned/env.sh).
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

# Fail loudly if this script runs on the wrong host, rather than letting a
# mismatched build silently produce a wrong-architecture wheel that only
# surfaces as a confusing failure several steps later.
gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- Resolve CUDA_HOME to the highest installed toolkit when not explicitly set ---
if [ -z "${CUDA_HOME:-}" ]; then
    _fa_highest="$(gpu_tuned_installed_cuda_toolkits | tail -1)"
    if [ -n "$_fa_highest" ]; then
        export CUDA_HOME="/usr/local/cuda-${_fa_highest}"
    else
        echo "[tuned/env] WARNING: no /usr/local/cuda-<ver> toolkit found; leaving CUDA_HOME unset." >&2
        echo "[tuned/env]          Set CUDA_HOME explicitly to an installed toolkit." >&2
    fi
    unset _fa_highest
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"

if [ -n "${CUDA_HOME:-}" ]; then
    CUDA_VERSION_COMPACT="$(basename "$CUDA_HOME" | sed -E 's/^cuda-([0-9]+)\.([0-9]+).*/\1\2/')"
    export CUDA_VERSION_COMPACT
fi

# --- Output layout: one tree per toolkit, tuned/build/cu<ver>/ (gitignored via build/):
#     venv/ (build and test venv), dist/ (wheels), wheels/ (the pinned torch wheel), temp/ (the
#     CMake/ninja tree), ccache/, and timing.tsv. env.sh refuses a CMake tree configured for
#     another toolkit or venv, and the legacy in-repo build/temp.* tree (below). ---
if [ -n "${CUDA_VERSION_COMPACT:-}" ]; then
    GPU_TUNED_BUILD_ROOT="${GPU_TUNED_SELF_DIR}/build/cu${CUDA_VERSION_COMPACT}"
    GPU_TUNED_VENV_DIR="${GPU_TUNED_BUILD_ROOT}/venv"
    GPU_TUNED_DIST_DIR="${GPU_TUNED_BUILD_ROOT}/dist"
    GPU_TUNED_WHEELS_DIR="${GPU_TUNED_BUILD_ROOT}/wheels"
    # The CMake/ninja tree. setup.py would use build/temp.<platform>-<python>/ in the repo; both
    # build.sh and wheel.sh pass --build-temp to put it here instead (no tracked-file edit), at a
    # fixed path per toolkit, which is what lets ccache hit and keeps ninja's log for timings.
    # setup.py would put the CMake tree in build/temp.<platform>-<python>/ inside the repo; build.sh
    # and wheel.sh pass --build-temp to put it here instead (no tracked-file edit). A fixed path per
    # toolkit is what lets ccache hit and keeps ninja's log for the timings.
    GPU_TUNED_BUILD_TEMP="${GPU_TUNED_BUILD_ROOT}/temp"
    export GPU_TUNED_BUILD_ROOT GPU_TUNED_VENV_DIR GPU_TUNED_DIST_DIR GPU_TUNED_WHEELS_DIR GPU_TUNED_BUILD_TEMP
    # ccache per toolkit. setup.py (upstream's) already passes ccache to CMake as the C++ and CUDA
    # compiler launcher whenever it is on PATH (sccache first, if present); CCACHE_DIR makes
    # ccache ignore ~/.config/ccache/ccache.conf, so the size cap is set here.
    export CCACHE_DIR="${GPU_TUNED_BUILD_ROOT}/ccache"
    export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
fi

# Which upstream ref the build-info stamp's base is computed against (the merge-base with it is
# what tuned-builds really contains). Override per build with GPU_TUNED_UPSTREAM_REF.
: "${GPU_TUNED_UPSTREAM_REF:=upstream/main}"
export GPU_TUNED_UPSTREAM_REF

# The torch this build is pinned to (tuned/torch.pins): URL, file name and version, for the venv
# install, the version assertion in build.sh, the smoke test and the stamp's "deps" field.
GPU_TUNED_TORCH_URL=""; GPU_TUNED_TORCH_VERSION=""
if [ -n "${CUDA_VERSION_COMPACT:-}" ] && [ -f "${GPU_TUNED_SELF_DIR}/torch.pins" ]; then
    GPU_TUNED_TORCH_URL="$(awk -v v="${GPU_TUNED_VARIANT}" -v c="cu${CUDA_VERSION_COMPACT}" '$1 !~ /^#/ && $1==v && $2==c {print $3}' "${GPU_TUNED_SELF_DIR}/torch.pins")"
    if [ -n "${GPU_TUNED_TORCH_URL}" ]; then
        GPU_TUNED_TORCH_VERSION="$(basename "${GPU_TUNED_TORCH_URL}" | sed -E 's/^torch-([^-]+)-.*/\1/; s/%2B/+/')"
        export GPU_TUNED_BUILD_INFO_DEPS="torch ${GPU_TUNED_TORCH_VERSION} ${GPU_TUNED_TORCH_URL%/download/*}/tag/$(basename "$(dirname "${GPU_TUNED_TORCH_URL}")")"
    fi
fi
export GPU_TUNED_TORCH_URL GPU_TUNED_TORCH_VERSION

# gpu_tuned_assert_cmake_cache_consistent: fail if the existing CMake tree under build/temp.* was
# configured for another toolkit or another Python (it would silently keep the old torch/CUDA).
gpu_tuned_assert_cmake_cache_consistent() {
    local cache cached legacy
    # The old layout's tree (setup.py's default location) would be compiled into by nothing and
    # timed by nothing: refuse it rather than let two trees drift.
    for legacy in "${GPU_TUNED_SELF_DIR}"/../build/temp.*/CMakeCache.txt; do
        [ -f "${legacy}" ] || continue
        echo "ERROR: ${legacy} is the old in-repo CMake tree; builds now use ${GPU_TUNED_BUILD_TEMP}. Remove it: rm -rf $(dirname "${legacy}")" >&2
        return 1
    done
    for cache in "${GPU_TUNED_BUILD_TEMP}/CMakeCache.txt"; do
        [ -f "${cache}" ] || continue
        cached="$(sed -n 's/^CMAKE_CUDA_COMPILER:FILEPATH=//p' "${cache}" | head -1)"
        if [ -n "${cached}" ] && [ "${cached}" != "${CUDA_HOME}/bin/nvcc" ]; then
            echo "ERROR: ${cache} was configured with ${cached}, but CUDA_HOME=${CUDA_HOME}. Remove that tree first: rm -rf $(dirname "${cache}")" >&2
            return 1
        fi
        cached="$(sed -n 's/^Python_EXECUTABLE:[A-Z]*=//p' "${cache}" | head -1)"
        if [ -n "${cached}" ] && [ -n "${GPU_TUNED_VENV_DIR:-}" ] && [[ "${cached}" != "${GPU_TUNED_VENV_DIR}"/* ]]; then
            echo "ERROR: ${cache} was configured with Python ${cached}, not this build's venv ${GPU_TUNED_VENV_DIR}. Remove that tree first: rm -rf $(dirname "${cache}")" >&2
            return 1
        fi
    done
}

export CUDA_ARCHS="${GPU_TUNED_FA2_ARCH}"
export FA2_TUNED_ARCH="${GPU_TUNED_FA2_ARCH}"

echo "[tuned/env] GPU_TUNED_VARIANT=${GPU_TUNED_VARIANT} FA2_TUNED_ARCH=${FA2_TUNED_ARCH} CUDA_HOME=${CUDA_HOME:-<unset>}"

# gpu_tuned_verify_arch now comes from common.sh (sourced above); call
# sites pass GPU_TUNED_FA2_ARCH explicitly (the shared version takes it
# as an arg instead of reading a global, since different repos in the
# fleet name their arch var differently).

# embed_build_info <so_path> <variant> <package> <version> [hw_label] —
# thin wrapper over gpu_tuned_embed_build_info (common.sh) that pins the
# section name to .flash_attn_build_info (this repo's pre-existing name,
# via the explicit [section-name] override -- otherwise the shared
# function would derive .vllm_flash_attn_build_info from the
# "vllm_flash_attn" package arg this repo's call sites pass, a rename),
# while keeping the real package name in the message.
embed_build_info() {
    local so_path="$1" variant="$2" package="$3" version="$4" hw_label="$5"
    gpu_tuned_embed_build_info "${so_path}" "${variant}" "${package}" "${version}" \
        "${hw_label}" "https://github.com/zbrad/flash-attention-vllm" "flash_attn_build_info"
}
