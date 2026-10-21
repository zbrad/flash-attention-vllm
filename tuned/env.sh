#!/bin/bash
# tuned/env.sh <variant> — device, toolkit and build-environment setup for a tuned
# single-arch flash-attention (vllm_flash_attn) build (gb10/rtx40/rtx50).
# Source this file with the variant as $1; do not execute it directly.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10     (or CUDA_VER=13.4)
#
# Device facts come from tuned/common-data/devices/<variant>.conf and the CUDA toolkit from
# tuned/common-data/toolkits/<id>.env (both vendored from zbrad/tuned-common by sync-common.sh),
# never from /usr/local/cuda or from what happens to be installed. There is no default: CUDA_TAG
# or CUDA_VER must name one. The environment is derived on every run and never saved; what a build
# used is recorded in its build info (package.sh).
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/CUDA_ARCH/HW_LABEL (shared), GPU_TUNED_FA2_ARCH
# (repo.env), CUDA_HOME/CUDA_TAG/CUDA_VER (the toolkit), CUDA_VERSION_COMPACT, CUDA_ARCHS,
# FA2_TUNED_ARCH, GPU_TUNED_BUILD_ROOT (tuned/build/<cuda-tag>), and GPU_TUNED_VENV_DIR /
# GPU_TUNED_WHEELS_DIR / GPU_TUNED_WHEEL_DIR / GPU_TUNED_DIST_DIR / GPU_TUNED_RELEASES_DIR /
# GPU_TUNED_BUILD_TEMP under it.
#
# CUDA_ARCHS/FA2_TUNED_ARCH are plain shell env vars, forwarded into cmake
# by setup.py's own cmake_build_ext (see setup.py's
# `if os.environ.get('CUDA_ARCHS')` block, added alongside this tooling --
# confirmed nothing forwarded -DCUDA_ARCHS before that).

GPU_TUNED_ARG_VARIANT="${1:-}"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10/rtx40/rtx50)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_TUNED_DATA_DIR="${GPU_TUNED_SELF_DIR}/common-data"
if [[ ! -f "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" ]]; then
    echo "ERROR: no device '${GPU_TUNED_ARG_VARIANT}' in ${GPU_TUNED_DATA_DIR}/devices (have: $(cd "${GPU_TUNED_DATA_DIR}/devices" && ls | sed 's/\.conf$//' | tr '\n' ' '))" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck source=common-data/devices/gb10.conf
source "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_CUDA_ARCH GPU_TUNED_HW_LABEL

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (pinned commit --
# see common.sh's own header/sync instructions to update). Provides
# gpu_tuned_verify_arch/assert_platform/select_toolkit and the stamp, label and
# release functions, shared verbatim across the fleet instead of
# hand-copied-and-edited per repo.
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

# Fail loudly if this script runs on the wrong host, rather than letting a
# mismatched build silently produce a wrong-architecture wheel that only
# surfaces as a confusing failure several steps later.
gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- CUDA toolkit: CUDA_TAG (cu134) or CUDA_VER (13.4) name one from common-data/toolkits. ---
gpu_tuned_select_toolkit "${GPU_TUNED_DATA_DIR}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- Repo-specific settings (loaded after the shared device and toolkit values; may not change them). ---
gpu_tuned_load_repo_env "${GPU_TUNED_SELF_DIR}/repo.env" || return 1 2>/dev/null || exit 1
# A sourced file's own `return 1` does not stop the loader, so check that repo.env really set its key.
if [[ -z "${GPU_TUNED_FA2_ARCH:-}" ]]; then
    echo "ERROR: tuned/repo.env did not set GPU_TUNED_FA2_ARCH for ${GPU_TUNED_VARIANT}." >&2
    return 1 2>/dev/null || exit 1
fi
export GPU_TUNED_FA2_ARCH GPU_TUNED_UPSTREAM_REF

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GPU_TUNED_TOOLKIT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_PATH="${CUDA_HOME}" CUDACXX="${CUDA_HOME}/bin/nvcc"
# CUDA_VERSION_COMPACT (e.g. "134" for CUDA 13.4) is the toolkit id's digits.
CUDA_VERSION_COMPACT="${CUDA_TAG#cu}"
export CUDA_VERSION_COMPACT

# --- Output layout: one tree per toolkit, tuned/build/<cuda-tag>/ (gitignored via build/; docs/ARCHITECTURE.md C1/C2):
#     venv/ (build and test venv), wheels/ (the pinned torch wheel), wheel/ (the unstamped wheel), dist/ (the
#     stamped, packaged wheel), releases/ (test logs and the ready-marker), temp/ (the CMake/ninja tree), ccache/,
#     and timing.tsv. env.sh also refuses a CMake tree configured for another toolkit or venv, and the legacy
#     in-repo build/temp.* tree (below). ---
GPU_TUNED_REPO_ROOT="$(cd "${GPU_TUNED_SELF_DIR}/.." && pwd)"
# A tree last used by another variant is refused, never mixed (gpu_tuned_claim_toolkit_tree).
gpu_tuned_claim_toolkit_tree "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1
GPU_TUNED_BUILD_ROOT="${GPU_TUNED_REPO_ROOT}/tuned/build/${CUDA_TAG}"
GPU_TUNED_VENV_DIR="${GPU_TUNED_BUILD_ROOT}/venv"
GPU_TUNED_WHEELS_DIR="${GPU_TUNED_BUILD_ROOT}/wheels"
GPU_TUNED_WHEEL_DIR="${GPU_TUNED_BUILD_ROOT}/wheel"
GPU_TUNED_DIST_DIR="$(gpu_tuned_out_dir dist "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
GPU_TUNED_RELEASES_DIR="$(gpu_tuned_out_dir releases "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
# The CMake/ninja tree. setup.py would use build/temp.<platform>-<python>/ in the repo; wheel.sh
# puts it here instead (no tracked-file edit), at a fixed path per toolkit, which is what lets
# ccache hit and keeps ninja's log for the timings.
GPU_TUNED_BUILD_TEMP="${GPU_TUNED_BUILD_ROOT}/temp"
export GPU_TUNED_BUILD_ROOT GPU_TUNED_VENV_DIR GPU_TUNED_WHEELS_DIR GPU_TUNED_WHEEL_DIR GPU_TUNED_DIST_DIR GPU_TUNED_RELEASES_DIR GPU_TUNED_BUILD_TEMP
# ccache per toolkit. setup.py (upstream's) already passes ccache to CMake as the C++ and CUDA
# compiler launcher whenever it is on PATH (sccache first, if present); CCACHE_DIR makes
# ccache ignore ~/.config/ccache/ccache.conf, so the size cap is set here.
export CCACHE_DIR="${GPU_TUNED_BUILD_ROOT}/ccache"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"

# The torch this build is pinned to (tuned/torch.pins): URL, file name and version, for the venv
# install, the version assertion in build.sh, the smoke test and the stamp's "deps" field.
GPU_TUNED_TORCH_URL=""; GPU_TUNED_TORCH_VERSION=""
if [ -f "${GPU_TUNED_SELF_DIR}/torch.pins" ]; then
    GPU_TUNED_TORCH_URL="$(awk -v v="${GPU_TUNED_VARIANT}" -v c="${CUDA_TAG}" '$1 !~ /^#/ && $1==v && $2==c {print $3}' "${GPU_TUNED_SELF_DIR}/torch.pins")"
    if [ -n "${GPU_TUNED_TORCH_URL}" ]; then
        GPU_TUNED_TORCH_VERSION="$(basename "${GPU_TUNED_TORCH_URL}" | sed -E 's/^torch-([^-]+)-.*/\1/; s/%2B/+/')"
        export GPU_TUNED_BUILD_INFO_DEPS="torch ${GPU_TUNED_TORCH_VERSION} ${GPU_TUNED_TORCH_URL%/download/*}/tag/$(basename "$(dirname "${GPU_TUNED_TORCH_URL}")")"
    fi
fi
export GPU_TUNED_TORCH_URL GPU_TUNED_TORCH_VERSION

# The CMake source tree is the repo root (CMakeLists.txt), where the shared cache check defaults to cpp/.
export GPU_TUNED_CMAKE_SOURCE_DIR="${GPU_TUNED_REPO_ROOT}"

# fa_vllm_assert_build_tree: fail if the CMake tree this build would compile into was configured for another
# toolkit, source tree or Python (it would silently keep the old torch/CUDA), or if the old in-repo tree exists.
# The toolkit and source-tree checks are the shared gpu_tuned_assert_cmake_cache_consistent; the venv-Python
# and old-layout checks are this repo's own.
fa_vllm_assert_build_tree() {
    local cached legacy cache="${GPU_TUNED_BUILD_TEMP}/CMakeCache.txt"
    # The old layout's tree (setup.py's default location) would be compiled into by nothing and
    # timed by nothing: refuse it rather than let two trees drift.
    for legacy in "${GPU_TUNED_SELF_DIR}"/../build/temp.*/CMakeCache.txt; do
        [ -f "${legacy}" ] || continue
        echo "ERROR: ${legacy} is the old in-repo CMake tree; builds now use ${GPU_TUNED_BUILD_TEMP}. Remove it: rm -rf $(dirname "${legacy}")" >&2
        return 1
    done
    gpu_tuned_assert_cmake_cache_consistent "${GPU_TUNED_BUILD_TEMP}" || return 1
    [ -f "${cache}" ] || return 0
    cached="$(sed -n 's/^Python_EXECUTABLE:[A-Z]*=//p' "${cache}" | head -1)"
    if [ -n "${cached}" ] && [ -n "${GPU_TUNED_VENV_DIR:-}" ] && [[ "${cached}" != "${GPU_TUNED_VENV_DIR}"/* ]]; then
        echo "ERROR: ${cache} was configured with Python ${cached}, not this build's venv ${GPU_TUNED_VENV_DIR}. Remove that tree first: rm -rf ${GPU_TUNED_BUILD_TEMP}" >&2
        return 1
    fi
}

export CUDA_ARCHS="${GPU_TUNED_FA2_ARCH}"
export FA2_TUNED_ARCH="${GPU_TUNED_FA2_ARCH}"

echo "[tuned/env] GPU_TUNED_VARIANT=${GPU_TUNED_VARIANT} FA2_TUNED_ARCH=${FA2_TUNED_ARCH} CUDA_HOME=${CUDA_HOME} tree=${GPU_TUNED_BUILD_ROOT}"
