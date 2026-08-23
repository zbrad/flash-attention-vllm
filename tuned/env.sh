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
        "${hw_label}" "https://github.com/zbrad/flash-attention" "flash_attn_build_info"
}
