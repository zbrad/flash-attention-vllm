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

# Fail loudly if this script runs on the wrong host, rather than letting a
# mismatched build silently produce a wrong-architecture wheel that only
# surfaces as a confusing failure several steps later.
if [[ "$(uname -m)" != "${GPU_TUNED_PLATFORM}" ]]; then
    echo "ERROR: tuned/env.sh: expected platform '${GPU_TUNED_PLATFORM}' for" \
         "variant '${GPU_TUNED_VARIANT}', but uname -m reports '$(uname -m)'." >&2
    return 1 2>/dev/null || exit 1
fi

# --- Resolve CUDA_HOME to the highest installed toolkit when not explicitly set ---
flash_attn_installed_cuda_toolkits() {
    local d
    for d in /usr/local/cuda-[0-9]*; do
        [ -d "$d" ] && basename "$d" | sed 's/^cuda-//'
    done | sort -V
}

if [ -z "${CUDA_HOME:-}" ]; then
    _fa_highest="$(flash_attn_installed_cuda_toolkits | tail -1)"
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
