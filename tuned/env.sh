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

# gpu_tuned_verify_arch <path-to-.so> — confirms a compiled library's
# embedded cubin(s) are EXACTLY sm_${GPU_TUNED_FA2_ARCH}, via cuobjdump.
# Same name/signature as the other tuned-builds repos' env.sh equivalents
# -- this repo previously had none of these empirical checks at all.
gpu_tuned_verify_arch() {
    local so_file="$1"
    if [[ ! -f "${so_file}" ]]; then
        echo "ERROR: gpu_tuned_verify_arch: no such file: ${so_file}" >&2
        return 1
    fi
    command -v cuobjdump >/dev/null 2>&1 || {
        echo "ERROR: gpu_tuned_verify_arch: cuobjdump not found on PATH (expected under \$CUDA_HOME/bin)." >&2
        return 1
    }
    local expected_arch found found_count
    expected_arch="$(echo "${GPU_TUNED_FA2_ARCH}" | tr -d '.')"
    found="$(cuobjdump --list-elf "${so_file}" 2>/dev/null | grep -oE 'sm_[0-9]+[a-z]?' | sort -u)"
    if [[ -z "${found}" ]]; then
        echo "ERROR: gpu_tuned_verify_arch: cuobjdump found no embedded cubins in ${so_file} at all." >&2
        return 1
    fi
    found_count="$(echo "${found}" | wc -l)"
    if [[ "${found_count}" -ne 1 ]]; then
        echo "ERROR: ${so_file} embeds MULTIPLE arch targets ($(echo "${found}" | tr '\n' ' ')) -- this is supposed to be a single-arch tuned build, not a fat multi-arch one." >&2
        return 1
    fi
    if [[ "${found}" != "sm_${expected_arch}" ]]; then
        echo "ERROR: ${so_file} is not built for sm_${expected_arch} (found: ${found})." >&2
        return 1
    fi
    echo "OK: ${so_file} confirmed single-arch ${found} (matches requested sm_${expected_arch})"
}

# embed_build_info <so_path> <variant> <package> <version> [hw_label] —
# embeds a greppable build-info string into a custom ELF section
# (.flash_attn_build_info) on the given .so, readable later via
# `readelf -p .flash_attn_build_info <so>` or plain `strings`. Safe at
# runtime: a custom section with no program-header entry is simply
# ignored by the dynamic loader. Same technique/name as the other
# tuned-builds repos' env.sh equivalents.
embed_build_info() {
    local so_path="$1" variant="$2" package="$3" version="$4" hw_label="${5:-${2}}"
    local tmp
    tmp="$(mktemp)"
    echo "flash-attention-${variant} build: ${package} v${version} (${hw_label}), https://github.com/zbrad/flash-attention, built $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${tmp}"
    # Idempotent: objcopy --add-section on a section name that already
    # exists (e.g. re-packaging without a clean) empirically corrupts its
    # own in-place rewrite ("file format not recognized" on its own temp
    # output) -- strip any prior stamp first. Same fix as the other repos'
    # tuned/env.sh, hit for real running a live verification.
    objcopy --remove-section .flash_attn_build_info "${so_path}" 2>/dev/null || true
    objcopy --add-section .flash_attn_build_info="${tmp}" "${so_path}"
    rm -f "${tmp}"
}
