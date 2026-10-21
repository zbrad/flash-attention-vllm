#!/bin/bash
# tuned/full_test.sh <variant> — test the wheel tuned/build.sh made, for ONE variant and ONE CUDA toolkit, and
# write a results log in that toolkit's tree (tuned/build/<cuda-tag>/releases/TEST_RESULTS_<variant>.log).
#
#   CUDA_TAG=cu134 bash tuned/full_test.sh gb10
#
# The unstamped wheel (tuned/build/<cuda-tag>/wheel/vllm_flash_attn-*.whl) is installed --no-deps into the build
# venv and every check runs from an empty directory, so vllm_flash_attn and its extension are the wheel's, not the
# source tree's. The gate is:
#   1. wheel label and build commit (release-label, build-commit), H6 source check;
#   2. vllm_flash_attn and _vllm_fa2_C load from the installed wheel, the wheel ships exactly that one extension,
#      the toolkit libraries resolve (G2), the extension is single-arch (verify_arch) and carries no PTX;
#   3. tuned/numeric_check.py: flash_attn_varlen_func (fa_version=2) against SDPA (plain, causal, GQA) on the device.
# Skips by design, named in the log header: upstream's pytest suite (tests/) is not run, FA3 and FA4 are not built
# for these variants, and nothing multi-GPU (a tuned build is single-GPU). This narrower gate is the same shape as
# vllm's recorded G3 decision and needs its own attestation in compliance.conf.
#
# The log ends with "=== Full test suite summary: gates + numeric check passed ===" only when every step
# passed; release.sh greps for that exact line. The other outcome is "=== Full test suite summary: FAILED
# (see above) ===". tuned/package.sh and tuned/release.sh refuse a log without the pass line.
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/full_test.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/vllm_flash_attn-*.whl 2>/dev/null | head -1)"
[[ -n "${WHEEL}" ]] || { echo "ERROR: no wheel in ${GPU_TUNED_WHEEL_DIR}. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
[[ -n "${BUILD_COMMIT}" && -n "${LABEL}" ]] || { echo "ERROR: ${GPU_TUNED_BUILD_ROOT}/build-commit or release-label missing: the last build did not finish. Re-run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
[[ "$(basename "${WHEEL}")" == vllm_flash_attn-*"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}-"* ]] \
    || { echo "ERROR: $(basename "${WHEEL}") does not carry label ${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from release-label." >&2; exit 1; }
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
# H6: the wheel must be from the source at HEAD; only tuned/ may differ.
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
if [[ -n "${STALE_SRC}" ]]; then
    echo "ERROR: the wheel is from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2
    echo "${STALE_SRC}" | sed 's/^/    /' >&2
    echo "  Rebuild (tuned/build.sh ${GPU_TUNED_VARIANT}) before testing." >&2
    exit 1
fi

PY="${GPU_TUNED_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: ${PY} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
RESULTS_FILE="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${RESULTS_FILE}"

# Tuned builds are single-GPU: the tests see exactly one device (GPU_TUNED_TEST_GPUS, default 0).
export CUDA_VISIBLE_DEVICES="${GPU_TUNED_TEST_GPUS:-0}"
TESTDIR="$(mktemp -d)"
trap 'rm -rf "${TESTDIR}"' EXIT

echo "installing $(basename "${WHEEL}") into the build venv (--no-deps)"
"${PY}" -m pip install --quiet --no-deps --force-reinstall "${WHEEL}" \
    || { echo "ERROR: pip could not install the wheel." >&2; exit 1; }

SITE_DIR="$(cd "${TESTDIR}" && "${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
{
    echo "vllm_flash_attn tuned test run -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit:  $(git rev-parse HEAD)"
    echo "build:   ${BUILD_COMMIT} (label ${LABEL})"
    echo "wheel:   $(basename "${WHEEL}") sha256 $(sha256sum "${WHEEL}" | awk '{print $1}')"
    echo "torch:   $(cd "${TESTDIR}" && "${PY}" -I -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no device")' 2>&1 | tail -1)"
    echo "scope:   gates; _vllm_fa2_C load, arch and PTX checks; tuned/numeric_check.py (flash_attn_varlen_func fa_version=2 vs SDPA: plain, causal, GQA)"
    echo "devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (single-GPU build: multi-GPU tests are not part of its scope)"
    echo "skips by design: upstream's pytest suite (tests/) is not run; FA3 and FA4 are not built for this variant; no multi-GPU tests"
    echo "gates: wheel installed --no-deps in the build venv; extension loads from it; toolkit libs resolve under ${CUDA_HOME}; single-arch ${GPU_TUNED_FA2_ARCH}, no PTX; build commit ${BUILD_COMMIT} == HEAD outside tuned/"
    echo ""
} > "${RESULTS_FILE}"

# Gates. A function, so its status comes back through the tee pipeline's PIPESTATUS.
gate_checks() {
    local rc=0 so ptx others
    (cd "${TESTDIR}" && "${PY}" -I -c '
import vllm_flash_attn
from vllm_flash_attn import _vllm_fa2_C
print("vllm_flash_attn: " + vllm_flash_attn.__file__)
print("_vllm_fa2_C: " + _vllm_fa2_C.__file__)
') || rc=1
    so="$(cd "${TESTDIR}" && "${PY}" -I -c 'from vllm_flash_attn import _vllm_fa2_C as m; print(m.__file__)' 2>/dev/null | tail -1)"
    if [[ "${so}" == "${SITE_DIR}/"* ]]; then echo "  ok:   _vllm_fa2_C is the wheel's: ${so}"
    else echo "  FAIL: _vllm_fa2_C loads from '${so}', not from ${SITE_DIR}"; return 1; fi
    # The one extension this variant ships. FA3 is a no-op target off Hopper; anything else is a decision, not an accident.
    others="$("${PY}" -I -c 'import sys, zipfile; print(" ".join(n for n in zipfile.ZipFile(sys.argv[1]).namelist() if ".so" in n and not n.endswith("/_vllm_fa2_C.abi3.so")))' "${WHEEL}")"
    if [[ -z "${others}" ]]; then echo "  ok:   the wheel ships _vllm_fa2_C.abi3.so and no other extension"
    else echo "  FAIL: the wheel ships extensions besides _vllm_fa2_C.abi3.so: ${others}"; rc=1; fi
    if gpu_tuned_verify_toolkit_libs "${so}"; then echo "  ok:   toolkit libraries resolve under ${CUDA_HOME}"
    else echo "  FAIL: toolkit libraries do not resolve under ${CUDA_HOME}"; rc=1; fi
    gpu_tuned_verify_arch "${so}" "${GPU_TUNED_FA2_ARCH}" || rc=1
    # Pre-compiled only: PTX would be JIT-compiled by the driver at first use. A cuobjdump that cannot run is a
    # failure, not "no PTX" (H4).
    if ptx="$(cuobjdump --list-ptx "${so}" 2>&1)"; then
        ptx="$(printf '%s\n' "${ptx}" | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | tr '\n' ' ' || true)"
        if [[ -n "${ptx// /}" ]]; then echo "  FAIL: ${so##*/} embeds PTX (${ptx})"; rc=1; else echo "  ok:   no PTX in ${so##*/}"; fi
    else
        echo "  FAIL: cuobjdump --list-ptx failed; the PTX check cannot run: ${ptx}"; rc=1
    fi
    return "${rc}"
}
# logrun <command...>: run with output to the terminal and the log; returns the command's own status.
logrun() { "$@" 2>&1 | tee -a "${RESULTS_FILE}"; return "${PIPESTATUS[0]}"; }

STATUS=0
echo "=== gates ===" | tee -a "${RESULTS_FILE}"
logrun gate_checks || STATUS=1
echo "" | tee -a "${RESULTS_FILE}"
echo "=== tuned/numeric_check.py ===" | tee -a "${RESULTS_FILE}"
(cd "${TESTDIR}" && "${PY}" -I "${REPO_ROOT}/tuned/numeric_check.py") 2>&1 | tee -a "${RESULTS_FILE}"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { STATUS=1; echo "numeric_check.py FAILED" | tee -a "${RESULTS_FILE}"; }

echo "" >> "${RESULTS_FILE}"
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RESULTS_FILE}"
if [[ "${STATUS}" -eq 0 ]]; then
    echo "=== Full test suite summary: gates + numeric check passed ===" | tee -a "${RESULTS_FILE}"
else
    echo "=== Full test suite summary: FAILED (see above) ===" | tee -a "${RESULTS_FILE}"
fi
echo "Results written to ${RESULTS_FILE}"
exit "${STATUS}"
