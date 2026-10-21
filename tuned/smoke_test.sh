#!/bin/bash
# tuned/smoke_test.sh <variant> — prove the PACKAGED wheel works, before it is released. It installs the stamped
# wheel tuned/package.sh made (named in package-info.txt) into the toolkit's build venv with --no-deps, and checks:
#   1. the wheel's sha256 matches package-info.txt and its .sha256 sidecar;
#   2. the installed _vllm_fa2_C.abi3.so is byte-identical to the packaged wheel's copy, so the bits in the
#      package are the bits that run;
#   3. the installed distribution is exactly the packaged version, vllm_flash_attn loads from the venv, and the
#      extension's build-info JSON reads back as this release, variant, toolkit and build commit (H7);
#   4. the libtorch and c10 libraries a process maps are the venv's, and the toolkit libraries resolve (G2);
#   5. tuned/numeric_check.py passes against the package (flash_attn_varlen_func fa_version=2 vs SDPA);
#   6. the wheel installs into a CLEAN throwaway venv (gpu_tuned_smoke_wheel): exactly one vllm_flash_attn at the
#      packaged version, loaded from inside that venv, with the pinned torch and no stock torch pulled in.
# A smoke test, not the suite: tuned/full_test.sh ran the gate against the unstamped wheel. What is new here is that
# the bits in the package are the ones running. Writes SMOKE_RESULTS_<variant>.log beside TEST_RESULTS; release.sh
# requires a passing log newer than the packaged wheel. A pass needs the end marker, so an abort cannot pass.
# Step 6 needs the network for torch's own pure-python dependencies, as the old wheel.sh smoke test did.
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/smoke_test.sh <variant>}"
REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPODIR}" || exit 1
# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

INFO="${GPU_TUNED_DIST_DIR}/package-info.txt"
[[ -f "${INFO}" ]] || { echo "ERROR: ${INFO} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/package.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
info() { sed -n "s/^$1=//p" "${INFO}"; }
WHEEL="${GPU_TUNED_DIST_DIR}/$(info wheel)"; WANT_SHA="$(info sha256)"; VERSION="$(info version)"
[[ -f "${WHEEL}" ]] || { echo "ERROR: ${WHEEL} not found." >&2; exit 1; }
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
LOG="${GPU_TUNED_RELEASES_DIR}/SMOKE_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${LOG}"

PY="${GPU_TUNED_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: ${PY} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
FAILS=0
fail() { echo "  FAIL: $*"; FAILS=$((FAILS + 1)); }
ok() { echo "  ok:   $*"; }
export CUDA_VISIBLE_DEVICES="${GPU_TUNED_TEST_GPUS:-0}"
SO_MEMBER="vllm_flash_attn/_vllm_fa2_C.abi3.so"

{
    echo "vllm_flash_attn tuned smoke test -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wheel:   $(basename "${WHEEL}")"
    echo "packaged commit: $(info commit)"
    echo ""

    got="$(sha256sum "${WHEEL}" | awk '{print $1}')"
    [[ "${got}" == "${WANT_SHA}" ]] && ok "wheel sha256 matches package-info.txt" || fail "wheel sha256 ${got} does not match package-info.txt (${WANT_SHA})"
    gpu_tuned_verify_sha256_sidecar "${WHEEL}" && ok "sidecar .sha256 verifies" || fail "sidecar .sha256 does not verify"

    # Install the packaged wheel over the unstamped one the test run used. --no-deps: the venv holds the torch the
    # build used.
    if "${PY}" -m pip install --quiet --no-deps --force-reinstall "${WHEEL}"; then
        ok "packaged wheel installed into ${GPU_TUNED_VENV_DIR} (--no-deps)"
    else
        fail "pip could not install the packaged wheel"
    fi
    SITE_DIR="$(cd "${WORK}" && "${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

    # Byte identity of the extension with the packaged wheel's copy.
    same="$("${PY}" -I - "${WHEEL}" "${SO_MEMBER}" "${SITE_DIR}/${SO_MEMBER}" <<'PY'
import hashlib, sys, zipfile
wheel, member, installed = sys.argv[1:4]
with zipfile.ZipFile(wheel) as z:
    packed = hashlib.sha256(z.read(member)).hexdigest()
with open(installed, "rb") as f:
    here = hashlib.sha256(f.read()).hexdigest()
print("same" if packed == here else "differs")
PY
)"
    [[ "${same}" == same ]] && ok "installed ${SO_MEMBER} is byte-identical to the packaged wheel's copy" || fail "installed ${SO_MEMBER} differs from the packaged wheel's copy"

    # The distribution version and where vllm_flash_attn loads from (an empty directory, so the source tree is not on
    # the path).
    meta="$(cd "${WORK}" && "${PY}" -I -c 'import importlib.metadata as md, vllm_flash_attn; print("dist=" + md.version("vllm_flash_attn")); print("file=" + vllm_flash_attn.__file__)' 2>&1)"
    echo "${meta}" | sed 's/^/  | /'
    [[ "$(sed -n 's/^dist=//p' <<< "${meta}")" == "${VERSION}" ]] && ok "installed vllm_flash_attn is exactly ${VERSION}" || fail "installed vllm_flash_attn is not ${VERSION}"
    [[ "$(sed -n 's/^file=//p' <<< "${meta}")" == "${SITE_DIR}/"* ]] && ok "vllm_flash_attn loads from the venv" || fail "vllm_flash_attn loads from outside ${SITE_DIR}"

    if gpu_tuned_verify_build_info_json "${SITE_DIR}/${SO_MEMBER}" "release=${VERSION}" "variant=${GPU_TUNED_VARIANT}" "cuda.tag=${CUDA_TAG}" \
        && gpu_tuned_verify_build_info_json "${SITE_DIR}/${SO_MEMBER}" "fork.commit=$(info built_commit)" "fa2.arch=${GPU_TUNED_FA2_ARCH}" >/dev/null 2>&1; then
        ok "${SO_MEMBER}: build-info stamp reads back as release ${VERSION}, ${GPU_TUNED_VARIANT}, ${CUDA_TAG}, the build commit and arch ${GPU_TUNED_FA2_ARCH}"
    else
        fail "${SO_MEMBER}: build-info stamp does not read back as this release"
    fi

    # Mapped libraries: a process that imports the extension maps the venv's libtorch/libc10, from one file each.
    maps="$(cd "${WORK}" && "${PY}" -I -c '
import collections, hashlib, os, sys
import torch
from vllm_flash_attn import _vllm_fa2_C
torch.zeros(1, device="cuda")
by = collections.defaultdict(set)
for line in open("/proc/self/maps"):
    p = line.split()[-1] if line.split() else ""
    if ".so" in p and p.startswith("/"):
        by[os.path.basename(p).split(".so")[0]].add(os.path.realpath(p))
# A name mapped from two files is a failure unless the files are byte-identical (the pip nvidia-cuda-runtime wheel and the
# toolkit ship the same libcudart). Python extension modules may share a file name across packages.
def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
bad = [f"{k}: {sorted(v)}" for k, v in by.items() if len(v) > 1 and ".cpython-" not in k and len({digest(x) for x in v}) > 1]
bad += [f"{k} from {sorted(v)}" for k, v in by.items() if k.startswith(("libtorch", "libc10", "_vllm_fa2_C")) and not all(x.startswith(sys.prefix) for x in v)]
same = [k for k, v in by.items() if len(v) > 1 and ".cpython-" not in k and len({digest(x) for x in v}) == 1]
print("\n".join(bad) if bad else "clean" + (" (byte-identical copies of: " + ", ".join(sorted(same)) + ")" if same else ""))
' 2>&1 | tail -5)"
    [[ "${maps}" == clean* ]] && ok "every mapped libtorch/libc10/_vllm_fa2_C library is the venv's, and no library name maps from two different files ${maps#clean}" \
        || fail "library identity: ${maps}"
    gpu_tuned_verify_toolkit_libs "${SITE_DIR}/${SO_MEMBER}" && ok "toolkit libraries resolve under ${CUDA_HOME}" \
        || fail "toolkit libraries do not resolve under ${CUDA_HOME}"

    out="$(cd "${WORK}" && "${PY}" -I "${REPODIR}/tuned/numeric_check.py" 2>&1)"; rc=$?
    printf '%s\n' "${out}" | tail -6 | sed 's/^/  | /'
    [[ ${rc} -eq 0 ]] && ok "tuned/numeric_check.py passes against the package" || fail "tuned/numeric_check.py (exit ${rc})"

    # A clean venv, as a user of the wheel would have: torch pinned to the very wheel this was built against (a loose
    # torch requirement would resolve to stock torch from PyPI), and vllm_flash_attn.__version__ is a static string,
    # so only its base version is compared.
    SMOKE_PIN="" SMOKE_LINKS=""
    [[ -z "${GPU_TUNED_TORCH_VERSION}" ]] || { SMOKE_PIN="torch==${GPU_TUNED_TORCH_VERSION}"; SMOKE_LINKS="${GPU_TUNED_WHEELS_DIR}"; }
    if GPU_TUNED_SMOKE_MODULE_VERSION=base GPU_TUNED_SMOKE_PIN="${SMOKE_PIN}" GPU_TUNED_SMOKE_EXTRA_LINKS="${SMOKE_LINKS}" \
        gpu_tuned_smoke_wheel "${WHEEL}" vllm_flash_attn vllm_flash_attn; then
        ok "the wheel installs and loads in a clean venv with the pinned torch"
    else
        fail "the wheel does not install and load in a clean venv (gpu_tuned_smoke_wheel)"
    fi

    echo ""
    if [[ ${FAILS} -eq 0 ]]; then
        echo "=== Smoke test summary: passed ==="
    else
        echo "=== Smoke test summary: FAILED (${FAILS}) ==="
    fi
    echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${LOG}"
grep -q '^=== Smoke test summary: passed ===$' "${LOG}"
