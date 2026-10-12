#!/bin/bash
# tuned/wheel.sh <variant> — package a built vllm_flash_attn tree into a wheel and publish it as a
# real GitHub release. Requires tuned/build.sh <variant> to have already succeeded (this reuses
# that venv, it does not set one up).
#
# FLASH_ATTN_WHEEL_NO_PUBLISH=1 stops before the GitHub release: label, packaging, stamp and the
# clean-venv smoke test all run, nothing is sent anywhere. The wheel lands in
# tuned/build/cu<ver>/dist/ (env.sh).
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

[[ -n "${GPU_TUNED_BUILD_ROOT:-}" ]] || { echo "ERROR: CUDA_HOME did not resolve to a /usr/local/cuda-X.Y directory; no build tree." >&2; exit 1; }
VENV_DIR="${GPU_TUNED_VENV_DIR}"
[[ -d "${VENV_DIR}" ]] || {
    echo "ERROR: ${VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
}
gpu_tuned_verify_venv "${VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

[[ -n "${CUDA_VERSION_COMPACT:-}" ]] || {
    echo "ERROR: CUDA_VERSION_COMPACT not set (CUDA_HOME must resolve to a" \
         "/usr/local/cuda-X.Y directory) -- cannot derive the version string." >&2
    exit 1
}

# setup.py's get_version() reads __version__ from vllm_flash_attn/__init__.py (a static string that
# only moves when upstream bumps it) and appends "+$FLASH_ATTN_LOCAL_VERSION" when that is set.
# The local label is <variant>.cu<cuda>.<YYYYMMDD>[.<seq>] from tuned-common (docs/VERSIONING.md):
# the date is HEAD's UTC committer date, the sequence the lowest unused one among this repo's
# release tags; GPU_TUNED_LABEL_SEQ=0 skips that lookup for a build that will never be published.
# It replaces the tuning.<N> commit counter, which a release-branch merge could inflate silently.
FLASH_ATTN_LOCAL_VERSION="$(gpu_tuned_build_local_version "${GPU_TUNED_VARIANT}" "${REPO_ROOT}")" || exit 1
[[ "${FLASH_ATTN_LOCAL_VERSION}" == *".cu${CUDA_VERSION_COMPACT}."* ]] || {
    echo "ERROR: label ${FLASH_ATTN_LOCAL_VERSION} disagrees with CUDA_VERSION_COMPACT=${CUDA_VERSION_COMPACT} (from ${CUDA_HOME})." >&2
    exit 1
}
export FLASH_ATTN_LOCAL_VERSION

echo "=========================================="
echo "Packaging vllm_flash_attn wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "FLASH_ATTN_LOCAL_VERSION: ${FLASH_ATTN_LOCAL_VERSION}"
echo ""

pip install --upgrade build wheel
rm -rf "${GPU_TUNED_DIST_DIR}"
# The wheel's compile must use the same fixed tree build.sh used (GPU_TUNED_BUILD_TEMP), so it finds
# the objects already built and ccache sees the same paths, instead of compiling twice. A
# --build-temp on the command line does not survive: bdist_wheel re-initialises the build
# subcommands and the default location comes back. A distutils config file is read by every
# invocation, so a throwaway HOME holds a .pydistutils.cfg for this one command (nothing in the repo
# or the real home directory changes).
# --skip-dependency-check: setup.py's install_requires pins torch to whatever stock version existed
# upstream when that constant was last set; build's pre-flight check rejects our torch for not
# matching it, although it is what tuned/build.sh built and ran against (same reason build.sh's own
# pip install uses --no-deps).
DISTUTILS_HOME="$(mktemp -d)"
printf '[build]\nbuild_temp = %s\n[build_ext]\nbuild_temp = %s\n' "${GPU_TUNED_BUILD_TEMP}" "${GPU_TUNED_BUILD_TEMP}" > "${DISTUTILS_HOME}/.pydistutils.cfg"
HOME="${DISTUTILS_HOME}" python3 -m build --wheel --no-isolation --skip-dependency-check --outdir "${GPU_TUNED_DIST_DIR}"
rm -rf "${DISTUTILS_HOME}"

WHEEL="$(ls "${GPU_TUNED_DIST_DIR}"/vllm_flash_attn-*.whl 2>/dev/null | head -1)"
[[ -z "${WHEEL}" ]] && { echo "ERROR: no wheel found in ${GPU_TUNED_DIST_DIR}" >&2; exit 1; }
echo "Built wheel: $(basename "${WHEEL}") ($(du -sh "${WHEEL}" | awk '{print $1}'))"

# Take the version from the built wheel's file name rather than re-deriving it (no drift between
# what setup.py computed and what this script assumes).
WHEEL_VERSION="$(gpu_tuned_wheel_version "${WHEEL}" vllm_flash_attn)" || exit 1

# _vllm_fa2_C.abi3.so is where FA2_TUNED_ARCH's device code lands -- verify and stamp it here, on
# the wheel's OWN contents, not a pre-build copy (python -m build may relink rather than reuse an
# already-stamped file byte for byte). FA3 (_vllm_fa3_C) is a no-op target on non-Hopper variants,
# so only FA2 gets this. Unpack -> stamp -> repack regenerates RECORD correctly, unlike a raw zip
# edit.
echo "Stamping build-info into the wheel's own _vllm_fa2_C.abi3.so"
UNPACK_DIR="$(mktemp -d)"
python3 -m wheel unpack "${WHEEL}" --dest "${UNPACK_DIR}"
WHEEL_SO="$(find "${UNPACK_DIR}" -name '_vllm_fa2_C.abi3.so' | head -1)"
[[ -z "${WHEEL_SO}" ]] && { echo "ERROR: _vllm_fa2_C.abi3.so not found inside ${WHEEL}." >&2; exit 1; }
gpu_tuned_verify_arch "${WHEEL_SO}" "${GPU_TUNED_FA2_ARCH}"
embed_build_info "${WHEEL_SO}" "${GPU_TUNED_VARIANT}" "vllm_flash_attn" "${WHEEL_VERSION}" "${GPU_TUNED_HW_LABEL}"
gpu_tuned_verify_build_info "${WHEEL_SO}" "vllm_flash_attn" "${WHEEL_VERSION}" "flash_attn_build_info"
rm -f "${WHEEL}"
UNPACKED_CONTENT_DIR="$(find "${UNPACK_DIR}" -maxdepth 1 -mindepth 1 -type d)"
python3 -m wheel pack "${UNPACKED_CONTENT_DIR}" --dest-dir "${GPU_TUNED_DIST_DIR}"
rm -rf "${UNPACK_DIR}"
WHEEL="$(ls "${GPU_TUNED_DIST_DIR}"/vllm_flash_attn-*.whl 2>/dev/null | head -1)"
echo "Re-packed with build-info stamp: $(basename "${WHEEL}")"

# Load the finished wheel once in a clean venv before anything is published: the label, exactly one
# distribution at that version, the stamp, and that it imports. torch is pinned to the very wheel
# this was built against (a loose torch requirement would resolve to stock torch from PyPI), and
# vllm_flash_attn.__version__ is a static string, so only its base version is compared.
if [[ "${FLASH_ATTN_WHEEL_SKIP_SMOKE:-}" == 1 ]]; then
    echo "SKIP: wheel smoke test (FLASH_ATTN_WHEEL_SKIP_SMOKE=1); do not publish this wheel"
else
    SMOKE_ENV=(GPU_TUNED_SMOKE_MODULE_VERSION=base)
    [[ -n "${GPU_TUNED_TORCH_VERSION}" ]] && SMOKE_ENV+=("GPU_TUNED_SMOKE_PIN=torch==${GPU_TUNED_TORCH_VERSION}" "GPU_TUNED_SMOKE_EXTRA_LINKS=${GPU_TUNED_WHEELS_DIR}")
    env "${SMOKE_ENV[@]}" bash -c 'source "$1"; gpu_tuned_smoke_wheel "$2" vllm_flash_attn vllm_flash_attn flash_attn_build_info _vllm_fa2_C.abi3.so' \
        _ "${REPO_ROOT}/tuned/common.sh" "${WHEEL}" || exit 1
fi

GIT_SHA="$(git rev-parse --short HEAD)"
WHEEL_BASE_VERSION="${WHEEL_VERSION%%+*}"

# RELEASE_TAG is the exact WHEEL_VERSION (a literal "+" is fine in a git tag; it needs %2B only in
# URLs that link to it). The title is the friendly form.
RELEASE_TAG="v${WHEEL_VERSION}"
RELEASE_TITLE="vllm_flash_attn ${WHEEL_BASE_VERSION} — ${GPU_TUNED_VARIANT} ${FLASH_ATTN_LOCAL_VERSION#*.} (${GIT_SHA}) — ${GPU_TUNED_HW_LABEL} wheel"
RELEASE_NOTES="vllm_flash_attn ${WHEEL_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (FA2_TUNED_ARCH=${FA2_TUNED_ARCH}), CUDA ${CUDA_HOME:-unknown}.${GPU_TUNED_TORCH_VERSION:+ Built and smoke-tested against torch ${GPU_TUNED_TORCH_VERSION} (${GPU_TUNED_BUILD_INFO_DEPS#torch * })}"

if [[ -n "${FLASH_ATTN_WHEEL_NO_PUBLISH:-}" ]]; then
    echo ""
    echo "FLASH_ATTN_WHEEL_NO_PUBLISH set: built ${RELEASE_TAG}, not published."
    echo "Wheel: ${WHEEL}"
    exit 0
fi

echo ""
echo "Publishing wheel to GitHub release ${RELEASE_TAG}..."
gpu_tuned_publish_release "zbrad/flash-attention-vllm" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "${RELEASE_NOTES}" \
    "${WHEEL}#$(basename "${WHEEL}")"

echo ""
echo "Release: https://github.com/zbrad/flash-attention-vllm/releases/tag/${RELEASE_TAG}"
echo "Done."
