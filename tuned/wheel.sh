#!/bin/bash
# tuned/wheel.sh <variant> — package a built vllm_flash_attn tree into a
# wheel and publish it as a real GitHub release, matching the tag scheme
# already established by the (manually) published GB10 wheel
# (v2.7.2.post1-gb10-cu133). Requires tuned/build.sh <variant> to have
# already succeeded (this reuses that venv, does not rebuild).
set -euo pipefail

GPU_TUNED_ARG_VARIANT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

VENV_DIR="${REPO_ROOT}/.venv-${GPU_TUNED_VARIANT}-vllm"
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

# setup.py's get_version() reads __version__ from vllm_flash_attn/__init__.py
# and optionally appends "+FLASH_ATTN_LOCAL_VERSION" if set, else
# auto-appends "+cu<compact>" when nvcc's version differs from
# MAIN_CUDA_VERSION ("12.1"). The already-published GB10 wheel used that
# auto path (plain "+cu133", no variant marker) -- fine for a single
# variant, but multiple tuned variants at the same CUDA version would
# collide on an identical wheel filename. Set FLASH_ATTN_LOCAL_VERSION
# explicitly here to disambiguate, matching every other repo's wheel
# naming this session -- a deliberate deviation from the exact original
# filename, not a bug.
#
# tuning-vN = commits on tuned-builds since it diverged from main (i.e.
# commits ahead of upstream/vllm-project's flash-attention) -- same
# convention adopted fleet-wide from zbrad/pytorch's tuned/wheel.sh: the
# static __version__ above only moves when upstream bumps it, so on its
# own it can't say "how much of our own tuned-builds work landed since an
# earlier wheel was built."
TUNED_COMMIT_COUNT="$(git rev-list --count main..HEAD)"
export FLASH_ATTN_LOCAL_VERSION="${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.tuning-v${TUNED_COMMIT_COUNT}"

echo "=========================================="
echo "Packaging vllm_flash_attn wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "FLASH_ATTN_LOCAL_VERSION: ${FLASH_ATTN_LOCAL_VERSION}"
echo ""

pip install --upgrade build wheel
rm -rf "${REPO_ROOT}/dist"
# --skip-dependency-check: setup.py's install_requires pins
# torch=={PYTORCH_VERSION} (whatever stock torch existed upstream when
# that constant was last set) -- build's own pre-flight dependency check
# rejects this box's custom GB10 torch build for not matching that exact
# pin, even though it's what tuned/build.sh actually built and ran
# against (same reason tuned/build.sh's own pip install uses --no-deps).
python3 -m build --wheel --no-isolation --skip-dependency-check

WHEEL="$(ls "${REPO_ROOT}"/dist/vllm_flash_attn-*.whl 2>/dev/null | head -1)"
[[ -z "${WHEEL}" ]] && { echo "ERROR: no wheel found in dist/" >&2; exit 1; }
echo "Built wheel: $(basename "${WHEEL}") ($(du -sh "${WHEEL}" | awk '{print $1}'))"

# Extract the actual package version from the built wheel filename rather
# than re-deriving it a second time (avoids any drift between what
# setup.py actually computed and what this script assumes it computed).
WHEEL_VERSION="$(gpu_tuned_wheel_version "${WHEEL}" vllm_flash_attn)" || exit 1

# _vllm_fa2_C.abi3.so is where FA2_TUNED_ARCH's actual device code lands
# -- verify + stamp it here, on the wheel's OWN contents, not a pre-build
# copy: a separate packaging invocation (python -m build) may
# rebuild/relink rather than reuse an already-stamped file byte-for-byte
# -- confirmed for pytorch's equivalent build this session (a test marker
# stamped before the build was completely absent afterward; see
# gpu_tuned_verify_build_info's header comment). FA3 (_vllm_fa3_C.abi3.so)
# is a no-op target on non-Hopper (gb10/rtx40/rtx50), so it's never built
# here -- only FA2 gets this treatment. Unpack -> stamp -> repack
# regenerates RECORD correctly, unlike a raw zip edit.
echo "Stamping build-info into the wheel's own _vllm_fa2_C.abi3.so"
UNPACK_DIR="$(mktemp -d)"
python3 -m wheel unpack "${WHEEL}" --dest "${UNPACK_DIR}"
WHEEL_SO="$(find "${UNPACK_DIR}" -name '_vllm_fa2_C.abi3.so' | head -1)"
[[ -z "${WHEEL_SO}" ]] && { echo "ERROR: _vllm_fa2_C.abi3.so not found inside ${WHEEL}." >&2; exit 1; }
gpu_tuned_verify_arch "${WHEEL_SO}" "${GPU_TUNED_FA2_ARCH}"
embed_build_info "${WHEEL_SO}" "${GPU_TUNED_VARIANT}" "vllm_flash_attn" "${WHEEL_VERSION}" "${GPU_TUNED_HW_LABEL}"
gpu_tuned_verify_build_info "${WHEEL_SO}" "vllm_flash_attn" "${WHEEL_VERSION}"
rm -f "${WHEEL}"
UNPACKED_CONTENT_DIR="$(find "${UNPACK_DIR}" -maxdepth 1 -mindepth 1 -type d)"
python3 -m wheel pack "${UNPACKED_CONTENT_DIR}" --dest-dir "${REPO_ROOT}/dist"
rm -rf "${UNPACK_DIR}"
WHEEL="$(ls "${REPO_ROOT}"/dist/vllm_flash_attn-*.whl 2>/dev/null | head -1)"
echo "Re-packed with build-info stamp: $(basename "${WHEEL}")"
# WHEEL_VERSION already includes the full "+FLASH_ATTN_LOCAL_VERSION" local
# segment (variant, cuda tag, and now tuning-vN) -- use it directly rather
# than stripping and re-appending only part of it, which would silently
# drop tuning-vN from the tag while it stayed visible in the title below.
# A literal "+" in a git tag is fine (needs %2B only in URLs that link to
# it, not in the tag itself or `gh release create`'s argument).
# Friendly title only -- RELEASE_TAG stays the exact WHEEL_VERSION.
GIT_SHA="$(git rev-parse --short HEAD)"
WHEEL_BASE_VERSION="${WHEEL_VERSION%%+*}"

RELEASE_TAG="v${WHEEL_VERSION}"
RELEASE_TITLE="vllm_flash_attn ${WHEEL_BASE_VERSION} — ${GPU_TUNED_VARIANT} tuning-v${TUNED_COMMIT_COUNT} (cu${CUDA_VERSION_COMPACT}, ${GIT_SHA}) — ${GPU_TUNED_HW_LABEL} wheel"

echo ""
echo "Publishing wheel to GitHub release ${RELEASE_TAG}..."
gpu_tuned_publish_release "zbrad/flash-attention-vllm" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "vllm_flash_attn ${WHEEL_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (FA2_TUNED_ARCH=${FA2_TUNED_ARCH})." \
    "${WHEEL}#$(basename "${WHEEL}")"

echo ""
echo "Release: https://github.com/zbrad/flash-attention-vllm/releases/tag/${RELEASE_TAG}"
echo "Done."
