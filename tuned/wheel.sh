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

VENV_DIR="${REPO_ROOT}/.venv-${GPU_TUNED_VARIANT}"
[[ -d "${VENV_DIR}" ]] || {
    echo "ERROR: ${VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
}
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
export FLASH_ATTN_LOCAL_VERSION="${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}"

echo "=========================================="
echo "Packaging vllm_flash_attn wheel (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "FLASH_ATTN_LOCAL_VERSION: ${FLASH_ATTN_LOCAL_VERSION}"
echo ""

# _vllm_fa2_C.abi3.so is where FA2_TUNED_ARCH's actual device code lands
# -- verify + stamp it before packaging, same discipline as the other
# tuned-builds repos' wheel.sh. FA3 (_vllm_fa3_C.abi3.so) is a no-op
# target on non-Hopper (gb10/rtx40/rtx50), so it's never built here --
# only FA2 gets this treatment.
FA2_SO="${REPO_ROOT}/vllm_flash_attn/_vllm_fa2_C.abi3.so"
if [[ -f "${FA2_SO}" ]]; then
    gpu_tuned_verify_arch "${FA2_SO}"
    embed_build_info "${FA2_SO}" "${GPU_TUNED_VARIANT}" "vllm_flash_attn" "${FLASH_ATTN_LOCAL_VERSION}" "${GPU_TUNED_HW_LABEL}"
else
    echo "ERROR: ${FA2_SO} not found -- run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2
    exit 1
fi

pip install --upgrade build
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
WHEEL_VERSION="$(basename "${WHEEL}" | sed -E 's/^vllm_flash_attn-([^-]+)-.*/\1/')"
# WHEEL_VERSION includes the "+FLASH_ATTN_LOCAL_VERSION" local-version
# segment (e.g. "2.7.2.post1+gb10.cu133") -- strip it for the release tag,
# which already appends -<variant>-cu<NNN> separately below; keeping both
# would duplicate the variant/cuda tag in the tag name (and a literal "+"
# in a git tag needs URL-encoding wherever it's linked).
WHEEL_BASE_VERSION="${WHEEL_VERSION%%+*}"

RELEASE_TAG="v${WHEEL_BASE_VERSION}-${GPU_TUNED_VARIANT}-cu${CUDA_VERSION_COMPACT}"
RELEASE_TITLE="vllm_flash_attn ${WHEEL_VERSION} — ${GPU_TUNED_HW_LABEL} wheel"

echo ""
echo "Publishing wheel to GitHub release ${RELEASE_TAG}..."
gh release create "${RELEASE_TAG}" \
    --repo zbrad/flash-attention \
    --title "${RELEASE_TITLE}" \
    --target "tuned-builds" \
    --notes "vllm_flash_attn ${WHEEL_VERSION} wheel for ${GPU_TUNED_HW_LABEL}, single-arch (FA2_TUNED_ARCH=${FA2_TUNED_ARCH})." \
    "${WHEEL}#$(basename "${WHEEL}")"

echo ""
echo "Release: https://github.com/zbrad/flash-attention/releases/tag/${RELEASE_TAG}"
echo "Done."
