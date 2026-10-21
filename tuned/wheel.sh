#!/bin/bash
# tuned/wheel.sh <variant> — make the unstamped vllm_flash_attn wheel for one GPU variant and one CUDA toolkit, into
# tuned/build/<cuda-tag>/wheel/, and record the release label and the build commit. This is the wheel step of
# tuned/build.sh, which prepares the build venv and the pinned torch first; it refuses to run without that venv.
# It never publishes, tests or packages (package.sh stamps a copy, H3).
#
#   CUDA_TAG=cu134 bash tuned/wheel.sh gb10     (normally run by tuned/build.sh)
#
# The release label (E3/E4) is allocated here, once: HEAD's UTC commit date and the lowest unused sequence among
# origin's tags (GPU_TUNED_LABEL_SEQ=0 skips that lookup for a build that will never be published). setup.py puts it
# in the wheel's local version through FLASH_ATTN_LOCAL_VERSION. release-label and build-commit are written last,
# so their presence means the build finished. package.sh reads the label and never recomputes it: the
# release-notes commit moves HEAD and its date.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/wheel.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

[[ -d "${GPU_TUNED_VENV_DIR}" ]] || {
    echo "ERROR: ${GPU_TUNED_VENV_DIR} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} (it prepares the build venv, then calls this script)." >&2
    exit 1
}
# The venv's torch is built for one CUDA release; compiling against another toolkit would label a wheel cu<X> that
# links a cu<Y> torch, and every later gate runs in this same venv and would pass. Refuse before anything compiles.
VENV_TORCH_CUDA="$("${GPU_TUNED_VENV_DIR}/bin/python" -c 'import torch; print(torch.version.cuda)' 2>/dev/null | tail -1)"
[[ -n "${VENV_TORCH_CUDA}" ]] || {
    echo "ERROR: no importable torch in ${GPU_TUNED_VENV_DIR}; tuned/build.sh installs or checks it." >&2
    exit 1
}
[[ "${VENV_TORCH_CUDA}" == "${CUDA_VER}" ]] || {
    echo "ERROR: ${GPU_TUNED_VENV_DIR} holds a torch built for CUDA ${VENV_TORCH_CUDA}, but ${CUDA_TAG} is CUDA ${CUDA_VER}." >&2
    echo "       Build with the toolkit torch was built for (CUDA_TAG=cu${VENV_TORCH_CUDA//./}), or use a venv with a torch for ${CUDA_TAG}." >&2
    exit 1
}
if [[ -n "${GPU_TUNED_TORCH_VERSION}" ]]; then
    VENV_TORCH="$("${GPU_TUNED_VENV_DIR}/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null | tail -1)"
    [[ "${VENV_TORCH}" == "${GPU_TUNED_TORCH_VERSION}" ]] || {
        echo "ERROR: the build venv has torch ${VENV_TORCH}, but tuned/torch.pins requires ${GPU_TUNED_TORCH_VERSION}." >&2
        exit 1
    }
fi
gpu_tuned_verify_venv "${GPU_TUNED_VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${GPU_TUNED_VENV_DIR}/bin/activate"
# A CMake tree configured for another toolkit or another venv's Python would silently keep them.
fa_vllm_assert_build_tree

# setup.py's get_version() reads __version__ from vllm_flash_attn/__init__.py (a static string that
# only moves when upstream bumps it) and appends "+$FLASH_ATTN_LOCAL_VERSION" when that is set.
# The local label is <variant>.cu<cuda>.<YYYYMMDD>[.<seq>] from tuned-common (docs/VERSIONING.md):
# the date is HEAD's UTC committer date, the sequence the lowest unused one among this repo's
# release tags. It replaces the tuning.<N> commit counter, which a release-branch merge could inflate silently.
LOCAL_VERSION="$(gpu_tuned_build_local_version "${GPU_TUNED_VARIANT}" "${REPO_ROOT}")" || exit 1
[[ "${LOCAL_VERSION}" == "${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}."* ]] || {
    echo "ERROR: label ${LOCAL_VERSION} disagrees with ${CUDA_TAG} (from ${CUDA_HOME}); nvcc and the toolkit definition must report the same release." >&2
    exit 1
}
LABEL="${LOCAL_VERSION#*.cu${CUDA_VERSION_COMPACT}.}"
export FLASH_ATTN_LOCAL_VERSION="${LOCAL_VERSION}"
echo "Release label: ${LABEL} (local version ${LOCAL_VERSION})"

# Remove the previous build's markers first: they are rewritten only if this build succeeds.
rm -f "${GPU_TUNED_BUILD_ROOT}/build-commit" "${GPU_TUNED_BUILD_ROOT}/release-label"
mkdir -p "${GPU_TUNED_WHEEL_DIR}"
rm -f "${GPU_TUNED_WHEEL_DIR}"/vllm_flash_attn-*.whl

echo "=========================================="
echo "Building vllm_flash_attn wheel for ${GPU_TUNED_HW_LABEL}"
echo "=========================================="
echo "FA2_TUNED_ARCH:         ${FA2_TUNED_ARCH}"
echo "CUDA_HOME:              ${CUDA_HOME}"
echo "FLASH_ATTN_LOCAL_VERSION: ${FLASH_ATTN_LOCAL_VERSION}"
echo "Python:                 $(python3 --version)"
echo "Git commit:             $(git rev-parse --short HEAD)"
echo ""

# Where ninja logs this build's targets, and how much of the log predates it. The log accumulates
# across runs, so only what is appended after this point is timed.
NINJA_LOG="${GPU_TUNED_BUILD_TEMP}/.ninja_log"
NINJA_OFFSET=0
[[ -f "${NINJA_LOG}" ]] && NINJA_OFFSET="$(wc -c < "${NINJA_LOG}")"
BUILD_T0="${EPOCHSECONDS}"

# The wheel's compile must use the same fixed tree every build uses (GPU_TUNED_BUILD_TEMP), so it finds
# the objects already built and ccache sees the same paths. A --build-temp on the command line does not
# survive: bdist_wheel re-initialises the build subcommands and the default location comes back. A
# distutils config file is read by every invocation, so a throwaway HOME holds a .pydistutils.cfg for this one
# command (nothing in the repo or the real home directory changes).
# --skip-dependency-check: setup.py's install_requires pins torch to whatever stock version existed
# upstream when that constant was last set; build's pre-flight check rejects our torch for not
# matching it, although it is what tuned/build.sh installed and ran against (same reason the
# smoke tests install with --no-deps).
DISTUTILS_HOME="$(mktemp -d)"
printf '[build]\nbuild_temp = %s\n[build_ext]\nbuild_temp = %s\n' "${GPU_TUNED_BUILD_TEMP}" "${GPU_TUNED_BUILD_TEMP}" > "${DISTUTILS_HOME}/.pydistutils.cfg"
HOME="${DISTUTILS_HOME}" python3 -m build --wheel --no-isolation --skip-dependency-check --outdir "${GPU_TUNED_WHEEL_DIR}"
rm -rf "${DISTUTILS_HOME}"
BUILD_WALL=$(( EPOCHSECONDS - BUILD_T0 ))

echo ""
echo "Build timing (wall ${BUILD_WALL}s):"
if [[ -f "${NINJA_LOG}" ]] && gpu_tuned_ninja_timing "${NINJA_LOG}" "${NINJA_OFFSET}" "${GPU_TUNED_BUILD_ROOT}/timing.tsv" "CMakeFiles/"; then
    echo "  per-target timings: ${GPU_TUNED_BUILD_ROOT}/timing.tsv"
else
    echo "  (no per-target timings: nothing was rebuilt, or ninja logged nothing)"
fi
if command -v ccache >/dev/null 2>&1; then
    echo "ccache: $(ccache -s | grep -E 'Hits:|Misses:' | head -2 | tr -s ' ' | tr '\n' ' ')"
fi

WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/vllm_flash_attn-*.whl 2>/dev/null | head -1 || true)"
[[ -n "${WHEEL}" ]] || { echo "ERROR: no wheel found in ${GPU_TUNED_WHEEL_DIR}" >&2; exit 1; }
# Take the version from the built wheel's file name rather than re-deriving it (no drift between
# what setup.py computed and what this script assumes).
VERSION="$(gpu_tuned_wheel_version "${WHEEL}" vllm_flash_attn)" || exit 1
[[ "${VERSION}" == *"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}" ]] || {
    echo "ERROR: $(basename "${WHEEL}") is version ${VERSION}, not carrying +${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from the release label." >&2
    exit 1
}
gpu_tuned_check_wheel_label "${WHEEL}"
echo "Built wheel: $(basename "${WHEEL}") ($(du -sh "${WHEEL}" | awk '{print $1}'))"

# build-commit: the source the wheel was made from. A dirty tree is recorded as such and package.sh refuses it.
BUILD_COMMIT="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || BUILD_COMMIT="${BUILD_COMMIT}-dirty"
printf '%s\n' "${LABEL}" > "${GPU_TUNED_BUILD_ROOT}/release-label"
printf '%s\n' "${BUILD_COMMIT}" > "${GPU_TUNED_BUILD_ROOT}/build-commit"

echo ""
echo "Wheel:        ${WHEEL}"
echo "Label:        ${LABEL}"
echo "Build commit: ${BUILD_COMMIT}"
