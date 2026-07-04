#!/usr/bin/env bash
# Configure-only regression test for the FA3 auto-skip CMake logic: when the
# requested CUDA_ARCHS don't intersect Hopper (9.0a), _vllm_fa3_C must become
# a no-op target instead of compiling FA3 kernels with no gencode flags.
#
# Runs `cmake` configure (no build) twice, once per arch case, and inspects
# the generated build.ninja for the presence/absence of real FA3 compile
# rules. Requires a Python interpreter (PYTHON_EXECUTABLE, default `python3`)
# with a CUDA-enabled torch install matching this repo's
# PYTHON_SUPPORTED_VERSIONS/TORCH_SUPPORTED_VERSION_CUDA.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../" && pwd)
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-$(command -v python3)}
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

FA3_SOURCE_OBJECT="hopper/flash_api_stable.cpp.o"
SKIP_MESSAGE="skipping _vllm_fa3_C build"

configure() {
    local cuda_archs=$1 out_dir=$2 log=$3
    cmake -S "$REPO_ROOT" -B "$out_dir" -G Ninja \
        -DVLLM_TARGET_DEVICE=cuda \
        -DPython_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -DCUDA_ARCHS="$cuda_archs" \
        >"$log" 2>&1
}

assert_skipped() {
    local out_dir=$1 log=$2
    grep -q "$SKIP_MESSAGE" "$log" \
        || { echo "FAIL: expected skip message in $log"; cat "$log"; return 1; }
    grep -q "build _vllm_fa3_C: phony" "$out_dir/build.ninja" \
        || { echo "FAIL: expected _vllm_fa3_C to be a phony target in $out_dir/build.ninja"; return 1; }
    if grep -q "$FA3_SOURCE_OBJECT" "$out_dir/build.ninja"; then
        echo "FAIL: FA3 sources should not have compile rules when skipped"
        return 1
    fi
}

assert_built() {
    local out_dir=$1 log=$2
    if grep -q "$SKIP_MESSAGE" "$log"; then
        echo "FAIL: did not expect skip message in $log"
        return 1
    fi
    grep -q "$FA3_SOURCE_OBJECT" "$out_dir/build.ninja" \
        || { echo "FAIL: expected a real compile rule for $FA3_SOURCE_OBJECT in $out_dir/build.ninja"; return 1; }
}

echo "== non-Hopper archs (8.0): expect _vllm_fa3_C skipped =="
configure "8.0" "$WORK_DIR/no_hopper" "$WORK_DIR/no_hopper.log"
assert_skipped "$WORK_DIR/no_hopper" "$WORK_DIR/no_hopper.log"
echo "PASS"

echo "== Hopper arch (9.0a): expect _vllm_fa3_C built =="
configure "9.0a" "$WORK_DIR/hopper" "$WORK_DIR/hopper.log"
assert_built "$WORK_DIR/hopper" "$WORK_DIR/hopper.log"
echo "PASS"
