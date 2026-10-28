# vllm_flash_attn 2.7.2.post1 — gb10 cu134 (2.7.2.post1+gb10.cu134.20261028.0)

Single-arch vllm_flash_attn wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FA2_TUNED_ARCH=12.1a, no PTX, FA2 only), CUDA 13.4.

- Release label: 20261028.0 (UTC date of the tuned commit)
- Built from commit 39a7a37b5dd9132e3c651fe758ce2f89789e3f04; tested at 39a7a37b5dd9132e3c651fe758ce2f89789e3f04
- Built and tested against torch 2.15.0+gb10.cu134.20261022.0 (https://github.com/zbrad/pytorch/releases/tag/v2.15.0%2Bgb10.cu134.20261022.0)
- Upstream base: upstream/main merge-base 9cd61de38763d712bb6ce56e2a02cc2bf718c89f
- Wheel sha256: 5cb77592cabffaf0a7cf53affabdfca0bb4bc66e57a1a72d364a71b043f38648

## Changes since the previous release

- Built and tested against torch 2.15.0+gb10.cu134.20261022.0 (no CPU BLAS, LAPACK or MPI); the previous pin, 20261012, links OpenMPI.
- The build venv has setuptools 84.2 or newer, and package.sh checks that and audits the wheel's runtime dependencies (the pinned torch is not on PyPI and is listed as not audited).
- No kernel or source changes from this fork's tuned code.

## Tests

Scope (from the attached test log): gates; _vllm_fa2_C load, arch and PTX checks; tuned/numeric_check.py (flash_attn_varlen_func fa_version=2 vs SDPA: plain, causal, GQA)
Skips by design: upstream's pytest suite (tests/) is not run; FA3 and FA4 are not built for this variant; no multi-GPU tests
Result: === Full test suite summary: gates + numeric check passed ===

## Install

```
pip install --no-deps "vllm_flash_attn @ https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261028.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261028.0-cp314-cp314-linux_aarch64.whl"
```
