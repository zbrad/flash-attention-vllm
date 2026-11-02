# vllm_flash_attn 2.7.2.post1 — gb10 cu134 (2.7.2.post1+gb10.cu134.20261102.0)

Single-arch vllm_flash_attn wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FA2_TUNED_ARCH=12.1a, no PTX, FA2 only), CUDA 13.4.

- Release label: 20261102.0 (UTC date of the tuned commit)
- Built from commit a819c63d085d2d19b341b607693c1164ce80d44d; tested at a819c63d085d2d19b341b607693c1164ce80d44d
- Built and tested against torch 2.15.0+gb10.cu134.20261030.1 (https://github.com/zbrad/pytorch/releases/tag/v2.15.0%2Bgb10.cu134.20261030.1)
- Upstream base: upstream/main merge-base b273a02621d3c94bc747df67204f591b67f38fae
- Wheel sha256: 9949788bfe7ab963a8bb4cbe322a72fa1fc850241f7b9450f7c5f9845538d2cc

## Changes since 2.7.2.post1+gb10.cu134.20261028.0

- Upstream merged up to `b273a02621` (4 commits: #198 SM90 SplitKV block range partitioning, #195 and #200 SM100 MLA
  sparse decode, #203 SM90 hdim 512 forward race fix). This is the commit vllm v0.33.0's `vllm_flash_attn.cmake` pins as
  its flash-attention source. `git diff --stat` of those 4 commits touches only `flash_attn/cute/*` (CuteDSL, Python) and
  `tests/cute/*`; no `csrc/` file changed, so the compiled FA2 extension has no upstream source change in this release.
- The wheel ships the updated `flash_attn/cute` Python files (54 paths with `cute` in the wheel). Nothing in the test
  scope below imports or runs them: FA3 and FA4 are not built for this variant, and the numeric check is FA2 only.
- Rebuilt and tested against torch 2.15.0+gb10.cu134.20261030.1 (upstream v2.15.0 source, same tuned build configuration),
  where the previous release used 20261022.0. The compiled extension links torch's C++ API, so it was rebuilt.
- Not checked: whether vllm v0.33.0 runs with this wheel. The vllm acceptance test has not been run against it.

## Tests

Scope (from the attached test log): gates; _vllm_fa2_C load, arch and PTX checks; tuned/numeric_check.py (flash_attn_varlen_func fa_version=2 vs SDPA: plain, causal, GQA)
Skips by design: upstream's pytest suite (tests/) is not run; FA3 and FA4 are not built for this variant; no multi-GPU tests
Result: === Full test suite summary: gates + numeric check passed ===

## Install

```
pip install --no-deps "vllm_flash_attn @ https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261102.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261102.0-cp314-cp314-linux_aarch64.whl"
```
