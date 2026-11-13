# vllm_flash_attn 2.7.2.post1 — gb10 cu134 (2.7.2.post1+gb10.cu134.20261113.0)

Single-arch vllm_flash_attn wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (FA2_TUNED_ARCH=12.1a, no PTX, FA2 only), CUDA 13.4.

- Release label: 20261113.0 (UTC date of the tuned commit)
- Built from commit 90da91a72ab2db77aeb12d63ee0bacd1456abc11; tested at 90da91a72ab2db77aeb12d63ee0bacd1456abc11
- Built and tested against torch 2.15.0+gb10.cu134.20261112.0 (https://github.com/zbrad/pytorch/releases/tag/v2.15.0%2Bgb10.cu134.20261112.0)
- Upstream base: upstream/main merge-base b273a02621d3c94bc747df67204f591b67f38fae (unchanged from 20261102.0)
- Wheel sha256: e82f1c1d46aab1270d87d85433ffc1036f3258fedf71b1ac8abaa36cf6fb11d4

## Changes since 2.7.2.post1+gb10.cu134.20261102.0

- Rebuilt against torch 2.15.0+gb10.cu134.20261112.0 (was 20261030.1). Same upstream v2.15.0 source; see that torch
  release's notes for what changed in its build. The compiled extension links torch's C++ API, so it was rebuilt.
- No source change to the extension: the other commits since 20261102.0 are rtx40 pins and notes, the tuned-common
  sync and smoke-test changes.

## Tests

Scope (from the attached test log): gates; _vllm_fa2_C load, arch and PTX checks; tuned/numeric_check.py (flash_attn_varlen_func fa_version=2 vs SDPA: plain, causal, GQA)
Skips by design: upstream's pytest suite (tests/) is not run; FA3 and FA4 are not built for this variant; no multi-GPU tests
Result: === Full test suite summary: gates + numeric check passed ===

Smoke test of the packaged wheel passed (2026-11-13T13:35:16Z): build-info stamp, library identity, numeric check against the package, clean-venv install with the pinned torch.

The model-level check is vllm's acceptance test, run in that repo against this wheel.

## Install

```
pip install --no-deps "vllm_flash_attn @ https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261113.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261113.0-cp314-cp314-linux_aarch64.whl"
```
