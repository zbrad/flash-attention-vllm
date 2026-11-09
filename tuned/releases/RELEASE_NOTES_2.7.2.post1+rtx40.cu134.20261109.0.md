# vllm_flash_attn 2.7.2.post1 — rtx40 cu134 (2.7.2.post1+rtx40.cu134.20261109.0)

Single-arch vllm_flash_attn wheel for RTX 4080/4090 (Ada Lovelace, SM 89) (FA2_TUNED_ARCH=8.9, no PTX, FA2 only), CUDA 13.4.

- Release label: 20261109.0 (UTC date of the tuned commit)
- Built from commit 302383dd1a105f28413102546612667f3ef2c395; tested at 302383dd1a105f28413102546612667f3ef2c395
- Built and tested against torch 2.15.0+rtx40.cu134.20261109.0 (https://github.com/zbrad/pytorch/releases/tag/v2.15.0%2Brtx40.cu134.20261109.0)
- Upstream base: upstream/main merge-base b273a02621d3c94bc747df67204f591b67f38fae
- Wheel sha256: 9d50c519c3761aa3aed4aac1dc96966dc4394842aa79a7ee2b6395f9d6c0b127

The first rtx40 release, from the same upstream base and fork source as 2.7.2.post1+gb10.cu134.20261102.0: between
the two build commits, nothing outside `tuned/` changed except three agent-guidance files under `rules_advisory/`. The
only build change for rtx40 is `tuned/torch.pins`, which pins the rtx40 torch above (without a pin, `build.sh` would fall back to a
PyPI CUDA torch). Built for SM 89 only, FA2 only: FA3 targets Hopper and FA4 is not built for this variant.

Built and tested on one RTX 4090 (24 GiB) under WSL2 (Ubuntu 24.04), Windows driver 621.14. Not run on native Linux
or on an RTX 4080. Not checked: whether vLLM runs with this wheel (vLLM's rtx40 acceptance test has not run yet).

## Tests

Scope (from the attached test log): gates; _vllm_fa2_C load, arch and PTX checks; tuned/numeric_check.py (flash_attn_varlen_func fa_version=2 vs SDPA: plain, causal, GQA)
Skips by design: upstream's pytest suite (tests/) is not run; FA3 and FA4 are not built for this variant; no multi-GPU tests
Result: === Full test suite summary: gates + numeric check passed ===

`_vllm_fa2_C.abi3.so` is the only extension in the wheel, single-arch sm_89 with no PTX. The numeric check (bf16,
batch 2, seqlen 256, 8 heads, head_dim 64; 2 KV heads for GQA) against torch SDPA, tolerance 0.005: plain 0.00098,
causal 0.00391, GQA 0.00049.

The packaging commit is f66d6344f1b9231bb6a5be20d6b8493a4d19d907. It changes only `tuned/smoke_test.sh`, whose
library-identity check now accepts WSL2's libcuda loader stub plus the driver-store library it loads (as in
zbrad/pytorch's smoke test); the first smoke run failed only on that check. The smoke test of the packaged wheel then
passed, including the numeric check against the package.

## Install

```
pip install --no-deps "vllm_flash_attn @ https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Brtx40.cu134.20261109.0/vllm_flash_attn-2.7.2.post1%2Brtx40.cu134.20261109.0-cp314-cp314-linux_x86_64.whl"
```
