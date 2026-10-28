# tuned/ — single-arch vllm_flash_attn wheels

Builds the `vllm_flash_attn` wheel from vLLM's flash-attention fork for one GPU variant and one CUDA toolkit. It is not
the standalone `flash_attn` package (see zbrad/flash-attention). The layout and rules are `docs/ARCHITECTURE.md` in
[zbrad/tuned-common](https://github.com/zbrad/tuned-common). Only gb10 is built from this repo today.

```
CUDA_TAG=cu134 bash tuned/build.sh gb10        # build venv + pinned torch + unstamped wheel -> tuned/build/cu134/wheel/
CUDA_TAG=cu134 bash tuned/full_test.sh gb10    # tests + numeric check          -> .../releases/TEST_RESULTS_gb10.log
CUDA_TAG=cu134 bash tuned/package.sh gb10      # stamp, sidecars, DRAFT notes   -> .../dist/
CUDA_TAG=cu134 bash tuned/smoke_test.sh gb10   # the packaged bits run          -> .../releases/SMOKE_RESULTS_gb10.log
# review and commit tuned/releases/RELEASE_NOTES_<version>.md
CUDA_TAG=cu134 bash tuned/release.sh gb10      # gates only; writes the ready-marker, never publishes
CUDA_TAG=cu134 bash tuned/publish.sh gb10      # publishes exactly what release.sh saw
```

| File | Purpose |
|---|---|
| `env.sh` | device, toolkit and tree setup; sourced by every script |
| `repo.env` | fork-only settings: upstream ref, the arch spelling the build takes (`GPU_TUNED_FA2_ARCH`, `12.1a` for gb10) |
| `torch.pins` | the zbrad/pytorch wheel this package is built and tested against, per variant and toolkit |
| `common.sh`, `sync-common.sh`, `common-data/` | vendored from tuned-common; refresh with `bash tuned/sync-common.sh sync` |
| `build.sh`, `wheel.sh`, `full_test.sh`, `package.sh`, `smoke_test.sh`, `release.sh`, `publish.sh` | the stages above |
| `numeric_check.py` | numeric comparison used by the test stage |
| `compliance.conf` | attestations for the checks `tools/tuned_compliance.sh` cannot make mechanically |

## Torch pin

The compiled `_vllm_fa*_C` extensions link torch's C++ API, so the torch wheel is part of the build's identity.
`build.sh` downloads the wheel named in `torch.pins` into `tuned/build/<cuda-tag>/wheels/`, installs exactly that, and
refuses to compile if the installed torch is any other version. A new zbrad/pytorch release means editing `torch.pins`
and rebuilding.

## Build venv and gates

- The build venv is `tuned/build/<cuda-tag>/venv`, made by `gpu_tuned_setup_build_venv`. `GPU_TUNED_CLEAN_VENV=1`
  removes and recreates it. The current setuptools (at least 84.2) is installed last (C8).
- `package.sh` fails if the venv's setuptools is below the minimum (H10) and runs the dependency audit on the wheel (H9),
  logged to `tuned/build/<cuda-tag>/dependency-audit.log`.
- Everything under `tuned/build/<cuda-tag>/` is gitignored.
