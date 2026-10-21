"""Numeric-correctness check of the installed vllm_flash_attn (FA2 path).

Compares flash_attn_varlen_func(fa_version=2) with
torch.nn.functional.scaled_dot_product_attention for the plain, causal and GQA
cases, on a batch of equal-length sequences packed the way varlen takes them.
Run by tuned/full_test.sh (the unstamped wheel) and tuned/smoke_test.sh (the
packaged wheel), from an empty directory with ``python -I``, so the
vllm_flash_attn it imports is the installed one. Exits 1 if any case is out of
tolerance, or if the FA2 extension is not usable on the device.
"""

import sys

from torch import Tensor, arange, bfloat16, int32, manual_seed, randn
from torch.nn.functional import scaled_dot_product_attention
from vllm_flash_attn import (
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    is_fa_version_supported,
)

# Batch=2, seqlen=256, nheads=8, headdim=64 (nheads_kv=2 for the GQA case).
BATCH = 2
SEQLEN = 256
NHEADS = 8
HEADDIM = 64
# Normal bf16 kernel-order noise is ~1e-3 in this shape; a real regression should
# blow well past this.
TOLERANCE = 5e-3
DEVICE = "cuda"
FA_VERSION = 2


class NumericCheck:
    """Run flash_attn_varlen_func against SDPA and report the largest difference."""

    @staticmethod
    def _sdpa_ref(
        q: Tensor, k: Tensor, v: Tensor, causal: bool, nheads_kv: int
    ) -> Tensor:
        """Reference attention: SDPA with the KV heads repeated up to NHEADS.

        q, k and v are packed (BATCH * SEQLEN, heads, HEADDIM); SDPA wants
        (BATCH, heads, SEQLEN, HEADDIM). The result is packed again.
        """
        qt = q.view(BATCH, SEQLEN, NHEADS, HEADDIM).transpose(1, 2)
        kt = k.view(BATCH, SEQLEN, nheads_kv, HEADDIM).transpose(1, 2)
        vt = v.view(BATCH, SEQLEN, nheads_kv, HEADDIM).transpose(1, 2)
        if nheads_kv != NHEADS:
            rep = NHEADS // nheads_kv
            kt = kt.repeat_interleave(rep, dim=1)
            vt = vt.repeat_interleave(rep, dim=1)
        out = scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
        return out.transpose(1, 2).reshape(BATCH * SEQLEN, NHEADS, HEADDIM)

    @staticmethod
    def run_case(name: str, causal: bool, nheads_kv: int) -> bool:
        """Compare one case.

        Args:
            name: label printed with the result.
            causal: whether to apply a causal mask.
            nheads_kv: number of KV heads (NHEADS for plain, fewer for GQA).

        Returns:
            True if the largest absolute difference is within TOLERANCE.
        """
        total = BATCH * SEQLEN
        q = randn(total, NHEADS, HEADDIM, device=DEVICE, dtype=bfloat16)
        k = randn(total, nheads_kv, HEADDIM, device=DEVICE, dtype=bfloat16)
        v = randn(total, nheads_kv, HEADDIM, device=DEVICE, dtype=bfloat16)
        cu_seqlens = arange(0, total + 1, SEQLEN, device=DEVICE, dtype=int32)
        fa_out = flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=SEQLEN,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=SEQLEN,
            cu_seqlens_k=cu_seqlens,
            causal=causal,
            fa_version=FA_VERSION,
        )
        ref_out = NumericCheck._sdpa_ref(q, k, v, causal, nheads_kv)
        max_diff = (fa_out - ref_out).abs().max().item()
        ok = max_diff < TOLERANCE
        print(f"  {name:10s} max_abs_diff={max_diff:.5f}  [{'OK' if ok else 'FAIL'}]")
        return ok

    @classmethod
    def main(cls) -> int:
        """Run all three cases; return the process exit status."""
        if not is_fa_version_supported(FA_VERSION):
            print(
                f"FAIL: FA{FA_VERSION} is not usable here: "
                f"{fa_version_unsupported_reason(FA_VERSION)}",
                file=sys.stderr,
            )
            return 1
        manual_seed(0)
        results = [
            cls.run_case("plain", causal=False, nheads_kv=NHEADS),
            cls.run_case("causal", causal=True, nheads_kv=NHEADS),
            cls.run_case("gqa", causal=False, nheads_kv=2),
        ]
        if not all(results):
            print(
                f"\nFAIL: one or more cases exceeded tolerance ({TOLERANCE}).",
                file=sys.stderr,
            )
            return 1
        print(f"\nAll cases within tolerance ({TOLERANCE}).")
        return 0


if __name__ == "__main__":
    sys.exit(NumericCheck.main())
