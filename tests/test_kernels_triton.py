"""
Tests for the Triton flash-attention kernel.

The kernel is CUDA-only.  On a CPU box we can only assert the graceful-fallback
contract; the numerical-correctness test is skipped unless torch+triton+CUDA are
present, in which case it checks the Triton output against the reference.
"""
import math
import pytest

from helix.kernels.attention_triton import flash_attention_triton, triton_available


class TestTritonFallback:
    def test_available_is_bool(self):
        assert isinstance(triton_available(), bool)

    @pytest.mark.skipif(triton_available(), reason="Triton present — fallback not active")
    def test_raises_without_cuda(self):
        with pytest.raises(RuntimeError):
            flash_attention_triton(None, None, None)


@pytest.mark.skipif(not triton_available(), reason="requires torch+triton on CUDA")
class TestTritonNumerics:
    def _inputs(self):
        import torch
        torch.manual_seed(0)
        B, S, H, D = 2, 128, 4, 64
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        return q, k, v

    def _ref(self, q, k, v, causal):
        import torch
        scale = 1.0 / math.sqrt(q.shape[-1])
        s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
        if causal:
            S = q.shape[1]
            mask = torch.tril(torch.ones(S, S, device=q.device, dtype=torch.bool))
            s = s.masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", w, v.float())

    @pytest.mark.parametrize("causal", [False, True])
    def test_matches_reference(self, causal):
        import torch
        q, k, v = self._inputs()
        out = flash_attention_triton(q, k, v, causal=causal).float()
        ref = self._ref(q, k, v, causal)
        assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
