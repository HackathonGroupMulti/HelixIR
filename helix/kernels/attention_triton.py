"""
Flash Attention forward via OpenAI Triton (GEMM/attention operator).

This is the Triton counterpart to the Pallas ``flash_attention`` in
``attention.py`` — the same online-softmax algorithm, written in the language
NVIDIA inference teams actually optimize operators in.  Having both lets HelixIR
compare a Pallas and a Triton implementation of the same primitive against the
reference on a given GPU architecture.

Triton is CUDA-only; on a CPU box the public entry point raises a clear
RuntimeError with an install hint, matching the Pallas fallback convention.

The kernel is a standard block-tiled flash-attention v1 forward:
  * grid = (num_q_blocks, batch * heads)
  * each program keeps running (m, l, acc) accumulators over K/V tiles
  * causal masking skips fully-future K blocks and masks the diagonal block
"""
from __future__ import annotations
import math

try:
    import torch
    import triton
    import triton.language as tl
    _TRITON = torch.cuda.is_available()
except Exception:                       # torch/triton missing, or no CUDA
    _TRITON = False


if _TRITON:

    @triton.jit
    def _flash_fwd(
        Q, K, V, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr, CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_zh = tl.program_id(1)
        off_z = off_zh // H
        off_h = off_zh % H

        q_base = Q + off_z * stride_qz + off_h * stride_qh
        k_base = K + off_z * stride_kz + off_h * stride_kh
        v_base = V + off_z * stride_vz + off_h * stride_vh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = q_base + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        n_end = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
        for start_n in range(0, n_end, BLOCK_N):
            n_idx = start_n + offs_n
            k_ptrs = k_base + (n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kk)
            v_ptrs = v_base + (n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk)
            k = tl.load(k_ptrs, mask=n_idx[:, None] < N_CTX, other=0.0)
            v = tl.load(v_ptrs, mask=n_idx[:, None] < N_CTX, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * scale
            qk = tl.where(n_idx[None, :] < N_CTX, qk, float("-inf"))
            if CAUSAL:
                qk = tl.where(offs_m[:, None] >= n_idx[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_base = Out + off_z * stride_oz + off_h * stride_oh
        o_ptrs = o_base + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

    def flash_attention_triton(q, k, v, causal: bool = False,
                               scale: float | None = None, block: int = 64):
        """
        Flash Attention (Triton) forward.

        q, k, v: torch.Tensor [batch, seq, heads, head_dim] on CUDA.
        Returns: torch.Tensor [batch, seq, heads, head_dim].
        """
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        B, S, Hn, D = q.shape
        # [B, S, H, D] -> [B, H, S, D] contiguous
        qc = q.transpose(1, 2).contiguous()
        kc = k.transpose(1, 2).contiguous()
        vc = v.transpose(1, 2).contiguous()
        out = torch.empty_like(qc)

        grid = (triton.cdiv(S, block), B * Hn)
        _flash_fwd[grid](
            qc, kc, vc, out,
            *qc.stride(), *kc.stride(), *vc.stride(), *out.stride(),
            B, Hn, S, scale,
            BLOCK_M=block, BLOCK_N=block, BLOCK_DMODEL=D, CAUSAL=causal,
        )
        return out.transpose(1, 2).contiguous()

else:

    def flash_attention_triton(q, k, v, causal: bool = False,
                               scale: float | None = None, block: int = 64):
        raise RuntimeError(
            "Triton flash attention needs torch + triton on a CUDA GPU. "
            "Install: pip install torch triton  (Linux + NVIDIA GPU)."
        )


def triton_available() -> bool:
    return _TRITON
