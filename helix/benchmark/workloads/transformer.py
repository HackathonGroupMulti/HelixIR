"""
GPT-style transformer block workload.

This is the canonical benchmark target — a single decoder block covering:
  QKV projection → RoPE → attention → output projection → RMSNorm → FFN (SwiGLU)

All weights are randomly initialised; only forward pass is benchmarked by
default.  Use jax.grad(lambda p: ...) for backward-pass timing.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

from ...kernels.rmsnorm import rmsnorm_ref
from ...kernels.rope import apply_rope
from ...kernels.attention import attention_ref


def gpt_block(
    x: jax.Array,
    wq: jax.Array, wk: jax.Array, wv: jax.Array, wo: jax.Array,
    w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array,
    norm_attn: jax.Array, norm_ffn: jax.Array,
    num_heads: int,
) -> jax.Array:
    """
    Single GPT decoder block (LLaMA-style):
      RMSNorm → QKV → RoPE → Attention → Out-proj → Residual
      RMSNorm → SwiGLU FFN → Residual
    """
    B, S, D = x.shape
    head_dim = D // num_heads

    # --- Self-attention sublayer ---
    h = rmsnorm_ref(x, norm_attn)
    q = jnp.einsum("bsd,dh->bsh", h, wq).reshape(B, S, num_heads, head_dim)
    k = jnp.einsum("bsd,dh->bsh", h, wk).reshape(B, S, num_heads, head_dim)
    v = jnp.einsum("bsd,dh->bsh", h, wv).reshape(B, S, num_heads, head_dim)

    q = apply_rope(q)
    k = apply_rope(k)

    attn = attention_ref(q, k, v)                             # [B, S, H, head_dim]
    attn = attn.reshape(B, S, D)
    x = x + jnp.einsum("bsd,dh->bsh", attn, wo)

    # --- FFN sublayer (SwiGLU) ---
    h = rmsnorm_ref(x, norm_ffn)
    gate = jax.nn.silu(jnp.einsum("bsd,df->bsf", h, w_gate))
    up   = jnp.einsum("bsd,df->bsf", h, w_up)
    x = x + jnp.einsum("bsf,fd->bsd", gate * up, w_down)

    return x


def make_inputs(
    batch: int = 2,
    seq_len: int = 512,
    d_model: int = 1024,
    num_heads: int = 16,
    ffn_mult: int = 4,
    key: jax.Array | None = None,
) -> dict:
    """Randomly initialise all inputs/weights for one GPT block."""
    if key is None:
        key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 12)

    D = d_model
    F = d_model * ffn_mult
    std = 0.02

    return dict(
        x        = jax.random.normal(keys[0], (batch, seq_len, D), dtype=jnp.float32),
        wq       = jax.random.normal(keys[1], (D, D), dtype=jnp.float32) * std,
        wk       = jax.random.normal(keys[2], (D, D), dtype=jnp.float32) * std,
        wv       = jax.random.normal(keys[3], (D, D), dtype=jnp.float32) * std,
        wo       = jax.random.normal(keys[4], (D, D), dtype=jnp.float32) * std,
        w_gate   = jax.random.normal(keys[5], (D, F), dtype=jnp.float32) * std,
        w_up     = jax.random.normal(keys[6], (D, F), dtype=jnp.float32) * std,
        w_down   = jax.random.normal(keys[7], (F, D), dtype=jnp.float32) * std,
        norm_attn= jnp.ones(D, dtype=jnp.float32),
        norm_ffn = jnp.ones(D, dtype=jnp.float32),
        num_heads= num_heads,
    )


def flop_count(batch: int, seq_len: int, d_model: int, ffn_mult: int = 4) -> int:
    """Approximate FLOPs for one forward pass through a GPT block."""
    D = d_model
    F = d_model * ffn_mult
    S = seq_len
    B = batch

    qkv_proj = 3 * 2 * B * S * D * D
    attn     = 2 * B * S * S * D          # QK^T + AV
    out_proj = 2 * B * S * D * D
    ffn      = 2 * (2 * B * S * D * F)   # gate+up then down (×2 for SwiGLU gate)
    norms    = 2 * 2 * B * S * D          # two rmsnorm passes

    return qkv_proj + attn + out_proj + ffn + norms
