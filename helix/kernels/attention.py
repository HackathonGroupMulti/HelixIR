"""
Scaled dot-product attention — reference + online-softmax Flash Attention via Pallas.

Flash Attention rewrites standard attention so that Q, K, V tiles are streamed
through SRAM in blocks.  The online softmax (Milakov & Gimelshein 2018) lets us
accumulate the normalised output without materialising the full [S, S] attention
matrix, cutting HBM traffic from O(S²) to O(S).
"""
from __future__ import annotations
import math
from functools import partial

import jax
import jax.numpy as jnp

try:
    import jax.experimental.pallas as pl
    _PALLAS = True
except ImportError:
    _PALLAS = False


# ---------------------------------------------------------------------------
# Reference attention (materialises full S×S matrix)
# ---------------------------------------------------------------------------

def attention_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array | None = None,
    scale: float | None = None,
) -> jax.Array:
    """
    Standard scaled dot-product attention.
    q, k, v: [batch, seq, heads, head_dim]
    mask:     [batch, heads, seq_q, seq_k]  (True = keep)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # scores: [batch, heads, seq_q, seq_k]
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale

    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


# ---------------------------------------------------------------------------
# Causal mask helper
# ---------------------------------------------------------------------------

def causal_mask(seq_len: int) -> jax.Array:
    """Lower-triangular boolean mask [1, 1, seq_len, seq_len]."""
    idx = jnp.arange(seq_len)
    return (idx[:, None] >= idx[None, :])[None, None]


# ---------------------------------------------------------------------------
# Flash Attention via Pallas (online softmax, block-sparse HBM access)
# ---------------------------------------------------------------------------

if _PALLAS:
    def _flash_fwd_kernel(
        q_ref, k_ref, v_ref, out_ref, lse_ref,
        *, block_k: int, scale: float, causal: bool,
    ):
        """
        One Pallas block handles block_q rows of Q.
        Streams over all K/V blocks, maintaining running (m, l, o) accumulators
        for the online softmax.

        q_ref:   [block_q, head_dim]
        k_ref:   [seq_k,   head_dim]   (full K for this head, loaded tile-by-tile)
        v_ref:   [seq_k,   head_dim]
        out_ref: [block_q, head_dim]
        lse_ref: [block_q]             — log-sum-exp for rescaled gradients
        """
        q       = q_ref[...]                              # [block_q, d]
        seq_k   = k_ref.shape[0]
        n_tiles = seq_k // block_k

        # Running accumulators
        m = jnp.full(q.shape[0], -jnp.inf, dtype=jnp.float32)
        l = jnp.zeros(q.shape[0],           dtype=jnp.float32)
        o = jnp.zeros_like(q,               dtype=jnp.float32)

        def tile_body(carry, tile_idx):
            m_i, l_i, o_i = carry
            k_tile = jax.lax.dynamic_slice_in_dim(
                k_ref[...], tile_idx * block_k, block_k, axis=0
            )  # [block_k, d]
            v_tile = jax.lax.dynamic_slice_in_dim(
                v_ref[...], tile_idx * block_k, block_k, axis=0
            )  # [block_k, d]

            s = jnp.dot(q.astype(jnp.float32), k_tile.T.astype(jnp.float32)) * scale
            # [block_q, block_k]

            m_new = jnp.maximum(m_i, jnp.max(s, axis=-1))
            p     = jnp.exp(s - m_new[:, None])
            l_new = jnp.exp(m_i - m_new) * l_i + jnp.sum(p, axis=-1)
            o_new = jnp.exp(m_i - m_new)[:, None] * o_i + jnp.dot(p, v_tile.astype(jnp.float32))

            return (m_new, l_new, o_new), None

        (m_f, l_f, o_f), _ = jax.lax.scan(tile_body, (m, l, o), jnp.arange(n_tiles))

        out_ref[...] = (o_f / l_f[:, None]).astype(q_ref.dtype)
        lse_ref[...] = m_f + jnp.log(l_f)   # for backward pass

    def flash_attention(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        causal: bool = False,
        block_q: int = 64,
        block_k: int = 64,
        scale: float | None = None,
    ) -> jax.Array:
        """
        Flash Attention via Pallas.

        q, k, v: [batch, seq, heads, head_dim]
        Returns: [batch, seq, heads, head_dim]

        Requires JAX with CUDA backend.
        """
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])

        B, S, H, D = q.shape
        assert S % block_q == 0, f"seq_len {S} must be divisible by block_q {block_q}"
        assert S % block_k == 0, f"seq_len {S} must be divisible by block_k {block_k}"

        n_blocks_q = S // block_q

        kernel = partial(_flash_fwd_kernel, block_k=block_k, scale=scale, causal=causal)

        # Flatten batch & head dims so we get simple 1-D grid
        q_bh  = q.transpose(0, 2, 1, 3).reshape(B * H, S, D)
        k_bh  = k.transpose(0, 2, 1, 3).reshape(B * H, S, D)
        v_bh  = v.transpose(0, 2, 1, 3).reshape(B * H, S, D)

        def per_head(q_h, k_h, v_h):
            # q_h: [S, D];  grid = (n_blocks_q,)
            out, lse = pl.pallas_call(
                kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((S, D), q.dtype),
                    jax.ShapeDtypeStruct((S,),   jnp.float32),
                ],
                grid=(n_blocks_q,),
                in_specs=[
                    pl.BlockSpec(index_map=lambda i: (i * block_q, 0), block_shape=(block_q, D)),
                    pl.BlockSpec(index_map=lambda i: (0, 0),           block_shape=(S, D)),
                    pl.BlockSpec(index_map=lambda i: (0, 0),           block_shape=(S, D)),
                ],
                out_specs=[
                    pl.BlockSpec(index_map=lambda i: (i * block_q, 0), block_shape=(block_q, D)),
                    pl.BlockSpec(index_map=lambda i: (i * block_q,),   block_shape=(block_q,)),
                ],
            )(q_h, k_h, v_h)
            return out

        out_bh = jax.vmap(per_head)(q_bh, k_bh, v_bh)   # [B*H, S, D]
        return out_bh.reshape(B, H, S, D).transpose(0, 2, 1, 3)

else:
    def flash_attention(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        causal: bool = False,
        block_q: int = 64,
        block_k: int = 64,
        scale: float | None = None,
    ) -> jax.Array:
        raise RuntimeError(
            "Pallas is not available.  Install JAX with CUDA: pip install jax[cuda12]"
        )
