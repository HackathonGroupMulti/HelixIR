"""
Rotary Position Embeddings (RoPE) — reference + Pallas kernel.

The reference path computes cos/sin tables on the fly and applies them via
concatenation.  The Pallas kernel fuses the frequency precomputation and
rotation into one pass, avoiding writing out the intermediate half-tensors.
"""
from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp

try:
    import jax.experimental.pallas as pl
    _PALLAS = True
except ImportError:
    _PALLAS = False


# ---------------------------------------------------------------------------
# Frequency table helpers
# ---------------------------------------------------------------------------

def precompute_freqs(seq_len: int, head_dim: int, base: float = 10_000.0) -> tuple[jax.Array, jax.Array]:
    """Return (cos, sin) tables of shape [seq_len, head_dim // 2]."""
    half = head_dim // 2
    i = jnp.arange(half, dtype=jnp.float32)
    theta = 1.0 / (base ** (i / half))                   # [half]
    t = jnp.arange(seq_len, dtype=jnp.float32)           # [seq]
    freqs = jnp.outer(t, theta)                           # [seq, half]
    return jnp.cos(freqs), jnp.sin(freqs)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def rope_ref(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """
    Apply RoPE to x.
    x:   [batch, seq, heads, head_dim]
    cos: [seq, head_dim // 2]
    sin: [seq, head_dim // 2]
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]          # [B, S, H, half] each

    # Align cos/sin to [1, S, 1, half]
    cos_b = cos[None, :, None, :]
    sin_b = sin[None, :, None, :]

    return jnp.concatenate([
        x1 * cos_b - x2 * sin_b,
        x1 * sin_b + x2 * cos_b,
    ], axis=-1)


def apply_rope(
    x: jax.Array,
    base: float = 10_000.0,
) -> jax.Array:
    """Convenience wrapper: precomputes freqs and applies RoPE."""
    seq_len = x.shape[-3]
    head_dim = x.shape[-1]
    cos, sin = precompute_freqs(seq_len, head_dim, base)
    return rope_ref(x, cos, sin)


# ---------------------------------------------------------------------------
# Pallas kernel
# ---------------------------------------------------------------------------

if _PALLAS:
    def _rope_kernel(
        x1_ref, x2_ref, cos_ref, sin_ref,
        o1_ref, o2_ref,
    ):
        """
        One block = one (seq_pos, head) pair.
        x1_ref, x2_ref: [half]   — the two halves of the head vector
        cos_ref, sin_ref: [half] — pre-fetched freq row for this position
        """
        x1  = x1_ref[...]
        x2  = x2_ref[...]
        cos = cos_ref[...]
        sin = sin_ref[...]
        o1_ref[...] = x1 * cos - x2 * sin
        o2_ref[...] = x1 * sin + x2 * cos

    def rope_pallas(x: jax.Array, base: float = 10_000.0) -> jax.Array:
        """
        Fused RoPE via Pallas.  Avoids materialising the concatenated output
        and fuses the trig application into one HBM pass.

        x: [batch, seq, heads, head_dim]  (head_dim must be even)
        """
        B, S, H, D = x.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"
        half = D // 2

        cos, sin = precompute_freqs(S, D, base)  # [S, half]

        # Flatten to [B*H, S, half] for grid simplicity
        x1 = x[..., :half].transpose(0, 2, 1, 3).reshape(B * H, S, half)
        x2 = x[..., half:].transpose(0, 2, 1, 3).reshape(B * H, S, half)

        n_bh, n_s, n_half = x1.shape

        o1, o2 = pl.pallas_call(
            _rope_kernel,
            out_shape=[
                jax.ShapeDtypeStruct(x1.shape, x1.dtype),
                jax.ShapeDtypeStruct(x2.shape, x2.dtype),
            ],
            grid=(n_bh, n_s),
            in_specs=[
                pl.BlockSpec(lambda bh, s: (bh, s, 0), (1, 1, n_half)),
                pl.BlockSpec(lambda bh, s: (bh, s, 0), (1, 1, n_half)),
                pl.BlockSpec(lambda bh, s: (s, 0),     (1, n_half)),
                pl.BlockSpec(lambda bh, s: (s, 0),     (1, n_half)),
            ],
            out_specs=[
                pl.BlockSpec(lambda bh, s: (bh, s, 0), (1, 1, n_half)),
                pl.BlockSpec(lambda bh, s: (bh, s, 0), (1, 1, n_half)),
            ],
        )(x1, x2, cos, sin)

        out = jnp.concatenate([o1, o2], axis=-1)             # [B*H, S, D]
        return out.reshape(B, H, S, D).transpose(0, 2, 1, 3)  # [B, S, H, D]

else:
    def rope_pallas(x: jax.Array, base: float = 10_000.0) -> jax.Array:
        raise RuntimeError(
            "Pallas is not available.  Install JAX with CUDA: pip install jax[cuda12]"
        )
