"""
Diffusion UNet residual block workload.

Models a single ResNet-style block from a latent diffusion U-Net:
  GroupNorm → SiLU → Conv → add time embedding → GroupNorm → SiLU → Conv → residual

Convolutions are approximated as einsum contractions (no spatial kernel for
simplicity; swap in jax.lax.conv_general_dilated for full fidelity).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def _group_norm(x: jax.Array, w: jax.Array, b: jax.Array, num_groups: int = 32) -> jax.Array:
    """Approximate group norm via layer norm over the channel axis."""
    # x: [B, H, W, C] — normalise over C, approximate groups with LN
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    x_n  = (x - mean) / jnp.sqrt(var + 1e-5)
    return x_n * w + b


def unet_res_block(
    x: jax.Array,
    w1: jax.Array, w2: jax.Array,
    norm_w1: jax.Array, norm_b1: jax.Array,
    norm_w2: jax.Array, norm_b2: jax.Array,
    time_w: jax.Array, time_b: jax.Array,
    time_emb: jax.Array,
    skip_w: jax.Array | None = None,
) -> jax.Array:
    """
    UNet residual block.
    x:        [B, H, W, C]
    time_emb: [B, time_dim]
    w1, w2:   [C, C]   (1×1 conv approximation)
    skip_w:   [C, C]   optional skip projection when in_channels ≠ out_channels
    """
    # Branch 1
    h = _group_norm(x, norm_w1, norm_b1)
    h = jax.nn.silu(h)
    h = jnp.einsum("bhwc,cd->bhwd", h, w1)

    # Add time conditioning
    t = jax.nn.silu(jnp.einsum("bt,td->bd", time_emb, time_w) + time_b)
    h = h + t[:, None, None, :]

    # Branch 2
    h = _group_norm(h, norm_w2, norm_b2)
    h = jax.nn.silu(h)
    h = jnp.einsum("bhwc,cd->bhwd", h, w2)

    # Residual
    skip = x if skip_w is None else jnp.einsum("bhwc,cd->bhwd", x, skip_w)
    return h + skip


def make_inputs(
    batch: int = 2,
    height: int = 32,
    width: int = 32,
    channels: int = 256,
    time_dim: int = 1024,
    key: jax.Array | None = None,
) -> dict:
    if key is None:
        key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, 10)
    C = channels
    T = time_dim
    std = 0.02

    return dict(
        x       = jax.random.normal(keys[0], (batch, height, width, C), dtype=jnp.float32),
        w1      = jax.random.normal(keys[1], (C, C), dtype=jnp.float32) * std,
        w2      = jax.random.normal(keys[2], (C, C), dtype=jnp.float32) * std,
        norm_w1 = jnp.ones(C,  dtype=jnp.float32),
        norm_b1 = jnp.zeros(C, dtype=jnp.float32),
        norm_w2 = jnp.ones(C,  dtype=jnp.float32),
        norm_b2 = jnp.zeros(C, dtype=jnp.float32),
        time_w  = jax.random.normal(keys[3], (T, C), dtype=jnp.float32) * std,
        time_b  = jnp.zeros(C, dtype=jnp.float32),
        time_emb= jax.random.normal(keys[4], (batch, T), dtype=jnp.float32),
        skip_w  = None,
    )


def flop_count(batch: int, height: int, width: int, channels: int, time_dim: int) -> int:
    B, H, W, C, T = batch, height, width, channels, time_dim
    conv1   = 2 * B * H * W * C * C
    conv2   = 2 * B * H * W * C * C
    time_proj = 2 * B * T * C
    norms   = 4 * B * H * W * C     # two group norms
    return conv1 + conv2 + time_proj + norms
