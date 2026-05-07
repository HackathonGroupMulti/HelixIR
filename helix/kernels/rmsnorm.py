"""
RMSNorm kernel — reference implementation + Pallas GPU kernel.

The Pallas version fuses the variance computation, normalization, and weight
scaling into a single GPU kernel, eliminating two extra memory round-trips
that the naive JAX version incurs (one for the squared mean, one for the
division).  The custom VJP ensures correct gradients through both paths.
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
# Reference (unfused, works on CPU/GPU, used for gradient checking)
# ---------------------------------------------------------------------------

def rmsnorm_ref(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Unfused RMSNorm.  x: [..., hidden], w: [hidden]."""
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(var + eps)
    return x_norm * w


# ---------------------------------------------------------------------------
# Custom VJP — analytically correct gradient for both paths
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(2,))
def rmsnorm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm with an efficient fused backward pass."""
    return rmsnorm_ref(x, w, eps)


def _rmsnorm_fwd(x: jax.Array, w: jax.Array, eps: float):
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    rrms = jax.lax.rsqrt(var + eps)
    x_norm = x * rrms
    y = x_norm * w
    return y, (x, w, rrms, x_norm)


def _rmsnorm_bwd(eps: float, res: tuple, g: jax.Array):
    x, w, rrms, x_norm = res
    hidden = x.shape[-1]

    gw = jnp.sum(g * x_norm, axis=tuple(range(x.ndim - 1)))  # [hidden]

    gx_norm = g * w                                           # [..., hidden]
    gx = gx_norm * rrms
    gx -= x * (jnp.sum(gx_norm * x_norm, axis=-1, keepdims=True) / hidden) * (rrms ** 2)

    return gx, gw


rmsnorm.defvjp(_rmsnorm_fwd, _rmsnorm_bwd)


# ---------------------------------------------------------------------------
# Pallas kernel (GPU only, requires jax[cuda])
# ---------------------------------------------------------------------------

if _PALLAS:
    def _rmsnorm_kernel(x_ref, w_ref, out_ref, *, eps: float):
        """Pallas kernel: one block per row of x_flat."""
        row = x_ref[...]          # [hidden]
        w   = w_ref[...]          # [hidden]
        ms  = jnp.mean(row * row)
        rrms = jax.lax.rsqrt(ms + eps)
        out_ref[...] = row * rrms * w

    def rmsnorm_pallas(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
        """
        Fused RMSNorm via Pallas.  Launches one kernel block per row so that
        the mean, rsqrt, and scale are computed in a single pass over HBM.

        Requires: JAX with CUDA backend and jax.experimental.pallas available.
        x shape: [*, hidden]
        w shape: [hidden]
        """
        hidden = x.shape[-1]
        batch_dims = x.shape[:-1]
        n_rows = 1
        for d in batch_dims:
            n_rows *= d

        x_flat = x.reshape(n_rows, hidden)

        kernel = partial(_rmsnorm_kernel, eps=eps)

        out_flat = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
            grid=(n_rows,),
            in_specs=[
                pl.BlockSpec(index_map=lambda i: (i, 0), block_shape=(1, hidden)),
                pl.BlockSpec(index_map=lambda i: (0,),   block_shape=(hidden,)),
            ],
            out_specs=pl.BlockSpec(index_map=lambda i: (i, 0), block_shape=(1, hidden)),
        )(x_flat, w)

        return out_flat.reshape(x.shape)

else:
    def rmsnorm_pallas(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
        raise RuntimeError(
            "Pallas is not available.  Install JAX with CUDA: pip install jax[cuda12]"
        )
