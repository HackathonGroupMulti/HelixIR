"""
HelixIR profile demo.

Run:
    python examples/profile_demo.py

This script defines a small GPT-block-like function, decorates it with
@helix.profile, and runs it once to trigger analysis.

Set HELIX_ARGS so that `helix profile examples/profile_demo.py` also works.
"""
import jax
import jax.numpy as jnp
import helix

# ---------------------------------------------------------------------------
# A realistic-looking model forward pass
# ---------------------------------------------------------------------------

@helix.profile(num_devices=8)
def transformer_step(x, wq, wk, wv, wo, w1, w2, norm_w):
    B, S, D = x.shape
    H = 8
    head_dim = D // H

    # RMSNorm
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    h = (x / rms) * norm_w

    # QKV
    q = jnp.einsum("bsd,dh->bsh", h, wq).reshape(B, S, H, head_dim)
    k = jnp.einsum("bsd,dh->bsh", h, wk).reshape(B, S, H, head_dim)
    v = jnp.einsum("bsd,dh->bsh", h, wv).reshape(B, S, H, head_dim)

    # Attention (naive, no masking)
    scale = head_dim ** -0.5
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    attn = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(B, S, D)

    # Output + FFN
    x = x + jnp.einsum("bsd,dh->bsh", attn, wo)
    ffn = jax.nn.gelu(jnp.einsum("bsd,df->bsf", x, w1))
    x = x + jnp.einsum("bsf,fd->bsd", ffn, w2)
    return x


# ---------------------------------------------------------------------------
# Example inputs (also used by `helix profile examples/profile_demo.py`)
# ---------------------------------------------------------------------------

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 8)
B, S, D = 2, 256, 512
F = D * 4

HELIX_ARGS = (
    jax.random.normal(keys[0], (B, S, D)),
    jax.random.normal(keys[1], (D, D)) * 0.02,
    jax.random.normal(keys[2], (D, D)) * 0.02,
    jax.random.normal(keys[3], (D, D)) * 0.02,
    jax.random.normal(keys[4], (D, D)) * 0.02,
    jax.random.normal(keys[5], (D, F)) * 0.02,
    jax.random.normal(keys[6], (F, D)) * 0.02,
    jnp.ones(D),
)

if __name__ == "__main__":
    out = transformer_step(*HELIX_ARGS)
    print(f"\nOutput shape: {out.shape}  dtype: {out.dtype}")

    # Also benchmark it
    result = helix.benchmark(
        jax.jit(lambda: transformer_step(*HELIX_ARGS)),
        name="transformer_step (jit)",
        warmup=5,
        iters=50,
    )
    print(result)
