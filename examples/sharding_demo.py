"""
Demonstrate the three new HelixIR features:
  1. generate_sharding  — auto-generates JAX device_put code
  2. analyze_backward   — analyzes the VJP computation graph
  3. analyze_full       — compares forward vs backward cost

Run:
    python examples/sharding_demo.py
"""
import jax
import jax.numpy as jnp
import helix

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 8)

B, S, D, F = 4, 256, 512, 2048

x    = jax.random.normal(keys[0], (B, S, D))
wq   = jax.random.normal(keys[1], (D, D))   * 0.02
wk   = jax.random.normal(keys[2], (D, D))   * 0.02
wv   = jax.random.normal(keys[3], (D, D))   * 0.02
wo   = jax.random.normal(keys[4], (D, D))   * 0.02
w1   = jax.random.normal(keys[5], (D, F))   * 0.02
w2   = jax.random.normal(keys[6], (F, D))   * 0.02
norm = jnp.ones(D)


def transformer_step(x, wq, wk, wv, wo, w1, w2, norm):
    B, S, D = x.shape
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    h   = (x / rms) * norm
    q   = jnp.einsum("bsd,dh->bsh", h, wq)
    k   = jnp.einsum("bsd,dh->bsh", h, wk)
    v   = jnp.einsum("bsd,dh->bsh", h, wv)
    s   = jnp.einsum("bqd,bkd->bqk", q, k) * (D**-0.5)
    a   = jax.nn.softmax(s, axis=-1)
    out = jnp.einsum("bqk,bkd->bqd", a, v)
    x   = x + jnp.einsum("bsd,dh->bsh", out, wo)
    ffn = jax.nn.gelu(jnp.einsum("bsd,df->bsf", x, w1))
    return x + jnp.einsum("bsf,fd->bsd", ffn, w2)


args = (x, wq, wk, wv, wo, w1, w2, norm)
arg_names = ["x", "wq", "wk", "wv", "wo", "w1", "w2", "norm"]

# ── 1. Auto-sharding code generator ─────────────────────────────────────────
print("\n" + "═"*60)
print("  1. Auto-Sharding Code Generator")
print("═"*60)

plan = helix.generate_sharding(
    transformer_step, *args,
    mesh_shape=(2, 4),
    axis_names=("batch", "model"),
    arg_names=arg_names,
)
print(plan)           # human-readable table
print("\n--- Generated code ---")
print(plan.code)

# ── 2. Backward pass analysis ────────────────────────────────────────────────
print("\n" + "═"*60)
print("  2. Backward Pass Analysis")
print("═"*60)

bwd = helix.analyze_backward(transformer_step, *args)
print(f"  Backward ops  : {bwd['num_ops']}")
print(f"  Backward GFLOPs: {bwd['total_flops']/1e9:.2f}")
print(f"  Backward MB   : {bwd['total_bytes']/1e6:.1f}")

# Top checkpoint candidate in backward
ckpt_recs = bwd["passes"][1].recommendations
if ckpt_recs:
    top = ckpt_recs[0]
    print(f"\n  Largest backward activation: {top['size_mb']:.1f} MB  {top['shape']}")
    print(f"  → {top['message']}")

# ── 3. Forward vs backward comparison ───────────────────────────────────────
print("\n" + "═"*60)
print("  3. Forward vs Backward")
print("═"*60)

full = helix.analyze_full(transformer_step, *args)
helix.print_full_report(full, fn_name="transformer_step")
