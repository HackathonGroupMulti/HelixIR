"""
Compare reference vs Pallas kernel implementations side-by-side.

Run:
    python examples/kernel_compare.py

On CPU: Pallas kernels raise RuntimeError (GPU required).
On GPU: timings prove the fused kernels win on memory-bandwidth-bound ops.
"""
import jax
import jax.numpy as jnp
import helix
from helix.kernels.rmsnorm import rmsnorm_ref, rmsnorm_pallas
from helix.kernels.rope import apply_rope, rope_pallas
from helix.kernels.attention import attention_ref, flash_attention
from helix.benchmark.runner import benchmark, compare

key = jax.random.PRNGKey(0)

# RMSNorm — memory bandwidth bound (low arithmetic intensity)
B, S, D = 4, 1024, 4096
x  = jax.random.normal(key, (B, S, D))
w  = jnp.ones(D)

print("=" * 56)
print("RMSNorm")
ref_result = benchmark(lambda: rmsnorm_ref(x, w), name="rmsnorm_ref", iters=100)
print(ref_result)
try:
    pal_result = benchmark(lambda: rmsnorm_pallas(x, w), name="rmsnorm_pallas", iters=100)
    print(pal_result)
    print(compare([ref_result, pal_result]))
except RuntimeError as e:
    print(f"  [Pallas unavailable] {e}")

# RoPE — elementwise, good fusion target
H = 32
x_rope = jax.random.normal(key, (B, S, H, D // H))

print("\n" + "=" * 56)
print("RoPE")
ref_rope = benchmark(lambda: apply_rope(x_rope), name="rope_ref", iters=100)
print(ref_rope)
try:
    pal_rope = benchmark(lambda: rope_pallas(x_rope), name="rope_pallas", iters=100)
    print(pal_rope)
    print(compare([ref_rope, pal_rope]))
except RuntimeError as e:
    print(f"  [Pallas unavailable] {e}")

# Attention — Flash Attention vs naive O(S²) attention
S_attn = 512
q = jax.random.normal(key, (B, S_attn, H, D // H))
k = jax.random.normal(key, (B, S_attn, H, D // H))
v = jax.random.normal(key, (B, S_attn, H, D // H))

print("\n" + "=" * 56)
print("Attention")
ref_attn = benchmark(lambda: attention_ref(q, k, v), name="attention_ref", iters=50)
print(ref_attn)
try:
    flash = benchmark(lambda: flash_attention(q, k, v), name="flash_attention", iters=50)
    print(flash)
    print(compare([ref_attn, flash]))
except RuntimeError as e:
    print(f"  [Pallas unavailable] {e}")
