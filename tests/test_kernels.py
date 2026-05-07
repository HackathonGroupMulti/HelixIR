"""
Tests for custom kernels — correctness via gradient checking and numerical
comparison against reference implementations.  Pallas tests are skipped on
CPU (no GPU available).
"""
import jax
import jax.numpy as jnp
import pytest

from helix.kernels.rmsnorm import rmsnorm_ref, rmsnorm
from helix.kernels.rope import apply_rope, rope_ref, precompute_freqs
from helix.kernels.attention import attention_ref, causal_mask


class TestRMSNorm:
    def test_output_shape(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 256))
        w = jnp.ones(256)
        y = rmsnorm_ref(x, w)
        assert y.shape == x.shape

    def test_normalisation(self):
        """RMSNorm output should have near-unit RMS per row."""
        x = jax.random.normal(jax.random.PRNGKey(1), (4, 128))
        w = jnp.ones(128)
        y = rmsnorm_ref(x, w)
        rms = jnp.sqrt(jnp.mean(y ** 2, axis=-1))
        assert jnp.allclose(rms, jnp.ones_like(rms), atol=1e-5)

    def test_gradient_shape(self):
        x = jax.random.normal(jax.random.PRNGKey(2), (2, 32, 64))
        w = jnp.ones(64)
        grad_fn = jax.grad(lambda x_, w_: jnp.sum(rmsnorm(x_, w_)))
        gx = grad_fn(x, w)
        assert gx.shape == x.shape

    def test_gradient_correctness(self):
        """Check custom VJP against jax.grad of the reference."""
        x = jax.random.normal(jax.random.PRNGKey(3), (2, 8, 16))
        w = jnp.ones(16)
        eps = 1e-6

        g_custom = jax.grad(lambda x_: jnp.sum(rmsnorm(x_, w, eps)))(x)
        g_ref    = jax.grad(lambda x_: jnp.sum(rmsnorm_ref(x_, w, eps)))(x)
        assert jnp.allclose(g_custom, g_ref, atol=1e-4)

    def test_scale_weight(self):
        x = jax.random.normal(jax.random.PRNGKey(4), (2, 32))
        w2 = jnp.full(32, 2.0)
        y1 = rmsnorm_ref(x, jnp.ones(32))
        y2 = rmsnorm_ref(x, w2)
        assert jnp.allclose(y2, 2.0 * y1, atol=1e-6)


class TestRoPE:
    def test_output_shape(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 8, 32))
        y = apply_rope(x)
        assert y.shape == x.shape

    def test_precompute_freqs_shape(self):
        cos, sin = precompute_freqs(128, 64)
        assert cos.shape == (128, 32)
        assert sin.shape == (128, 32)

    def test_rope_preserves_norm(self):
        """RoPE is an isometric rotation — norm should be preserved."""
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 16, 4, 64))
        y = apply_rope(x)
        norm_x = jnp.linalg.norm(x, axis=-1)
        norm_y = jnp.linalg.norm(y, axis=-1)
        assert jnp.allclose(norm_x, norm_y, atol=1e-5)

    def test_rope_ref_matches_apply_rope(self):
        x = jax.random.normal(jax.random.PRNGKey(2), (2, 32, 4, 64))
        cos, sin = precompute_freqs(32, 64)
        y1 = rope_ref(x, cos, sin)
        y2 = apply_rope(x)
        assert jnp.allclose(y1, y2, atol=1e-6)


class TestAttention:
    def test_output_shape(self):
        B, S, H, D = 2, 32, 4, 16
        q = jax.random.normal(jax.random.PRNGKey(0), (B, S, H, D))
        k = jax.random.normal(jax.random.PRNGKey(1), (B, S, H, D))
        v = jax.random.normal(jax.random.PRNGKey(2), (B, S, H, D))
        out = attention_ref(q, k, v)
        assert out.shape == (B, S, H, D)

    def test_softmax_weights_sum_to_one(self):
        """Verify attention weights sum to 1 along the key dimension."""
        B, S, H, D = 1, 8, 2, 8
        q = jax.random.normal(jax.random.PRNGKey(0), (B, S, H, D))
        k = jax.random.normal(jax.random.PRNGKey(1), (B, S, H, D))
        v = jnp.ones((B, S, H, D))
        out = attention_ref(q, k, v)
        # When V=1, output should be ~1 everywhere
        assert jnp.allclose(out, jnp.ones_like(out), atol=1e-5)

    def test_causal_mask_shape(self):
        mask = causal_mask(16)
        assert mask.shape == (1, 1, 16, 16)

    def test_causal_mask_lower_triangular(self):
        S = 8
        mask = causal_mask(S).squeeze()
        # Upper triangle (above diagonal) should be False
        for i in range(S):
            for j in range(i + 1, S):
                assert not mask[i, j]
