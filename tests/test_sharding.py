"""Tests for auto-sharding code generator and backward analyser."""
import jax
import jax.numpy as jnp
import pytest

from helix.sharding import generate_sharding, ShardingPlan, TensorPlan
from helix.backward import analyze_backward, analyze_full


def matmul_fn(x, w):
    return jnp.dot(x, w)


def mlp(x, w1, w2):
    return jnp.dot(jax.nn.relu(jnp.dot(x, w1)), w2)


def transformer_tiny(x, wq, wk, wv, wo, norm):
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    h   = (x / rms) * norm
    q   = jnp.einsum("bsd,dh->bsh", h, wq)
    k   = jnp.einsum("bsd,dh->bsh", h, wk)
    v   = jnp.einsum("bsd,dh->bsh", h, wv)
    s   = jnp.einsum("bqd,bkd->bqk", q, k) * (h.shape[-1] ** -0.5)
    a   = jax.nn.softmax(s, axis=-1)
    out = jnp.einsum("bqk,bkd->bqd", a, v)
    return x + jnp.einsum("bsd,dh->bsh", out, wo)


# ── ShardingPlan tests ───────────────────────────────────────────────────────

class TestGenerateSharding:
    def _make_matmul_args(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 128))
        w = jax.random.normal(jax.random.PRNGKey(1), (128, 512))
        return x, w

    def test_returns_sharding_plan(self):
        x, w = self._make_matmul_args()
        plan = generate_sharding(matmul_fn, x, w, mesh_shape=(2, 4))
        assert isinstance(plan, ShardingPlan)

    def test_code_is_string(self):
        x, w = self._make_matmul_args()
        plan = generate_sharding(matmul_fn, x, w)
        assert isinstance(plan.code, str)
        assert len(plan.code) > 0

    def test_code_contains_mesh(self):
        x, w = self._make_matmul_args()
        plan = generate_sharding(matmul_fn, x, w, mesh_shape=(2, 4))
        assert "Mesh" in plan.code
        assert "device_put" in plan.code

    def test_correct_tensor_count(self):
        x, w = self._make_matmul_args()
        plan = generate_sharding(matmul_fn, x, w)
        assert len(plan.tensors) == 2

    def test_arg_names_in_code(self):
        x, w = self._make_matmul_args()
        plan = generate_sharding(matmul_fn, x, w, arg_names=["x_input", "weight"])
        assert "x_input" in plan.code
        assert "weight" in plan.code

    def test_activation_gets_batch_shard(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (16, 128))
        w = jax.random.normal(jax.random.PRNGKey(1), (128, 256))
        plan = generate_sharding(matmul_fn, x, w, mesh_shape=(2, 4))
        # First tensor (activation) should have 'batch' in its partition spec
        act = plan.tensors[0]
        assert "batch" in act.partition_spec or act.role == "activation"

    def test_weight_gets_model_shard(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 128))
        w = jax.random.normal(jax.random.PRNGKey(1), (128, 512))  # fan-out
        plan = generate_sharding(matmul_fn, x, w, mesh_shape=(2, 4))
        wt = plan.tensors[1]
        assert wt.role in ("weight_col", "weight_row")

    def test_mlp_plan(self):
        x  = jax.random.normal(jax.random.PRNGKey(0), (4, 64))
        w1 = jax.random.normal(jax.random.PRNGKey(1), (64, 256))
        w2 = jax.random.normal(jax.random.PRNGKey(2), (256, 64))
        plan = generate_sharding(mlp, x, w1, w2, mesh_shape=(1, 4))
        assert len(plan.tensors) == 3
        # w1 is fan-out (64→256), w2 is fan-in (256→64)
        roles = {t.name: t.role for t in plan.tensors}
        assert any(r == "weight_col" for r in roles.values())
        assert any(r == "weight_row" for r in roles.values())

    def test_str_renders(self):
        x, w = self._make_matmul_args()
        plan = generate_sharding(matmul_fn, x, w)
        s = str(plan)
        assert "ShardingPlan" in s

    def test_to_dict(self):
        import json
        x, w = self._make_matmul_args()
        plan = generate_sharding(matmul_fn, x, w)
        d = plan.to_dict()
        json.dumps(d)  # must be JSON-serialisable


# ── Backward analyser tests ──────────────────────────────────────────────────

class TestAnalyzeBackward:
    def test_returns_report_dict(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 32))
        w = jax.random.normal(jax.random.PRNGKey(1), (32, 64))
        report = analyze_backward(matmul_fn, x, w)
        assert "graph" in report
        assert "roofline" in report
        assert "num_ops" in report

    def test_backward_has_more_ops_than_forward(self):
        import helix
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 64))
        w = jax.random.normal(jax.random.PRNGKey(1), (64, 128))
        fwd = helix.analyze(matmul_fn, x, w)
        bwd = analyze_backward(matmul_fn, x, w)
        assert bwd["num_ops"] >= fwd["num_ops"]

    def test_backward_flops_exceed_forward(self):
        import helix
        x  = jax.random.normal(jax.random.PRNGKey(0), (4, 64))
        w1 = jax.random.normal(jax.random.PRNGKey(1), (64, 256))
        w2 = jax.random.normal(jax.random.PRNGKey(2), (256, 64))
        fwd = helix.analyze(mlp, x, w1, w2)
        bwd = analyze_backward(mlp, x, w1, w2)
        assert bwd["total_flops"] > fwd["total_flops"]

    def test_analyze_full_ratio(self):
        x  = jax.random.normal(jax.random.PRNGKey(0), (4, 64))
        w1 = jax.random.normal(jax.random.PRNGKey(1), (64, 256))
        w2 = jax.random.normal(jax.random.PRNGKey(2), (256, 64))
        full = analyze_full(mlp, x, w1, w2)
        assert "forward" in full
        assert "backward" in full
        assert "bwd_fwd_flop_ratio" in full
        assert full["bwd_fwd_flop_ratio"] > 1.0
