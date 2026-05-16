"""Tests for runtime_diagnostics."""
import jax
import jax.numpy as jnp
import pytest

from helix.diagnostics import runtime_diagnostics, print_diagnostics


def matmul_fn(x, w):
    return jnp.dot(x, w)


def mlp(x, w1, w2):
    return jnp.dot(jax.nn.relu(jnp.dot(x, w1)), w2)


class TestRuntimeDiagnostics:
    def _args(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 64))
        w = jax.random.normal(jax.random.PRNGKey(1), (64, 128))
        return x, w

    def test_returns_dict(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert isinstance(d, dict)

    def test_all_keys_present(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        expected = {
            "compile_latency_ms", "run_latency_ms", "compile_overhead_ms",
            "jit_speedup", "analysis_overhead_ms",
            "estimated_flops", "xla_flops", "flop_accuracy_pct",
            "estimated_bytes", "xla_bytes", "memory_accuracy_pct",
            "roofline_compute_pct", "roofline_bw_pct", "ridge_point",
            "recompile_risk", "recompile_overhead_ms",
        }
        assert expected.issubset(d.keys())

    def test_latencies_positive(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert d["compile_latency_ms"] > 0
        assert d["run_latency_ms"] > 0
        assert d["compile_overhead_ms"] >= 0

    def test_jit_speedup_at_least_one(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert d["jit_speedup"] >= 1.0

    def test_analysis_overhead_positive(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert d["analysis_overhead_ms"] > 0

    def test_estimated_flops_positive(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert d["estimated_flops"] > 0

    def test_estimated_bytes_positive(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert d["estimated_bytes"] > 0

    def test_roofline_pcts_sum_to_100(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        total = d["roofline_compute_pct"] + d["roofline_bw_pct"]
        assert abs(total - 100.0) < 0.1 or total == 0.0

    def test_ridge_point_positive(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert d["ridge_point"] > 0

    def test_recompile_risk_is_bool(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert isinstance(d["recompile_risk"], bool)

    def test_recompile_overhead_nonnegative(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert d["recompile_overhead_ms"] >= 0

    def test_flop_accuracy_none_or_float(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert d["flop_accuracy_pct"] is None or isinstance(d["flop_accuracy_pct"], float)

    def test_memory_accuracy_none_or_float(self):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        assert d["memory_accuracy_pct"] is None or isinstance(d["memory_accuracy_pct"], float)

    def test_mlp_diagnostics(self):
        x  = jax.random.normal(jax.random.PRNGKey(0), (4, 32))
        w1 = jax.random.normal(jax.random.PRNGKey(1), (32, 64))
        w2 = jax.random.normal(jax.random.PRNGKey(2), (64, 16))
        d = runtime_diagnostics(mlp, x, w1, w2, iters=5)
        assert d["estimated_flops"] > 0
        assert d["compile_latency_ms"] > 0

    def test_print_diagnostics_runs(self, capsys):
        x, w = self._args()
        d = runtime_diagnostics(matmul_fn, x, w, iters=5)
        print_diagnostics(d, fn_name="matmul_fn")
        out = capsys.readouterr().out
        assert "Latency" in out
        assert "FLOP" in out
        assert "Roofline" in out
        assert "Recompile" in out
