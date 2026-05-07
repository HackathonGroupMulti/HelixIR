"""Tests for analyzer passes: memory accounting, roofline, fusion detection."""
import jax.numpy as jnp
import pytest

from helix.tracer.capture import capture
from helix.analyzer.memory import analyze_memory
from helix.analyzer.bottleneck import analyze_bottleneck, RooflineResult
from helix.analyzer.fusion import detect_fusion_opportunities
from helix.passes.fusion_advisor import FusionAdvisorPass
from helix.passes.checkpoint_advisor import CheckpointAdvisorPass
from helix.passes.sharding_advisor import ShardingAdvisorPass


def matmul_fn(x, w):
    return jnp.dot(x, w)


def elementwise_fn(x):
    return jnp.exp(jnp.sin(x) + jnp.cos(x)) * jnp.log(jnp.abs(x) + 1.0)


def big_activation_fn(x, w):
    h = jnp.dot(x, w)          # [B, S, 4096] — large activation
    return jnp.tanh(h) + h


class TestMemoryAnalyzer:
    def test_matmul_flops_nonzero(self):
        x = jnp.ones((16, 512))
        w = jnp.ones((512, 1024))
        _, graph = capture(matmul_fn, x, w)
        graph = analyze_memory(graph)
        mm_nodes = [n for n in graph.nodes if n.category == "matmul"]
        assert mm_nodes
        assert mm_nodes[0].flops > 0

    def test_matmul_flops_correct(self):
        M, K, N = 8, 64, 128
        x = jnp.ones((M, K))
        w = jnp.ones((K, N))
        _, graph = capture(matmul_fn, x, w)
        graph = analyze_memory(graph)
        mm = next(n for n in graph.nodes if n.category == "matmul")
        expected = 2 * M * K * N
        assert mm.flops == expected

    def test_elementwise_has_bytes(self):
        x = jnp.ones((32, 64))
        _, graph = capture(elementwise_fn, x)
        graph = analyze_memory(graph)
        for node in graph.nodes:
            if node.category == "elementwise":
                assert node.bytes_read > 0 or node.bytes_written > 0

    def test_arithmetic_intensity_set(self):
        x = jnp.ones((16, 512))
        w = jnp.ones((512, 1024))
        _, graph = capture(matmul_fn, x, w)
        graph = analyze_memory(graph)
        mm = next(n for n in graph.nodes if n.category == "matmul")
        assert mm.arithmetic_intensity > 0


class TestBottleneckAnalyzer:
    def test_returns_roofline_result(self):
        x = jnp.ones((4, 128))
        w = jnp.ones((128, 256))
        _, graph = capture(matmul_fn, x, w)
        graph = analyze_memory(graph)
        result = analyze_bottleneck(graph, device="A100")
        assert isinstance(result, RooflineResult)

    def test_ridge_point_positive(self):
        x = jnp.ones((4, 128))
        w = jnp.ones((128, 256))
        _, graph = capture(matmul_fn, x, w)
        graph = analyze_memory(graph)
        r = analyze_bottleneck(graph, device="A100")
        assert r.ridge_point > 0

    def test_large_matmul_compute_bound(self):
        # Square 1024×1024 matmul: intensity = 2*M³ / (3*M²*4) = M/6 ≈ 171 FLOP/byte
        # A100 ridge point = 312 TFLOPS / 2 TB/s = 156 FLOP/byte → compute-bound
        x = jnp.ones((1024, 1024))
        w = jnp.ones((1024, 1024))
        _, graph = capture(matmul_fn, x, w)
        graph = analyze_memory(graph)
        r = analyze_bottleneck(graph, device="A100")
        assert len(r.compute_bound_ops) > 0


class TestFusionAdvisor:
    def test_finds_opportunities_in_chain(self):
        x = jnp.ones((64, 256))
        _, graph = capture(elementwise_fn, x)
        graph = analyze_memory(graph)
        opps = detect_fusion_opportunities(graph)
        assert len(opps) > 0

    def test_pass_result_has_summary(self):
        x = jnp.ones((64, 256))
        _, graph = capture(elementwise_fn, x)
        graph = analyze_memory(graph)
        result = FusionAdvisorPass().run(graph)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0


class TestCheckpointAdvisor:
    def test_flags_large_activation(self):
        x = jnp.ones((4, 512, 1024))
        w = jnp.ones((1024, 4096))
        _, graph = capture(big_activation_fn, x, w)
        graph = analyze_memory(graph)
        result = CheckpointAdvisorPass().run(graph)
        assert len(result.recommendations) > 0
        types = {r["type"] for r in result.recommendations}
        assert "checkpoint_candidate" in types


class TestShardingAdvisor:
    def test_data_parallel_suggestion(self):
        x = jnp.ones((16, 512, 1024))  # batch=16 divisible by 8
        w = jnp.ones((1024, 1024))
        _, graph = capture(matmul_fn, x.reshape(16 * 512, 1024), w)
        graph = analyze_memory(graph)
        result = ShardingAdvisorPass(num_devices=8).run(graph)
        types = [r["type"] for r in result.recommendations]
        assert "data_parallel" in types or "tensor_parallel" in types
