"""Tests for the tracer layer (capture + graph construction)."""
import jax
import jax.numpy as jnp
import pytest

from helix.tracer.capture import capture
from helix.tracer.graph import OpGraph, MATMUL_PRIMITIVES


def simple_fn(x, w):
    return jnp.dot(x, w)


def elementwise_chain(x):
    return jnp.exp(jnp.log(jnp.sqrt(jnp.abs(x) + 1.0)))


def mixed_fn(x, w):
    h = jnp.dot(x, w)
    return jax.nn.relu(h) + jnp.exp(-h)


class TestCapture:
    def test_returns_opgraph(self):
        x = jnp.ones((4, 8))
        w = jnp.ones((8, 16))
        _, graph = capture(simple_fn, x, w)
        assert isinstance(graph, OpGraph)

    def test_has_matmul_node(self):
        x = jnp.ones((4, 8))
        w = jnp.ones((8, 16))
        _, graph = capture(simple_fn, x, w)
        categories = {n.category for n in graph.nodes}
        assert "matmul" in categories

    def test_input_shapes_recorded(self):
        x = jnp.ones((2, 3, 64))
        w = jnp.ones((64, 128))
        _, graph = capture(simple_fn, x.reshape(6, 64), w)
        assert graph.input_shapes != []

    def test_elementwise_chain(self):
        x = jnp.ones((16, 32))
        _, graph = capture(elementwise_chain, x)
        ew_nodes = [n for n in graph.nodes if n.category == "elementwise"]
        assert len(ew_nodes) >= 2

    def test_edges_connect_nodes(self):
        x = jnp.ones((8, 16))
        w = jnp.ones((16, 32))
        _, graph = capture(mixed_fn, x, w)
        assert len(graph.edges) > 0
        node_ids = {n.id for n in graph.nodes}
        for edge in graph.edges:
            assert edge.src in node_ids
            assert edge.dst in node_ids

    def test_to_dict_serialisable(self):
        import json
        x = jnp.ones((4, 4))
        w = jnp.ones((4, 8))
        _, graph = capture(simple_fn, x, w)
        d = graph.to_dict()
        # Should be JSON-serialisable without custom encoder
        json.dumps(d)
