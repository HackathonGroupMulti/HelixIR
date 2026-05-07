from __future__ import annotations
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .graph import (
    OpGraph, OpNode, OpEdge,
    ELEMENTWISE_PRIMITIVES, REDUCTION_PRIMITIVES,
    MATMUL_PRIMITIVES, MEMORY_PRIMITIVES, COLLECTIVE_PRIMITIVES,
)


def _aval_shape(aval: Any) -> tuple[int, ...]:
    return tuple(aval.shape) if hasattr(aval, "shape") else ()


def _aval_dtype(aval: Any) -> str:
    return str(aval.dtype) if hasattr(aval, "dtype") else "unknown"


def _categorize(name: str) -> str:
    if name in ELEMENTWISE_PRIMITIVES:
        return "elementwise"
    if name in REDUCTION_PRIMITIVES:
        return "reduction"
    if name in MATMUL_PRIMITIVES:
        return "matmul"
    if name in MEMORY_PRIMITIVES:
        return "memory"
    if name in COLLECTIVE_PRIMITIVES:
        return "collective"
    return "other"


def capture(fn: Callable, *args: Any) -> tuple[Any, OpGraph]:
    """
    Trace fn(*args) via jax.make_jaxpr and return (jaxpr, OpGraph).

    The OpGraph has flops/bytes set to zero — run analyzer passes to fill them.
    """
    jaxpr = jax.make_jaxpr(fn)(*args)

    nodes: list[OpNode] = []
    edges: list[OpEdge] = []

    # Map python id(Var) → node id that last wrote it
    var_producer: dict[int, int] = {}

    for i, eqn in enumerate(jaxpr.jaxpr.eqns):
        from jax._src.core import Var, Literal

        input_shapes = [
            _aval_shape(v.aval) if isinstance(v, Var) else ()
            for v in eqn.invars
        ]
        output_shapes = [_aval_shape(v.aval) for v in eqn.outvars]

        dtype = _aval_dtype(eqn.outvars[0].aval) if eqn.outvars else "unknown"

        node = OpNode(
            id=i,
            name=eqn.primitive.name,
            category=_categorize(eqn.primitive.name),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            dtype=dtype,
            params={k: str(v) for k, v in eqn.params.items()},
        )
        nodes.append(node)

        # Record edges from producer nodes to this node
        for v in eqn.invars:
            if isinstance(v, Var):
                vid = id(v)
                if vid in var_producer:
                    edges.append(OpEdge(
                        src=var_producer[vid],
                        dst=i,
                        shape=_aval_shape(v.aval),
                        dtype=_aval_dtype(v.aval),
                    ))

        # Register outputs
        for v in eqn.outvars:
            if isinstance(v, Var):
                var_producer[id(v)] = i

    from jax._src.core import Var
    in_shapes = [_aval_shape(v.aval) for v in jaxpr.jaxpr.invars if isinstance(v, Var)]
    out_shapes = [_aval_shape(v.aval) for v in jaxpr.jaxpr.outvars if isinstance(v, Var)]

    return jaxpr, OpGraph(nodes=nodes, edges=edges, input_shapes=in_shapes, output_shapes=out_shapes)


def capture_stablehlo(fn: Callable, *args: Any) -> str:
    """Return the StableHLO text for fn(*args) after JIT lowering."""
    abstract = [
        jax.ShapeDtypeStruct(a.shape, a.dtype) if hasattr(a, "shape") else a
        for a in args
    ]
    try:
        lowered = jax.jit(fn).lower(*abstract)
        return str(lowered.compiler_ir(dialect="stablehlo"))
    except Exception as exc:
        return f"[StableHLO unavailable: {exc}]"


def xla_cost_analysis(fn: Callable, *args: Any) -> list[dict]:
    """
    Return XLA's own per-op cost analysis dicts (flop_count, bytes_accessed, …).
    Requires compilation so it's slower than JAXPR tracing.
    """
    abstract = [
        jax.ShapeDtypeStruct(a.shape, a.dtype) if hasattr(a, "shape") else a
        for a in args
    ]
    try:
        compiled = jax.jit(fn).lower(*abstract).compile()
        return compiled.cost_analysis()
    except Exception:
        return []
