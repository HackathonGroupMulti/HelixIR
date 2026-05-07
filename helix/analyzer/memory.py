from __future__ import annotations
import math
from ..tracer.graph import OpGraph, OpNode, MATMUL_PRIMITIVES, REDUCTION_PRIMITIVES

_DTYPE_BYTES: dict[str, int] = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
    "bool": 1, "complex64": 8, "complex128": 16,
}


def _bytes_for(dtype: str) -> int:
    for key, val in _DTYPE_BYTES.items():
        if key in dtype:
            return val
    return 4


def _numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    out = 1
    for d in shape:
        out *= d
    return out


def _matmul_flops(node: OpNode) -> int:
    if len(node.input_shapes) < 2:
        return 0
    a, b = node.input_shapes[0], node.input_shapes[1]
    if len(a) < 2 or len(b) < 2:
        return 0
    # Batch dims
    batch = 1
    for d in a[:-2]:
        batch *= d
    M, K = a[-2], a[-1]
    N = b[-1]
    return 2 * batch * M * K * N


# Rough multipliers for transcendental ops relative to a simple add
_TRANSCENDENTAL_COST: dict[str, int] = {
    "exp": 20, "exp2": 15, "expm1": 22,
    "log": 20, "log1p": 22, "log2": 18,
    "sqrt": 5, "rsqrt": 6, "cbrt": 8,
    "sin": 20, "cos": 20, "tan": 25,
    "asin": 30, "acos": 30, "atan": 25,
    "sinh": 25, "cosh": 25, "tanh": 20,
    "erf": 20, "erfc": 22, "erfinv": 30,
    "logistic": 22,
}


def _estimate_flops(node: OpNode) -> int:
    name = node.name
    out_elems = sum(_numel(s) for s in node.output_shapes)

    if name in MATMUL_PRIMITIVES:
        return _matmul_flops(node)
    if name in REDUCTION_PRIMITIVES:
        return sum(_numel(s) for s in node.input_shapes) if node.input_shapes else 0
    if name == "conv_general_dilated":
        return _matmul_flops(node)  # approximation
    cost = _TRANSCENDENTAL_COST.get(name, 1)
    return out_elems * cost


def analyze_memory(graph: OpGraph) -> OpGraph:
    """Fill in flops, bytes_read, bytes_written, arithmetic_intensity for every node."""
    for node in graph.nodes:
        bpe = _bytes_for(node.dtype)
        node.bytes_read = sum(_numel(s) for s in node.input_shapes) * bpe
        node.bytes_written = sum(_numel(s) for s in node.output_shapes) * bpe
        node.flops = _estimate_flops(node)
        total_bytes = node.bytes_read + node.bytes_written
        node.arithmetic_intensity = node.flops / total_bytes if total_bytes > 0 else 0.0
    return graph
