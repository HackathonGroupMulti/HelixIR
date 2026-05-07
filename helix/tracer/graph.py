from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

ELEMENTWISE_PRIMITIVES: frozenset[str] = frozenset({
    "add", "sub", "mul", "div", "rem", "neg", "abs",
    "exp", "exp2", "expm1", "log", "log1p", "log2",
    "sqrt", "rsqrt", "cbrt", "sin", "cos", "tan",
    "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "pow", "integer_pow", "logistic", "erf", "erfc", "erfinv",
    "max", "min", "sign", "floor", "ceil", "round",
    "lt", "le", "gt", "ge", "eq", "ne",
    "and", "or", "xor", "not",
    "convert_element_type", "bitcast_convert_type",
    "select_n", "clamp",
    "broadcast_in_dim", "squeeze",
})

REDUCTION_PRIMITIVES: frozenset[str] = frozenset({
    "reduce_sum", "reduce_max", "reduce_min", "reduce_prod",
    "reduce_and", "reduce_or", "reduce_xor",
    "reduce_window_sum", "reduce_window_max", "reduce_window_min",
})

MATMUL_PRIMITIVES: frozenset[str] = frozenset({
    "dot_general", "dot", "conv_general_dilated",
})

MEMORY_PRIMITIVES: frozenset[str] = frozenset({
    "transpose", "reshape",
    "gather", "scatter", "scatter_add", "scatter_mul",
    "slice", "dynamic_slice", "dynamic_update_slice",
    "concatenate", "pad", "rev",
    "sort", "sort_key_val",
})

COLLECTIVE_PRIMITIVES: frozenset[str] = frozenset({
    "all_reduce", "all_gather", "reduce_scatter", "psum",
    "all_to_all", "axis_index",
})


@dataclass
class OpNode:
    id: int
    name: str
    category: str  # elementwise | reduction | matmul | memory | collective | other
    input_shapes: list[tuple[int, ...]]
    output_shapes: list[tuple[int, ...]]
    dtype: str
    params: dict

    # Populated by analyzer passes
    flops: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    arithmetic_intensity: float = 0.0
    is_compute_bound: bool = False
    fusion_group: Optional[int] = None


@dataclass
class OpEdge:
    src: int
    dst: int
    shape: tuple[int, ...]
    dtype: str


@dataclass
class OpGraph:
    nodes: list[OpNode]
    edges: list[OpEdge]
    input_shapes: list[tuple[int, ...]]
    output_shapes: list[tuple[int, ...]]

    def node(self, node_id: int) -> Optional[OpNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def successors(self, node_id: int) -> list[OpNode]:
        dst_ids = {e.dst for e in self.edges if e.src == node_id}
        return [n for n in self.nodes if n.id in dst_ids]

    def predecessors(self, node_id: int) -> list[OpNode]:
        src_ids = {e.src for e in self.edges if e.dst == node_id}
        return [n for n in self.nodes if n.id in src_ids]

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "category": n.category,
                    "input_shapes": n.input_shapes,
                    "output_shapes": n.output_shapes,
                    "dtype": n.dtype,
                    "flops": n.flops,
                    "bytes_read": n.bytes_read,
                    "bytes_written": n.bytes_written,
                    "arithmetic_intensity": n.arithmetic_intensity,
                    "is_compute_bound": n.is_compute_bound,
                    "fusion_group": n.fusion_group,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "shape": list(e.shape),
                    "dtype": e.dtype,
                }
                for e in self.edges
            ],
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
        }
