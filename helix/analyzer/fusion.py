from __future__ import annotations
from ..tracer.graph import OpGraph, OpNode, ELEMENTWISE_PRIMITIVES, MEMORY_PRIMITIVES

# Ops that XLA can freely fuse into elementwise clusters
_XLA_FUSABLE = ELEMENTWISE_PRIMITIVES | frozenset({
    "broadcast_in_dim", "convert_element_type", "select_n",
})

# Ops that always break XLA's fusion boundary
_FUSION_BARRIERS = frozenset({
    "dot_general", "dot", "conv_general_dilated",
    "reduce_sum", "reduce_max", "reduce_min", "reduce_prod",
    "reduce_window_sum", "reduce_window_max",
    "all_reduce", "all_gather", "reduce_scatter", "psum",
    "gather", "scatter", "scatter_add",
    "sort", "sort_key_val",
    "fft", "ifft",
})

# Memory ops that XLA sometimes fuses but sometimes doesn't — flag them
_SOFT_BARRIERS = frozenset({"transpose", "reshape", "pad", "concatenate", "slice"})


def _assign_fusion_groups(graph: OpGraph) -> dict[int, int]:
    """
    Walk nodes in topological order and assign each to a fusion group.
    A new group starts whenever a fusion barrier is hit.
    Returns {node_id -> group_id}.
    """
    group_map: dict[int, int] = {}
    current_group = 0

    for node in graph.nodes:
        if node.name in _FUSION_BARRIERS:
            node.fusion_group = None   # barriers live outside any group
            current_group += 1
        elif node.name in _SOFT_BARRIERS:
            node.fusion_group = None
            current_group += 1
        else:
            node.fusion_group = current_group
            group_map[node.id] = current_group

    return group_map


def detect_fusion_opportunities(graph: OpGraph) -> list[dict]:
    """
    Return a list of dicts describing fusion opportunities and soft barriers.

    Two kinds of result:
      type="fusion_opportunity"  — a group of ≥3 fusable ops that could save memory traffic
      type="soft_barrier"        — a reshape/transpose that may break an elementwise chain
    """
    group_map = _assign_fusion_groups(graph)

    # Collect nodes per fusion group
    groups: dict[int, list[OpNode]] = {}
    for node in graph.nodes:
        if node.fusion_group is not None:
            groups.setdefault(node.fusion_group, []).append(node)

    opportunities: list[dict] = []

    for g, ops in groups.items():
        fusable = [op for op in ops if op.name in _XLA_FUSABLE]
        if len(fusable) < 3:
            continue

        # Memory traffic without fusion: every op reads its inputs + writes outputs
        unfused_bytes = sum(op.bytes_read + op.bytes_written for op in fusable)
        # With ideal fusion: only first op's reads + last op's writes
        fused_bytes = (
            (fusable[0].bytes_read if fusable else 0) +
            (fusable[-1].bytes_written if fusable else 0)
        )
        savings = max(0, unfused_bytes - fused_bytes)

        opportunities.append({
            "type": "fusion_opportunity",
            "group": g,
            "ops": [op.name for op in fusable],
            "op_ids": [op.id for op in fusable],
            "unfused_bytes": unfused_bytes,
            "fused_bytes": fused_bytes,
            "estimated_savings_bytes": savings,
            "estimated_savings_mb": round(savings / 1e6, 3),
        })

    # Flag soft barriers that sit between elementwise chains
    for node in graph.nodes:
        if node.name not in _SOFT_BARRIERS:
            continue
        preds = graph.predecessors(node.id)
        succs = graph.successors(node.id)
        has_ew_pred = any(p.name in ELEMENTWISE_PRIMITIVES for p in preds)
        has_ew_succ = any(s.name in ELEMENTWISE_PRIMITIVES for s in succs)
        if has_ew_pred and has_ew_succ:
            opportunities.append({
                "type": "soft_barrier",
                "node_id": node.id,
                "op": node.name,
                "message": (
                    f"'{node.name}' at node {node.id} sits between two "
                    f"elementwise chains and may prevent XLA from fusing them."
                ),
            })

    return opportunities
