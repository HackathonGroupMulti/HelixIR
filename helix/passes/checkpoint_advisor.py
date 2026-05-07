from __future__ import annotations
from ..tracer.graph import OpGraph, MATMUL_PRIMITIVES
from ..analyzer.memory import _numel, _bytes_for
from .base import OptimizationPass, PassResult

_MIN_TENSOR_MB = 10.0  # only flag tensors larger than this


class CheckpointAdvisorPass(OptimizationPass):
    """
    Finds the largest intermediate activations — those are the tensors that
    autodiff must keep alive during the backward pass.  Flagging them as
    jax.checkpoint candidates trades recompute cost for memory savings.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @property
    def name(self) -> str:
        return "CheckpointAdvisor"

    def run(self, graph: OpGraph) -> PassResult:
        candidates: list[dict] = []

        for node in graph.nodes:
            # Matmul outputs are usually rematerialized by XLA grad already
            if node.name in MATMUL_PRIMITIVES:
                continue
            for shape in node.output_shapes:
                size_bytes = _numel(shape) * _bytes_for(node.dtype)
                size_mb = size_bytes / 1e6
                if size_mb >= _MIN_TENSOR_MB:
                    candidates.append({
                        "node_id": node.id,
                        "op": node.name,
                        "shape": list(shape),
                        "size_mb": round(size_mb, 2),
                    })

        candidates.sort(key=lambda c: c["size_mb"], reverse=True)
        top = candidates[: self.top_k]
        total_mb = sum(c["size_mb"] for c in top)

        recs = [
            {
                "type": "checkpoint_candidate",
                **c,
                "message": (
                    f"Node {c['node_id']} ({c['op']}) produces {c['size_mb']:.1f} MB "
                    f"tensor {c['shape']} — wrap containing function with "
                    f"jax.checkpoint to recompute instead of store."
                ),
                "code_hint": "@jax.checkpoint",
            }
            for c in top
        ]

        summary = (
            f"Top {len(recs)} activation checkpoint candidates "
            f"hold {total_mb:.1f} MB during backward pass."
        )
        return PassResult(pass_name=self.name, recommendations=recs, summary=summary)
