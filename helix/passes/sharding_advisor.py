from __future__ import annotations
from ..tracer.graph import OpGraph, MATMUL_PRIMITIVES
from ..analyzer.memory import _numel, _bytes_for
from .base import OptimizationPass, PassResult

_TENSOR_PARALLEL_MIN_DIM = 2048   # only suggest TP if output dim ≥ this
_FSDP_PARAM_THRESHOLD_GB = 1.0    # suggest FSDP if estimated params exceed this


class ShardingAdvisorPass(OptimizationPass):
    """
    Recommends jax.sharding strategies — data parallelism, tensor parallelism,
    and FSDP — based on input shapes and the matmul topology of the graph.
    """

    def __init__(self, num_devices: int = 8):
        self.num_devices = num_devices

    @property
    def name(self) -> str:
        return "ShardingAdvisor"

    def run(self, graph: OpGraph) -> PassResult:
        recs: list[dict] = []
        nd = self.num_devices

        # --- Data parallelism ---
        if graph.input_shapes:
            batch = graph.input_shapes[0][0] if graph.input_shapes[0] else None
            if batch and batch % nd == 0:
                recs.append({
                    "type": "data_parallel",
                    "batch_size": batch,
                    "num_devices": nd,
                    "per_device_batch": batch // nd,
                    "message": (
                        f"Batch {batch} divides evenly across {nd} devices "
                        f"({batch // nd} per device). Use PartitionSpec('batch') "
                        f"on input axis 0."
                    ),
                    "code_hint": (
                        f"mesh = jax.sharding.Mesh(devices, ('batch',))\n"
                        f"sharding = NamedSharding(mesh, P('batch'))\n"
                        f"x = jax.device_put(x, sharding)"
                    ),
                })

        # --- Tensor parallelism on large matmuls ---
        for node in graph.nodes:
            if node.name not in MATMUL_PRIMITIVES:
                continue
            if not node.output_shapes:
                continue
            out = node.output_shapes[0]
            if len(out) < 2:
                continue
            col_dim = out[-1]
            if col_dim >= _TENSOR_PARALLEL_MIN_DIM and col_dim % nd == 0:
                recs.append({
                    "type": "tensor_parallel",
                    "node_id": node.id,
                    "output_shape": list(out),
                    "shard_dim": col_dim,
                    "message": (
                        f"matmul at node {node.id} outputs dim {col_dim} — "
                        f"column-wise tensor parallelism across {nd} devices "
                        f"gives {col_dim // nd} columns each."
                    ),
                    "code_hint": (
                        f"# Shard weight matrix along output (column) axis\n"
                        f"w_sharding = NamedSharding(mesh, P(None, 'model'))\n"
                        f"w = jax.device_put(w, w_sharding)"
                    ),
                })

        # --- FSDP if total matmul input bytes suggest large parameters ---
        total_param_bytes = sum(
            _numel(node.input_shapes[1]) * _bytes_for(node.dtype)
            for node in graph.nodes
            if node.name in MATMUL_PRIMITIVES and len(node.input_shapes) >= 2
        )
        if total_param_bytes >= _FSDP_PARAM_THRESHOLD_GB * 1e9:
            recs.append({
                "type": "fsdp",
                "estimated_param_gb": round(total_param_bytes / 1e9, 2),
                "message": (
                    f"~{total_param_bytes / 1e9:.2f} GB of parameter weight tensors detected. "
                    f"Consider jax.experimental.shard_map for FSDP-style full sharding."
                ),
                "code_hint": (
                    "from jax.experimental.shard_map import shard_map\n"
                    "# Shard params across all devices, replicate activations"
                ),
            })

        summary = (
            f"Generated {len(recs)} sharding recommendations for "
            f"{nd} devices."
        )
        return PassResult(pass_name=self.name, recommendations=recs, summary=summary)
