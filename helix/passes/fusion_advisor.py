from __future__ import annotations
from ..tracer.graph import OpGraph
from ..analyzer.fusion import detect_fusion_opportunities
from .base import OptimizationPass, PassResult


class FusionAdvisorPass(OptimizationPass):
    """
    Identifies elementwise chains that could be kernel-fused to eliminate
    redundant memory round-trips, and soft barriers (reshape/transpose) that
    may prevent XLA from merging adjacent fusion clusters.
    """

    @property
    def name(self) -> str:
        return "FusionAdvisor"

    def run(self, graph: OpGraph) -> PassResult:
        raw = detect_fusion_opportunities(graph)
        recs: list[dict] = []
        total_savings_mb = 0.0

        for item in raw:
            if item["type"] == "fusion_opportunity":
                savings_mb = item["estimated_savings_mb"]
                total_savings_mb += savings_mb
                recs.append({
                    "type": "fusion_opportunity",
                    "ops": item["ops"],
                    "op_ids": item["op_ids"],
                    "estimated_savings_mb": savings_mb,
                    "message": (
                        f"Fuse {len(item['ops'])} elementwise ops "
                        f"(~{savings_mb:.1f} MB memory traffic saved). "
                        f"Apply jax.jit — XLA will auto-fuse if no barriers exist."
                    ),
                    "code_hint": "@jax.jit",
                })
            elif item["type"] == "soft_barrier":
                recs.append({
                    "type": "soft_barrier",
                    "node_id": item["node_id"],
                    "op": item["op"],
                    "message": item["message"],
                    "code_hint": (
                        f"# Consider restructuring to avoid {item['op']} "
                        f"between elementwise ops"
                    ),
                })

        summary = (
            f"Found {sum(1 for r in recs if r['type'] == 'fusion_opportunity')} fusion groups "
            f"(~{total_savings_mb:.1f} MB potential savings) and "
            f"{sum(1 for r in recs if r['type'] == 'soft_barrier')} soft barriers."
        )
        return PassResult(pass_name=self.name, recommendations=recs, summary=summary)
