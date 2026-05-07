from __future__ import annotations
from dataclasses import dataclass, field
from ..tracer.graph import OpGraph

# Peak hardware specs: (peak_flops/s FP16, peak_bandwidth bytes/s)
_DEVICE_SPECS: dict[str, dict[str, float]] = {
    "H100 SXM5 80GB":    {"peak_flops": 989e12,  "peak_bw": 3.35e12},
    "H100 SXM":          {"peak_flops": 989e12,  "peak_bw": 3.35e12},
    "H100":              {"peak_flops": 989e12,  "peak_bw": 3.35e12},
    "A100-SXM4-80GB":    {"peak_flops": 312e12,  "peak_bw": 2.0e12},
    "A100-SXM4-40GB":    {"peak_flops": 312e12,  "peak_bw": 1.555e12},
    "A100":              {"peak_flops": 312e12,  "peak_bw": 2.0e12},
    "V100-SXM2-32GB":    {"peak_flops": 112e12,  "peak_bw": 900e9},
    "V100":              {"peak_flops": 112e12,  "peak_bw": 900e9},
    "RTX 4090":          {"peak_flops": 165e12,  "peak_bw": 1.0e12},
    "RTX 3090":          {"peak_flops": 71e12,   "peak_bw": 936e9},
    "RTX 3080":          {"peak_flops": 60e12,   "peak_bw": 760e9},
    "CPU":               {"peak_flops": 1e12,    "peak_bw": 50e9},
}


def _detect_device() -> str:
    try:
        import jax
        device = jax.devices()[0]
        kind = device.device_kind
        for name in _DEVICE_SPECS:
            if name.lower() in kind.lower():
                return name
        if "cpu" in kind.lower():
            return "CPU"
        return "A100"
    except Exception:
        return "CPU"


@dataclass
class RooflineResult:
    device: str
    peak_flops: float        # FLOP/s
    peak_bandwidth: float    # bytes/s
    ridge_point: float       # FLOP/byte — crossover point

    compute_bound_ops: list[str] = field(default_factory=list)
    bandwidth_bound_ops: list[str] = field(default_factory=list)
    total_flops: int = 0
    total_bytes: int = 0

    def achievable_tflops(self, intensity: float) -> float:
        """Roofline-predicted TFLOPS for a given arithmetic intensity."""
        return min(self.peak_flops, intensity * self.peak_bandwidth) / 1e12

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "peak_flops": self.peak_flops,
            "peak_bandwidth": self.peak_bandwidth,
            "ridge_point": self.ridge_point,
            "compute_bound_ops": self.compute_bound_ops,
            "bandwidth_bound_ops": self.bandwidth_bound_ops,
            "total_flops": self.total_flops,
            "total_bytes": self.total_bytes,
        }


def analyze_bottleneck(graph: OpGraph, device: str | None = None) -> RooflineResult:
    """
    Classify every node as compute-bound or memory-bandwidth-bound using the
    roofline model.  Requires analyze_memory() to have been run first.
    """
    if device is None:
        device = _detect_device()

    specs = _DEVICE_SPECS.get(device, _DEVICE_SPECS["CPU"])
    peak_flops = specs["peak_flops"]
    peak_bw = specs["peak_bw"]
    ridge = peak_flops / peak_bw

    compute_bound: list[str] = []
    bw_bound: list[str] = []
    total_flops = 0
    total_bytes = 0

    for node in graph.nodes:
        total_flops += node.flops
        total_bytes += node.bytes_read + node.bytes_written

        if node.flops == 0:
            continue

        if node.arithmetic_intensity >= ridge:
            node.is_compute_bound = True
            compute_bound.append(node.name)
        else:
            node.is_compute_bound = False
            bw_bound.append(node.name)

    return RooflineResult(
        device=device,
        peak_flops=peak_flops,
        peak_bandwidth=peak_bw,
        ridge_point=ridge,
        compute_bound_ops=compute_bound,
        bandwidth_bound_ops=bw_bound,
        total_flops=total_flops,
        total_bytes=total_bytes,
    )
