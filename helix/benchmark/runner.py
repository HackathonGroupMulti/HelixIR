"""
Benchmark harness with proper JAX semantics.

Key points:
  - jax.block_until_ready() is mandatory — JAX uses async dispatch so
    time.perf_counter() after a bare call measures queue-submission, not execution.
  - Warmup triggers JIT compilation and flushes cache effects.
  - We report mean ± std over many iters, plus optional TFLOPS.
"""
from __future__ import annotations
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Any

import jax
import jax.numpy as jnp


@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    flops: int = 0
    achieved_tflops: float = 0.0
    peak_tflops: float = 0.0
    efficiency_pct: float = 0.0

    def __str__(self) -> str:
        lines = [
            f"  [{self.name}]",
            f"    mean {self.mean_ms:.3f} ms  std {self.std_ms:.3f} ms",
            f"    min  {self.min_ms:.3f} ms  max {self.max_ms:.3f} ms",
        ]
        if self.achieved_tflops > 0:
            lines.append(
                f"    {self.achieved_tflops:.2f} TFLOPS  "
                f"({self.efficiency_pct:.1f}% of peak {self.peak_tflops:.0f} TFLOPS)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "iterations": self.iterations,
            "flops": self.flops,
            "achieved_tflops": self.achieved_tflops,
            "peak_tflops": self.peak_tflops,
            "efficiency_pct": self.efficiency_pct,
        }


def benchmark(
    fn: Callable,
    *args: Any,
    name: str = "fn",
    warmup: int = 10,
    iters: int = 100,
    flops: int = 0,
    peak_tflops: float = 0.0,
) -> BenchmarkResult:
    """
    Time a JIT-compiled JAX function.

    Args:
        fn:         Function to benchmark.
        *args:      Arguments passed to fn on every call.
        name:       Label for the result.
        warmup:     Number of calls before timing starts (covers JIT compile).
        iters:      Number of timed iterations.
        flops:      Known FLOP count for the operation (used for TFLOPS calc).
        peak_tflops: Device peak TFLOPS (used for efficiency %).
    """
    jit_fn = jax.jit(fn)

    for _ in range(warmup):
        jax.block_until_ready(jit_fn(*args))

    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fn(*args))
        times_ms.append((time.perf_counter() - t0) * 1e3)

    mean_ms = statistics.mean(times_ms)
    std_ms  = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    achieved_tflops = 0.0
    efficiency_pct  = 0.0
    if flops > 0 and mean_ms > 0:
        achieved_tflops = (flops / (mean_ms / 1e3)) / 1e12
        if peak_tflops > 0:
            efficiency_pct = (achieved_tflops / peak_tflops) * 100.0

    return BenchmarkResult(
        name=name,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        iterations=iters,
        flops=flops,
        achieved_tflops=achieved_tflops,
        peak_tflops=peak_tflops,
        efficiency_pct=efficiency_pct,
    )


def compare(results: list[BenchmarkResult], baseline_name: str | None = None) -> str:
    """Format a speedup comparison table against a baseline result."""
    if not results:
        return ""
    baseline = next((r for r in results if r.name == baseline_name), results[0])

    header = f"\n{'─'*56}\n  Benchmark Comparison (baseline: {baseline.name})\n{'─'*56}"
    rows = []
    for r in results:
        speedup = baseline.mean_ms / r.mean_ms if r.mean_ms > 0 else float("inf")
        tag = "(baseline)" if r.name == baseline.name else f"{speedup:.2f}x faster"
        rows.append(f"  {r.name:<30}  {r.mean_ms:>8.3f} ms  {tag}")

    return header + "\n" + "\n".join(rows) + f"\n{'─'*56}"
