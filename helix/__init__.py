"""
HelixIR — Automatic Performance Optimizer for JAX Programs.

Quick start
-----------
    import helix
    import jax.numpy as jnp

    @helix.profile
    def my_model(x, w):
        return jnp.dot(x, w)

    out = my_model(x, w)              # prints analysis on first call
    report = my_model._helix_report   # access structured report dict

Sharding code generation
------------------------
    plan = helix.generate_sharding(
        my_model, x, w,
        mesh_shape=(2, 4),
        arg_names=['x', 'w'],
    )
    print(plan.code)   # copy-paste into training script

Backward pass analysis
----------------------
    bwd  = helix.analyze_backward(my_model, x, w)
    full = helix.analyze_full(my_model, x, w)
    print(f"bwd/fwd FLOP ratio: {full['bwd_fwd_flop_ratio']:.2f}×")

Jupyter integration
-------------------
    %load_ext helix.jupyter
    %helix my_model x w --bwd --devices 8
"""
from __future__ import annotations
import functools
from typing import Callable, Any

from .tracer.capture import capture, capture_stablehlo, xla_cost_analysis
from .tracer.graph import OpGraph
from .analyzer.memory import analyze_memory
from .analyzer.bottleneck import analyze_bottleneck, RooflineResult
from .analyzer.fusion import detect_fusion_opportunities
from .passes.fusion_advisor import FusionAdvisorPass
from .passes.checkpoint_advisor import CheckpointAdvisorPass
from .passes.sharding_advisor import ShardingAdvisorPass
from .benchmark.runner import benchmark, compare, BenchmarkResult
from .sharding import generate_sharding, ShardingPlan
from .backward import analyze_backward, analyze_full, print_full_report
from .diagnostics import runtime_diagnostics, print_diagnostics

__version__ = "0.2.0"
__all__ = [
    # Core analysis
    "profile", "analyze",
    # Sharding
    "generate_sharding", "ShardingPlan",
    # Backward
    "analyze_backward", "analyze_full", "print_full_report",
    # Diagnostics
    "runtime_diagnostics", "print_diagnostics",
    # Benchmark
    "benchmark", "compare", "BenchmarkResult",
    # Low-level
    "capture", "capture_stablehlo", "xla_cost_analysis",
    "OpGraph", "RooflineResult",
]


def analyze(fn: Callable, *args: Any, num_devices: int = 8, device: str | None = None) -> dict:
    """
    Full static analysis of fn(*args).

    Returns a dict with keys:
        graph       OpGraph with per-node memory/flop annotations
        roofline    RooflineResult (device classification, ridge point, …)
        passes      list[PassResult] from all three advisor passes
        num_ops     int
        total_flops int
        total_bytes int
    """
    _, graph = capture(fn, *args)
    graph = analyze_memory(graph)
    roofline = analyze_bottleneck(graph, device=device)

    passes = [
        FusionAdvisorPass().run(graph),
        CheckpointAdvisorPass().run(graph),
        ShardingAdvisorPass(num_devices=num_devices).run(graph),
    ]

    return {
        "graph": graph,
        "roofline": roofline,
        "passes": passes,
        "num_ops": len(graph.nodes),
        "total_flops": roofline.total_flops,
        "total_bytes": roofline.total_bytes,
    }


def _print_report(report: dict, fn_name: str) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    r = report["roofline"]

    console.rule(f"[bold cyan]HelixIR[/bold cyan] · {fn_name}")
    console.print(f"  Device       [bold]{r.device}[/bold]")
    console.print(f"  Ops          {report['num_ops']}")
    console.print(f"  Total FLOPs  {report['total_flops'] / 1e9:.2f} GFLOPs")
    console.print(f"  Total bytes  {report['total_bytes'] / 1e6:.2f} MB")
    console.print(
        f"  Ridge point  {r.ridge_point:.1f} FLOP/byte  "
        f"([green]{len(r.compute_bound_ops)} compute-bound[/green] | "
        f"[yellow]{len(r.bandwidth_bound_ops)} bandwidth-bound[/yellow])"
    )

    for pr in report["passes"]:
        console.print(f"\n  [bold]{pr.pass_name}[/bold]  {pr.summary}")
        for rec in pr.recommendations[:4]:
            console.print(f"    [dim]•[/dim] {rec['message']}")
            if "code_hint" in rec:
                console.print(f"      [green]{rec['code_hint']}[/green]")

    console.rule()


def profile(
    fn: Callable | None = None,
    *,
    num_devices: int = 8,
    device: str | None = None,
    verbose: bool = True,
) -> Callable:
    """
    Decorator that analyses a JAX function on its first call and attaches the
    report to ``wrapper._helix_report``.

    Can be used with or without arguments::

        @helix.profile
        def fn(x): ...

        @helix.profile(num_devices=4)
        def fn(x): ...
    """
    def decorator(f: Callable) -> Callable:
        _done = False

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal _done
            if not _done:
                _done = True
                try:
                    report = analyze(f, *args, num_devices=num_devices, device=device)
                    if verbose:
                        _print_report(report, f.__name__)
                    wrapper._helix_report = report
                except Exception as exc:
                    print(f"[HelixIR] Analysis skipped ({type(exc).__name__}: {exc})")
                    wrapper._helix_report = None
            return f(*args, **kwargs)

        wrapper._helix_report = None
        return wrapper

    if fn is not None:  # called as @helix.profile (no parens)
        return decorator(fn)
    return decorator           # called as @helix.profile(...)
