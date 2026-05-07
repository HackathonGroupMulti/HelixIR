"""
Backward pass analyser.

Most JAX profiling tools only look at the forward pass.  The backward pass
(VJP) typically costs 2-3× as much and contains its own fusion opportunities,
memory bottlenecks, and rematerialisation trade-offs that are invisible to
forward-only analysis.

helix.analyze_backward traces the *joint* forward+backward computation as a
single JAXPR (which is what jax.grad actually compiles) and runs the full
HelixIR analysis pipeline over it.

Usage
-----
    import helix

    fwd_report = helix.analyze(my_fn, x, w)
    bwd_report = helix.analyze_backward(my_fn, x, w)

    # Or both at once:
    full = helix.analyze_full(my_fn, x, w)
    print(full['backward']['roofline'].total_flops / full['forward']['roofline'].total_flops)
    # → typically 2-3x
"""
from __future__ import annotations
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .tracer.capture import capture
from .analyzer.memory import analyze_memory
from .analyzer.bottleneck import analyze_bottleneck, RooflineResult
from .passes.fusion_advisor import FusionAdvisorPass
from .passes.checkpoint_advisor import CheckpointAdvisorPass
from .passes.sharding_advisor import ShardingAdvisorPass


def _make_bwd_fn(fn: Callable) -> Callable:
    """
    Return a function that computes the VJP of fn.

    The returned function traces as a single JAXPR covering both the
    forward computation *and* the backward (gradient) computation,
    which is exactly what jax.grad compiles under the hood.
    """
    def fwd_and_bwd(*args):
        primals, vjp_fn = jax.vjp(fn, *args)
        # Create unit cotangent matching the primal output structure
        ct = jax.tree_util.tree_map(jnp.ones_like, primals)
        return vjp_fn(ct)

    return fwd_and_bwd


def analyze_backward(
    fn: Callable,
    *args: Any,
    num_devices: int = 8,
    device: str | None = None,
) -> dict:
    """
    Analyse the backward pass (VJP) of fn(*args).

    Returns the same dict structure as helix.analyze():
        graph, roofline, passes, num_ops, total_flops, total_bytes

    Notes
    -----
    - The JAXPR traces the *joint* fwd+bwd computation, so total_flops here
      already includes the forward cost.  To get backward-only cost, subtract
      helix.analyze(fn, *args)['total_flops'].
    - Requires fn to be differentiable (no integer ops at the top level).
    """
    bwd_fn = _make_bwd_fn(fn)

    try:
        _, graph = capture(bwd_fn, *args)
    except Exception as exc:
        raise RuntimeError(
            f"Could not trace backward pass for {getattr(fn, '__name__', fn)}: {exc}\n"
            "Make sure the function is differentiable and all args are floating-point."
        ) from exc

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


def analyze_full(
    fn: Callable,
    *args: Any,
    num_devices: int = 8,
    device: str | None = None,
) -> dict:
    """
    Run both forward and backward analysis and return a combined report.

    Returns
    -------
    {
      'forward':  { graph, roofline, passes, num_ops, total_flops, total_bytes },
      'backward': { graph, roofline, passes, num_ops, total_flops, total_bytes },
      'bwd_fwd_flop_ratio': float,   # typically 2-3×
      'bwd_fwd_byte_ratio': float,
    }
    """
    from . import analyze  # avoid circular import at module level

    fwd = analyze(fn, *args, num_devices=num_devices, device=device)
    bwd = analyze_backward(fn, *args, num_devices=num_devices, device=device)

    fwd_flops = max(fwd["total_flops"], 1)
    fwd_bytes = max(fwd["total_bytes"], 1)

    return {
        "forward":  fwd,
        "backward": bwd,
        "bwd_fwd_flop_ratio": bwd["total_flops"] / fwd_flops,
        "bwd_fwd_byte_ratio": bwd["total_bytes"] / fwd_bytes,
    }


def print_full_report(full: dict, fn_name: str = "fn") -> None:
    """Print a side-by-side summary of forward vs backward analysis."""
    fwd = full["forward"]["roofline"]
    bwd = full["backward"]["roofline"]

    print(f"\n{'═'*60}")
    print(f"  HelixIR Full Analysis: {fn_name}")
    print(f"{'═'*60}")
    print(f"  {'':30} {'Forward':>12}  {'Backward':>12}")
    print(f"  {'─'*56}")
    print(f"  {'Ops':30} {full['forward']['num_ops']:>12}  {full['backward']['num_ops']:>12}")
    print(f"  {'GFLOPs':30} {fwd.total_flops/1e9:>12.2f}  {bwd.total_flops/1e9:>12.2f}")
    print(f"  {'MB total':30} {fwd.total_bytes/1e6:>12.1f}  {bwd.total_bytes/1e6:>12.1f}")
    print(f"  {'Compute-bound ops':30} {len(fwd.compute_bound_ops):>12}  {len(bwd.compute_bound_ops):>12}")
    print(f"  {'Bandwidth-bound ops':30} {len(fwd.bandwidth_bound_ops):>12}  {len(bwd.bandwidth_bound_ops):>12}")
    print(f"\n  Backward / Forward FLOP ratio : {full['bwd_fwd_flop_ratio']:.2f}×")
    print(f"  Backward / Forward byte ratio : {full['bwd_fwd_byte_ratio']:.2f}×")
    print(f"{'═'*60}\n")
