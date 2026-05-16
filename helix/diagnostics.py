"""
Runtime diagnostics for JAX functions.

Six metric categories:
  1. compile_latency_ms     — wall time of first JIT call (compile + execute)
  2. run_latency_ms         — mean wall time of subsequent calls (execute only)
  3. compile_overhead_ms    — difference: how long pure compilation takes
  4. analysis_overhead_ms   — time taken by helix.analyze() itself
  5. flop / memory accuracy — HelixIR static estimates vs XLA cost_analysis()
  6. recompile_risk         — whether a shape change would trigger recompilation,
                              and how much it would cost

Usage
-----
    import helix

    diag = helix.runtime_diagnostics(my_fn, x, w)
    helix.print_diagnostics(diag)
"""
from __future__ import annotations
import statistics
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# XLA cost_analysis bridge
# ---------------------------------------------------------------------------

def _xla_cost(fn: Callable, *args) -> dict:
    """
    Ask XLA for its own FLOP and byte estimates via cost_analysis().
    Returns {} if unavailable (older JAX, CPU-only build, etc.).
    """
    try:
        analysis = jax.jit(fn).lower(*args).cost_analysis()
        if isinstance(analysis, list):
            flops      = sum(d.get("flops", 0)           for d in analysis)
            bytes_acc  = sum(d.get("bytes accessed", 0)  for d in analysis)
        elif isinstance(analysis, dict):
            flops      = analysis.get("flops", 0)
            bytes_acc  = analysis.get("bytes accessed", 0)
        else:
            return {}
        return {"flops": float(flops), "bytes_accessed": float(bytes_acc)}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Recompile risk detection
# ---------------------------------------------------------------------------

def _perturb_batch(args: tuple) -> tuple | None:
    """Return args with the first array's leading dimension doubled."""
    out = []
    found = False
    for a in args:
        if not found and hasattr(a, "shape") and a.shape:
            out.append(jnp.zeros((a.shape[0] * 2,) + a.shape[1:], dtype=a.dtype))
            found = True
        else:
            out.append(a)
    return tuple(out) if found else None


def _check_recompile(fn_jit, args: tuple, baseline_ms: float) -> tuple[bool, float]:
    """
    Call fn_jit with a perturbed batch size; measure whether JAX recompiles.

    Returns (recompile_risk: bool, recompile_overhead_ms: float).
    recompile_overhead_ms is the extra cost of the first call vs the second.
    """
    perturbed = _perturb_batch(args)
    if perturbed is None:
        return False, 0.0
    try:
        t0 = time.perf_counter()
        jax.block_until_ready(fn_jit(*perturbed))
        first_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        jax.block_until_ready(fn_jit(*perturbed))
        second_ms = (time.perf_counter() - t0) * 1e3

        overhead = max(first_ms - second_ms, 0.0)
        risk = first_ms > second_ms * 2.5 and first_ms > baseline_ms * 2
        return risk, round(overhead, 3)
    except Exception:
        return True, 0.0


# ---------------------------------------------------------------------------
# Main diagnostic function
# ---------------------------------------------------------------------------

def runtime_diagnostics(
    fn: Callable,
    *args: Any,
    warmup: int = 3,
    iters: int = 20,
    device: str | None = None,
) -> dict:
    """
    Collect runtime diagnostics for fn(*args).

    Parameters
    ----------
    fn      : JAX function to profile.
    *args   : Example inputs (arrays with representative shapes).
    warmup  : Additional warm-up calls after the initial compile call.
    iters   : Number of timed steady-state iterations.
    device  : Device name for roofline (e.g. 'A100'). Auto-detected if None.

    Returns
    -------
    dict with keys:

    Latency
      compile_latency_ms    First JIT call (compile + execute)
      run_latency_ms        Mean of steady-state calls
      compile_overhead_ms   compile_latency - run_latency
      jit_speedup           compile_latency / run_latency

    Analysis cost
      analysis_overhead_ms  Time taken by helix.analyze()

    FLOP accuracy
      estimated_flops       HelixIR static FLOP count
      xla_flops             XLA cost_analysis FLOPs (0 if unavailable)
      flop_accuracy_pct     Agreement with XLA: 100% = exact match

    Memory accuracy
      estimated_bytes       HelixIR static byte count
      xla_bytes             XLA cost_analysis bytes (0 if unavailable)
      memory_accuracy_pct   Agreement with XLA: 100% = exact match

    Roofline
      roofline_compute_pct  % of classified ops that are compute-bound
      roofline_bw_pct       % of classified ops that are bandwidth-bound
      ridge_point           FLOP/byte crossover for the target device

    Recompile
      recompile_risk            True if shape changes would trigger recompilation
      recompile_overhead_ms     Cost paid per new shape (0 if no recompile detected)
      recompile_breakeven_calls Calls needed at new shape to amortize recompile cost
      recompile_advice          List of actionable fixes ranked by impact
    """
    from . import analyze  # avoid circular import

    # ── 1 & 2: Compile + run latency ─────────────────────────────────────────
    fn_jit = jax.jit(fn)  # fresh trace cache

    t0 = time.perf_counter()
    jax.block_until_ready(fn_jit(*args))
    compile_latency_ms = (time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        jax.block_until_ready(fn_jit(*args))

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn_jit(*args))
        times.append((time.perf_counter() - t0) * 1e3)

    run_latency_ms    = statistics.mean(times)
    compile_overhead  = max(compile_latency_ms - run_latency_ms, 0.0)
    jit_speedup       = compile_latency_ms / run_latency_ms if run_latency_ms > 0 else 1.0

    # ── 3: Analysis overhead ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    report = analyze(fn, *args, device=device)
    analysis_overhead_ms = (time.perf_counter() - t0) * 1e3

    # ── 4: FLOP + memory accuracy ─────────────────────────────────────────────
    xla             = _xla_cost(fn, *args)
    est_flops       = report["total_flops"]
    est_bytes       = report["total_bytes"]
    xla_flops       = xla.get("flops", 0)
    xla_bytes       = xla.get("bytes_accessed", 0)

    def _accuracy(est: float, ref: float) -> float | None:
        if ref <= 0:
            return None
        return round(max(0.0, (1 - abs(est - ref) / ref) * 100), 1)

    # ── 5: Roofline ───────────────────────────────────────────────────────────
    r = report["roofline"]
    n_classified = len(r.compute_bound_ops) + len(r.bandwidth_bound_ops)
    compute_pct  = len(r.compute_bound_ops) / n_classified * 100 if n_classified else 0.0
    bw_pct       = len(r.bandwidth_bound_ops) / n_classified * 100 if n_classified else 0.0

    # ── 6: Recompile risk + advice ────────────────────────────────────────────
    recompile_risk, recompile_overhead_ms = _check_recompile(fn_jit, args, run_latency_ms)

    breakeven = (
        int(recompile_overhead_ms / run_latency_ms) + 1
        if recompile_risk and run_latency_ms > 0
        else 0
    )
    advice = _recompile_advice(args, recompile_risk, compile_overhead, run_latency_ms)

    return {
        # Latency
        "compile_latency_ms":         round(compile_latency_ms, 3),
        "run_latency_ms":             round(run_latency_ms, 3),
        "compile_overhead_ms":        round(compile_overhead, 3),
        "jit_speedup":                round(jit_speedup, 1),
        # Analysis cost
        "analysis_overhead_ms":       round(analysis_overhead_ms, 3),
        # FLOP accuracy
        "estimated_flops":            est_flops,
        "xla_flops":                  int(xla_flops),
        "flop_accuracy_pct":          _accuracy(est_flops, xla_flops),
        # Memory accuracy
        "estimated_bytes":            est_bytes,
        "xla_bytes":                  int(xla_bytes),
        "memory_accuracy_pct":        _accuracy(est_bytes, xla_bytes),
        # Roofline
        "roofline_compute_pct":       round(compute_pct, 1),
        "roofline_bw_pct":            round(bw_pct, 1),
        "ridge_point":                r.ridge_point,
        # Recompile
        "recompile_risk":             recompile_risk,
        "recompile_overhead_ms":      recompile_overhead_ms,
        "recompile_breakeven_calls":  breakeven,
        "recompile_advice":           advice,
        # Downstream use
        "_report":                    report,
    }


# ---------------------------------------------------------------------------
# Recompile advice
# ---------------------------------------------------------------------------

def _recompile_advice(
    args: tuple,
    risk: bool,
    compile_overhead_ms: float,
    run_latency_ms: float,
) -> list[str]:
    """
    Return ranked, actionable advice for reducing recompilation cost.
    Always returned even when risk=False (preventive guidance).
    """
    tips: list[str] = []

    if risk:
        # How bad is it?
        ratio = compile_overhead_ms / run_latency_ms if run_latency_ms > 0 else 0
        tips.append(
            f"Pad inputs to a fixed shape so every call hits the same JIT trace. "
            f"Each new shape pays ~{compile_overhead_ms:.0f} ms overhead "
            f"({ratio:.0f}× a steady-state call)."
        )
        tips.append(
            "Use jax.jit(fn, static_argnums=...) for integer / shape arguments "
            "that change but should not retrace array-valued args."
        )
        # Check if any arg has a small leading dim — likely a batch size
        for i, a in enumerate(args):
            if hasattr(a, "shape") and a.shape and a.shape[0] <= 8:
                tips.append(
                    f"Arg {i} has batch size {a.shape[0]}. If batch size varies, "
                    f"use jax.vmap over a fixed single-item batch instead of "
                    f"calling with different batch sizes."
                )
                break
        tips.append(
            "For dynamic sequence lengths, pad to the next power-of-two bucket "
            "(128, 256, 512, …) — limits distinct shapes to O(log S) traces."
        )
    else:
        tips.append(
            "No recompile detected for the tested shape perturbation. "
            "Keep inputs at consistent shapes to maintain this behaviour."
        )
        if compile_overhead_ms > run_latency_ms * 5:
            tips.append(
                f"Compile overhead is {compile_overhead_ms:.0f} ms vs "
                f"{run_latency_ms:.1f} ms per run. Ensure the function is "
                f"called enough times (>{int(compile_overhead_ms/run_latency_ms)}) "
                f"to amortize compilation before benchmarking."
            )

    return tips


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_diagnostics(diag: dict, fn_name: str = "fn") -> None:
    """Print a formatted diagnostics report."""
    def _pct(v) -> str:
        return f"{v:.1f}%" if v is not None else "n/a (XLA unavailable)"

    def _ms(v) -> str:
        return f"{v:.3f} ms"

    w = 38
    sep = "═" * (w + 20)

    print(f"\n{sep}")
    print(f"  HelixIR Runtime Diagnostics · {fn_name}")
    print(sep)

    print(f"\n  {'── Latency':─<{w+18}}")
    print(f"  {'Compile (first JIT call)':<{w}} {_ms(diag['compile_latency_ms'])}")
    print(f"  {'Run (steady-state mean)':<{w}} {_ms(diag['run_latency_ms'])}")
    print(f"  {'Compile overhead':<{w}} {_ms(diag['compile_overhead_ms'])}")
    print(f"  {'JIT speedup':<{w}} {diag['jit_speedup']:.1f}×")

    print(f"\n  {'── Analysis cost':─<{w+18}}")
    print(f"  {'helix.analyze() overhead':<{w}} {_ms(diag['analysis_overhead_ms'])}")

    print(f"\n  {'── FLOP accuracy':─<{w+18}}")
    print(f"  {'HelixIR estimate':<{w}} {diag['estimated_flops'] / 1e9:.3f} GFLOPs")
    if diag["xla_flops"] > 0:
        print(f"  {'XLA cost_analysis':<{w}} {diag['xla_flops'] / 1e9:.3f} GFLOPs")
    print(f"  {'Agreement with XLA':<{w}} {_pct(diag['flop_accuracy_pct'])}")

    print(f"\n  {'── Memory accuracy':─<{w+18}}")
    print(f"  {'HelixIR estimate':<{w}} {diag['estimated_bytes'] / 1e6:.2f} MB")
    if diag["xla_bytes"] > 0:
        print(f"  {'XLA cost_analysis':<{w}} {diag['xla_bytes'] / 1e6:.2f} MB")
    print(f"  {'Agreement with XLA':<{w}} {_pct(diag['memory_accuracy_pct'])}")

    print(f"\n  {'── Roofline':─<{w+18}}")
    print(f"  {'Ridge point':<{w}} {diag['ridge_point']:.1f} FLOP/byte")
    print(f"  {'Compute-bound ops':<{w}} {diag['roofline_compute_pct']:.1f}%")
    print(f"  {'Bandwidth-bound ops':<{w}} {diag['roofline_bw_pct']:.1f}%")

    print(f"\n  {'── Recompile risk':─<{w+18}}")
    risk_str = "YES" if diag["recompile_risk"] else "no"
    print(f"  {'Shape-change triggers recompile':<{w}} {risk_str}")
    if diag["recompile_risk"]:
        print(f"  {'Overhead per new shape':<{w}} {_ms(diag['recompile_overhead_ms'])}")
        print(f"  {'Break-even calls at new shape':<{w}} {diag['recompile_breakeven_calls']} calls")

    print(f"\n  {'── Recompile advice':─<{w+18}}")
    for i, tip in enumerate(diag["recompile_advice"], 1):
        # Word-wrap at 74 chars
        words = tip.split()
        line, lines = [], []
        for word in words:
            if len(" ".join(line + [word])) > 74:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        print(f"  {i}. {lines[0]}")
        for cont in lines[1:]:
            print(f"     {cont}")

    print(f"\n{sep}\n")
