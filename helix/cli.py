"""
HelixIR command-line interface.

Commands
--------
  helix profile <script.py>            Trace + analyse a Python script
  helix benchmark <workload>           Run a built-in benchmark (transformer|diffusion)
  helix compare <workload>             Benchmark naive vs jit vs helix kernels
  helix kernels                        List available Pallas kernels
  helix serve [--port N]               Start the dashboard server
  helix hlo <script.py>                Dump StableHLO text
"""
from __future__ import annotations
import importlib.util
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app  = typer.Typer(name="helix", help="HelixIR — JAX program performance optimizer")
console = Console()


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

@app.command()
def profile(
    script: Annotated[Path, typer.Argument(help="Python script to trace")],
    fn_name: Annotated[str, typer.Option("--fn", help="Function to profile")] = "fn",
    num_devices: Annotated[int, typer.Option(help="Target device count for sharding advisor")] = 8,
    push: Annotated[bool, typer.Option(help="Push result to dashboard server")] = False,
    server: Annotated[str, typer.Option(help="Dashboard server URL")] = "http://localhost:8765",
) -> None:
    """Trace a JAX function in SCRIPT and print an optimization report."""
    import helix

    spec = importlib.util.spec_from_file_location("_helix_target", script)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)                  # type: ignore

    if not hasattr(mod, fn_name):
        console.print(f"[red]Error:[/red] No function '{fn_name}' found in {script}")
        raise typer.Exit(1)

    fn = getattr(mod, fn_name)

    if not hasattr(mod, "HELIX_ARGS"):
        console.print(
            "[yellow]Warning:[/yellow] Define HELIX_ARGS = (arg1, arg2, …) in the script "
            "to provide example inputs.  Skipping analysis."
        )
        raise typer.Exit(1)

    args = mod.HELIX_ARGS
    report = helix.analyze(fn, *args, num_devices=num_devices)

    # Rich table for pass results
    for pr in report["passes"]:
        table = Table(title=pr.pass_name, box=box.SIMPLE)
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Details")
        for rec in pr.recommendations:
            table.add_row(rec.get("type", ""), rec.get("message", ""))
        console.print(table)

    if push:
        from helix.server import push_report
        push_report(report, server_url=server)
        console.print(f"[green]✓[/green] Report pushed to {server}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

@app.command()
def benchmark(
    workload: Annotated[str, typer.Argument(help="transformer | diffusion")] = "transformer",
    batch: Annotated[int,    typer.Option(help="Batch size")]       = 2,
    seq:   Annotated[int,    typer.Option(help="Sequence length")]  = 512,
    dim:   Annotated[int,    typer.Option(help="Model dimension")]  = 1024,
    heads: Annotated[int,    typer.Option(help="Attention heads")]  = 16,
    warmup: Annotated[int,   typer.Option(help="Warmup iterations")] = 10,
    iters:  Annotated[int,   typer.Option(help="Benchmark iterations")] = 50,
    push:   Annotated[bool,  typer.Option(help="Push to dashboard")] = False,
    server: Annotated[str,   typer.Option()] = "http://localhost:8765",
) -> None:
    """Run a built-in benchmark workload."""
    import helix
    from helix.benchmark.runner import benchmark as bmark, compare

    if workload == "transformer":
        from helix.benchmark.workloads.transformer import gpt_block, make_inputs, flop_count
        inputs = make_inputs(batch=batch, seq_len=seq, d_model=dim, num_heads=heads)
        fn = lambda: gpt_block(**inputs)
        flops = flop_count(batch, seq, dim)
        label = f"GPT-block  B={batch} S={seq} D={dim} H={heads}"
    elif workload == "diffusion":
        from helix.benchmark.workloads.diffusion import unet_res_block, make_inputs, flop_count
        inputs = make_inputs(batch=batch)
        fn = lambda: unet_res_block(**inputs)
        flops = flop_count(batch, 32, 32, 256, 1024)
        label = f"UNet-block  B={batch} C=256"
    else:
        console.print(f"[red]Unknown workload:[/red] {workload}")
        raise typer.Exit(1)

    console.print(f"Benchmarking [bold]{label}[/bold] …")

    naive = bmark(fn, name="naive (no jit)", warmup=warmup, iters=iters, flops=flops)
    jit   = bmark(jax_jit(fn), name="jax.jit", warmup=warmup, iters=iters, flops=flops)

    console.print(str(naive))
    console.print(str(jit))
    console.print(compare([naive, jit], baseline_name="naive (no jit)"))

    if push:
        import httpx
        for r in [naive, jit]:
            httpx.post(f"{server}/api/benchmarks", json=r.to_dict())
        console.print(f"[green]✓[/green] Results pushed to {server}")


def jax_jit(fn):
    import jax
    return jax.jit(fn)


# ---------------------------------------------------------------------------
# compare (naive vs helix kernels)
# ---------------------------------------------------------------------------

@app.command()
def compare(
    kernel: Annotated[str, typer.Argument(help="rmsnorm | rope | attention")] = "rmsnorm",
    size:   Annotated[int, typer.Option(help="Hidden/head dim")] = 4096,
    batch:  Annotated[int, typer.Option(help="Batch size")]      = 16,
    seq:    Annotated[int, typer.Option(help="Sequence length")]  = 512,
) -> None:
    """Compare reference vs Pallas kernel performance."""
    import jax
    import jax.numpy as jnp
    from helix.benchmark.runner import benchmark as bmark, compare as cmp_table

    if kernel == "rmsnorm":
        from helix.kernels.rmsnorm import rmsnorm_ref, rmsnorm_pallas
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, size))
        w = jnp.ones(size)
        ref  = bmark(lambda: rmsnorm_ref(x, w),    name="rmsnorm_ref")
        try:
            pal  = bmark(lambda: rmsnorm_pallas(x, w), name="rmsnorm_pallas")
            results = [ref, pal]
        except RuntimeError as e:
            console.print(f"[yellow]Pallas unavailable:[/yellow] {e}")
            results = [ref]
    elif kernel == "rope":
        from helix.kernels.rope import apply_rope, rope_pallas
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, 16, size // 16))
        ref = bmark(lambda: apply_rope(x), name="rope_ref")
        try:
            pal = bmark(lambda: rope_pallas(x), name="rope_pallas")
            results = [ref, pal]
        except RuntimeError as e:
            console.print(f"[yellow]Pallas unavailable:[/yellow] {e}")
            results = [ref]
    elif kernel == "attention":
        from helix.kernels.attention import attention_ref, flash_attention
        q = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, 16, size // 16))
        k = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, 16, size // 16))
        v = jax.random.normal(jax.random.PRNGKey(2), (batch, seq, 16, size // 16))
        ref = bmark(lambda: attention_ref(q, k, v), name="attention_ref")
        try:
            fla = bmark(lambda: flash_attention(q, k, v), name="flash_attention")
            results = [ref, fla]
        except RuntimeError as e:
            console.print(f"[yellow]Pallas unavailable:[/yellow] {e}")
            results = [ref]
    else:
        console.print(f"[red]Unknown kernel:[/red] {kernel}")
        raise typer.Exit(1)

    for r in results:
        console.print(str(r))
    if len(results) > 1:
        console.print(cmp_table(results, baseline_name=results[0].name))


# ---------------------------------------------------------------------------
# kernels list
# ---------------------------------------------------------------------------

@app.command()
def kernels() -> None:
    """List available Pallas kernels."""
    try:
        import jax.experimental.pallas  # noqa: F401
        pallas_ok = True
    except ImportError:
        pallas_ok = False

    table = Table(title="HelixIR Kernels", box=box.SIMPLE_HEAVY)
    table.add_column("Kernel",    style="cyan")
    table.add_column("Pallas",    style="green" if pallas_ok else "red")
    table.add_column("Description")

    rows = [
        ("rmsnorm",   "RMSNorm with fused rsqrt + scale — saves 2 HBM round-trips"),
        ("rope",      "Rotary Position Embeddings — fused freq generation + rotation"),
        ("attention", "Flash Attention (online softmax) — O(S) instead of O(S²) HBM"),
    ]
    tag = "✓ GPU" if pallas_ok else "✗ CPU only"
    for name, desc in rows:
        table.add_row(name, tag, desc)

    console.print(table)
    if not pallas_ok:
        console.print("[yellow]Install GPU JAX for Pallas:[/yellow] pip install jax[cuda12]")


# ---------------------------------------------------------------------------
# hlo dump
# ---------------------------------------------------------------------------

@app.command()
def hlo(
    script: Annotated[Path, typer.Argument(help="Python script to lower")],
    fn_name: Annotated[str, typer.Option("--fn")] = "fn",
    out: Annotated[Optional[Path], typer.Option("--out", help="Write to file")] = None,
) -> None:
    """Dump StableHLO text for a function."""
    from helix.tracer.capture import capture_stablehlo
    import importlib.util

    spec = importlib.util.spec_from_file_location("_helix_target", script)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)                  # type: ignore

    fn   = getattr(mod, fn_name)
    args = getattr(mod, "HELIX_ARGS")
    text = capture_stablehlo(fn, *args)

    if out:
        out.write_text(text)
        console.print(f"[green]✓[/green] StableHLO written to {out}")
    else:
        console.print(text)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port")] = 8765,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code change")] = False,
) -> None:
    """Start the HelixIR dashboard API server."""
    import uvicorn
    console.print(f"[bold cyan]HelixIR[/bold cyan] dashboard → [link]http://{host}:{port}[/link]")
    uvicorn.run("helix.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
