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
        except Exception as e:
            console.print(f"[yellow]Pallas unavailable:[/yellow] {type(e).__name__}: {e}")
            results = [ref]
    elif kernel == "rope":
        from helix.kernels.rope import apply_rope, rope_pallas
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, 16, size // 16))
        ref = bmark(lambda: apply_rope(x), name="rope_ref")
        try:
            pal = bmark(lambda: rope_pallas(x), name="rope_pallas")
            results = [ref, pal]
        except Exception as e:
            console.print(f"[yellow]Pallas unavailable:[/yellow] {type(e).__name__}: {e}")
            results = [ref]
    elif kernel == "attention":
        from helix.kernels.attention import attention_ref, flash_attention
        q = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, 16, size // 16))
        k = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, 16, size // 16))
        v = jax.random.normal(jax.random.PRNGKey(2), (batch, seq, 16, size // 16))
        ref = bmark(lambda: attention_ref(q, k, v), name="attention_ref")
        results = [ref]
        try:
            fla = bmark(lambda: flash_attention(q, k, v), name="flash_attention (pallas)")
            results.append(fla)
        except Exception as e:
            # Pallas raises RuntimeError when absent and ValueError on CPU / cc<8.0.
            console.print(f"[yellow]Pallas unavailable:[/yellow] {type(e).__name__}: {e}")
        from helix.kernels.attention_triton import triton_available
        if triton_available():
            # Triton uses torch tensors, so time it with a torch loop (the JAX
            # bmark harness can't jit a torch op) and wrap in a BenchmarkResult.
            import torch, statistics, time as _time
            from helix.benchmark.runner import BenchmarkResult
            from helix.kernels.attention_triton import flash_attention_triton
            tq, tk, tv = (torch.randn(batch, seq, 16, size // 16, device="cuda",
                                      dtype=torch.float16) for _ in range(3))
            for _ in range(10):
                flash_attention_triton(tq, tk, tv); torch.cuda.synchronize()
            ts = []
            for _ in range(50):
                t0 = _time.perf_counter()
                flash_attention_triton(tq, tk, tv); torch.cuda.synchronize()
                ts.append((_time.perf_counter() - t0) * 1e3)
            results.append(BenchmarkResult(
                name="flash_attention (triton)", mean_ms=statistics.mean(ts),
                std_ms=statistics.stdev(ts), min_ms=min(ts), max_ms=max(ts),
                iterations=len(ts)))
        else:
            console.print("[yellow]Triton unavailable:[/yellow] needs torch+triton on CUDA")
    else:
        console.print(f"[red]Unknown kernel:[/red] {kernel}")
        raise typer.Exit(1)

    for r in results:
        console.print(str(r))
    if len(results) > 1:
        console.print(cmp_table(results, baseline_name=results[0].name))


# ---------------------------------------------------------------------------
# infer — LLM inference roofline (KV-cache, prefill vs decode)
# ---------------------------------------------------------------------------

@app.command()
def infer(
    model:  Annotated[str, typer.Argument(help="Preset: llama3-8b, llama2-70b, mistral-7b, …")] = "llama3-8b",
    batch:  Annotated[int, typer.Option(help="Batch size")]        = 16,
    prompt: Annotated[int, typer.Option(help="Prompt length")]     = 2048,
    gen:    Annotated[int, typer.Option(help="Tokens to generate")] = 256,
    device: Annotated[Optional[str], typer.Option(help="GPU (A100/H100/…); auto if unset")] = None,
    dtype:  Annotated[int, typer.Option(help="Bytes per weight/KV element (2=fp16, 1=fp8)")] = 2,
) -> None:
    """Analyze LLM inference: KV cache, prefill vs decode roofline, serving."""
    from helix.inference import ModelConfig, analyze_inference, print_inference_report, available_presets
    if model not in available_presets():
        console.print(f"[red]Unknown preset:[/red] {model}. Available: {available_presets()}")
        raise typer.Exit(1)
    cfg = ModelConfig.from_preset(model, dtype_bytes=dtype)
    report = analyze_inference(cfg, batch=batch, prompt_len=prompt, gen_len=gen, device=device)
    print_inference_report(report)


# ---------------------------------------------------------------------------
# infer-sweep — grid over batch × context, table/markdown/csv out
# ---------------------------------------------------------------------------

@app.command(name="infer-sweep")
def infer_sweep(
    model:   Annotated[str, typer.Argument(help="Preset model name")] = "llama3-8b",
    batches: Annotated[str, typer.Option(help="Comma-separated batch sizes")] = "1,8,32,64",
    prompts: Annotated[str, typer.Option(help="Comma-separated prompt lengths")] = "2048",
    gen:     Annotated[int, typer.Option(help="Tokens to generate")] = 128,
    device:  Annotated[Optional[str], typer.Option(help="GPU name; auto if unset")] = None,
    fmt:     Annotated[str, typer.Option("--format", help="table | markdown | csv")] = "table",
) -> None:
    """Sweep batch × prompt-length and report KV/throughput/offload per config."""
    from helix.inference import (
        ModelConfig, available_presets, sweep_inference,
        print_sweep, sweep_to_markdown, sweep_to_csv,
    )
    if model not in available_presets():
        console.print(f"[red]Unknown preset:[/red] {model}. Available: {available_presets()}")
        raise typer.Exit(1)
    cfg = ModelConfig.from_preset(model)
    b = [int(x) for x in batches.split(",") if x.strip()]
    p = [int(x) for x in prompts.split(",") if x.strip()]
    rows = sweep_inference(cfg, batches=b, prompt_lens=p, gen_len=gen, device=device)

    if fmt == "markdown":
        print(sweep_to_markdown(rows))
    elif fmt == "csv":
        print(sweep_to_csv(rows))
    else:
        print_sweep(rows, cfg_name=model, device=device or "auto")


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

    from helix.kernels.attention_triton import triton_available
    triton_ok = triton_available()

    rows = [
        ("rmsnorm",          "pallas", "RMSNorm with fused rsqrt + scale — saves 2 HBM round-trips"),
        ("rope",             "pallas", "Rotary Position Embeddings — fused freq generation + rotation"),
        ("attention",        "pallas", "Flash Attention (online softmax) — O(S) instead of O(S²) HBM"),
        ("attention_triton", "triton", "Flash Attention forward in Triton — Pallas counterpart"),
    ]
    for name, backend, desc in rows:
        ok = triton_ok if backend == "triton" else pallas_ok
        table.add_row(name, "✓ GPU" if ok else "✗ CPU only", desc)

    console.print(table)
    if not pallas_ok:
        console.print("[yellow]Install GPU JAX for Pallas:[/yellow] pip install jax[cuda12]")
    if not triton_ok:
        console.print("[yellow]Install Triton:[/yellow] pip install torch triton (Linux + CUDA)")


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
