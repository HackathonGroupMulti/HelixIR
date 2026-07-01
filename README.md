# HelixIR

[![CI](https://github.com/HackathonGroupMulti/HelixIR/actions/workflows/ci.yml/badge.svg)](https://github.com/HackathonGroupMulti/HelixIR/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HackathonGroupMulti/HelixIR/blob/main/demo.ipynb)

**Automatic performance optimizer for JAX programs.**

HelixIR traces your JAX function through its compiler intermediate representation, classifies every op on a hardware roofline model, and gives you three things: fusion/checkpointing/sharding recommendations, Megatron-style sharding code you can paste directly into a training script, and a breakdown of backward-pass FLOP cost. Everything works offline — no GPU required for analysis.

```
pip install helixir                   # CPU analysis only
pip install "helixir[gpu]"            # + Pallas kernels (Ampere A100/H100)
pip install "helixir[jupyter]"        # + %helix magic
```

---

## What it does

| Layer | What happens |
|---|---|
| **Trace** | `jax.make_jaxpr` captures the full computation graph including shapes, dtypes, and primitive ops |
| **Analyze** | Per-op FLOP and byte counts feed a roofline model; ops are labelled compute-bound or bandwidth-bound |
| **Advise** | Three passes emit ranked recommendations: XLA fusion groups, activation checkpointing targets, and sharding strategy |
| **Generate** | `generate_sharding` emits copy-paste device-placement code using `jax.device_put` + `NamedSharding` |
| **Backward** | `analyze_backward` traces `jax.vjp` to report the true bwd/fwd FLOP ratio |
| **Inference** | `analyze_inference` models LLM serving: KV-cache footprint, multi-tier offload, and prefill (compute-bound) vs. decode (memory-bound) roofline classification |

---

## Quickstart

```python
import helix
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
x  = jax.random.normal(key, (4, 512, 1024))
wq = jax.random.normal(key, (1024, 1024))

@helix.profile
def my_attn(x, w):
    return jnp.einsum("bsd,dh->bsh", x, w)

out = my_attn(x, wq)   # analysis prints on first call
```

```
─────────────────── HelixIR · my_attn ───────────────────
  Device       Unknown (H100 profile used)
  Ops          3
  Total FLOPs  4.29 GFLOPs
  Total bytes  20.97 MB
  Ridge point  204.8 FLOP/byte  (1 compute-bound | 2 bandwidth-bound)

  FusionAdvisorPass   3 elementwise ops in 1 fusable group (~0.1 MB savings)
  CheckpointAdvisorPass  no tensors >10 MB detected
  ShardingAdvisorPass  data-parallel over batch; tensor-parallel over hidden
─────────────────────────────────────────────────────────
```

---

## Auto-sharding code generation

HelixIR reads the JAXPR to classify every argument as an activation, a column-parallel weight (fan-out), or a row-parallel weight (fan-in), then emits Megatron-style device-placement code:

```python
plan = helix.generate_sharding(
    transformer_tiny, x, wq, wk, wv, wo, w1, w2, norm,
    mesh_shape=(2, 4),          # 2 data-parallel × 4 model-parallel
    axis_names=("batch", "model"),
    arg_names=["x","wq","wk","wv","wo","w1","w2","norm"],
)
print(plan.code)
```

Generated output (paste directly into your training script):

```python
# ──────────────────────────────────────────────────────
# HelixIR Auto-Generated Sharding Plan
# Function : transformer_tiny
# Mesh     : 2×4  (batch × model)
# Strategy : Megatron-style column–row tensor parallelism
# ──────────────────────────────────────────────────────
import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = np.array(jax.devices())
mesh = Mesh(devices[:8].reshape(2, 4), axis_names=('batch', 'model'))

x    = jax.device_put(x,    NamedSharding(mesh, P('batch', None, None)))  # activation
wq   = jax.device_put(wq,   NamedSharding(mesh, P(None, 'model')))        # weight_col — column-parallel
wk   = jax.device_put(wk,   NamedSharding(mesh, P(None, 'model')))        # weight_col — column-parallel
wv   = jax.device_put(wv,   NamedSharding(mesh, P(None, 'model')))        # weight_col — column-parallel
wo   = jax.device_put(wo,   NamedSharding(mesh, P('model', None)))        # weight_row — row-parallel
w1   = jax.device_put(w1,   NamedSharding(mesh, P(None, 'model')))        # weight_col — column-parallel
w2   = jax.device_put(w2,   NamedSharding(mesh, P('model', None)))        # weight_row — row-parallel
norm = jax.device_put(norm,  NamedSharding(mesh, P(None,)))               # replicated — 1-D weight
```

---

## Backward pass analysis

```python
full = helix.analyze_full(transformer_tiny, x, wq, wk, wv, wo, w1, w2, norm)
print(f"Forward  FLOPs : {full['forward']['total_flops']/1e9:.2f} G")
print(f"Backward FLOPs : {full['backward']['total_flops']/1e9:.2f} G")
print(f"bwd/fwd ratio  : {full['bwd_fwd_flop_ratio']:.2f}×")
helix.print_full_report(full)
```

Measured on a LLaMA-style GPT block (B=4, S=512, D=1024):

```
Forward  FLOPs : 3.45 G
Backward FLOPs : 12.47 G
bwd/fwd ratio  : 3.61×
```

This matches the theoretical ~3× rule for transformer backward passes and gives you an exact number for gradient-accumulation and memory budgeting decisions.

---

## LLM inference analysis (KV-cache, prefill vs. decode)

Serving an autoregressive model is bottlenecked by the **KV cache**, not the weights, and the two phases sit on opposite sides of the roofline: prefill is compute-bound, decode is memory-bandwidth-bound. `analyze_inference` models this analytically — no GPU required — including a FlexKV-style multi-tier offload plan when the cache overflows HBM:

```python
from helix.inference import ModelConfig, analyze_inference, print_inference_report

cfg = ModelConfig.from_preset("llama3-8b")          # GQA-aware; also 7b/13b/70b, mistral, qwen2
report = analyze_inference(cfg, batch=64, prompt_len=4096, gen_len=256, device="A100")
print_inference_report(report)
```

```
── Prefill────────────────────────────────
  Arithmetic intensity   83739.5 FLOP/byte
  Ridge point            156 FLOP/byte
  Classification         COMPUTE-BOUND

── Decode─────────────────────────────────
  Arithmetic intensity   19.8 FLOP/byte
  Classification         MEMORY-BOUND

── Serving────────────────────────────────
  Decode throughput      2435 tok/s

  Advice
  1. Decode is memory-bound (AI 19.8 < ridge 156). Extra FLOPs won't help;
     the wins are KV-cache reuse (prefix/paged), fp8 KV, and batching.
  2. Separate prefill and decode onto different schedules/devices — the basis
     of disaggregated / chunked-prefill serving in vLLM and SGLang.
```

When the cache exceeds free HBM, `plan_kv_offload` spills it across GPU → CPU → NVMe → remote tiers and reports the blended read bandwidth and the resulting decode slowdown — the quantity a KV-offload framework has to minimize.

### Batch × context sweep

`helix infer-sweep` (or `sweep_inference`) grids over batch and prompt length so you can see exactly where the KV cache overflows HBM and decode throughput collapses:

```bash
helix infer-sweep llama3-8b --batches 1,16,64,256 --prompts 4096 --device A100
```

| B | prompt | KV GB | HBM | TPOT ms | tok/s | slow× |
|---|---|---|---|---|---|---|
| 1 | 4096 | 0.55 | yes | 8.31 | 120 | 1.00 |
| 16 | 4096 | 8.86 | yes | 12.46 | 1284 | 1.00 |
| 64 | 4096 | 35.43 | yes | 25.75 | 2486 | 1.00 |
| 256 | 4096 | 141.73 | **no** | 1489.80 | 172 | **20.91** |

Throughput scales with batch (decode is memory-bound, so larger batches are near-free) right up to batch 256, where the 142 GB cache overflows the A100's 80 GB and spills to CPU/SSD — a 20.9× decode slowdown. `--format markdown|csv` emits the table for reports or CI.

### Serving benchmark (vLLM, with analytic fallback)

`serving_benchmark` measures TTFT / TPOT under vLLM when a GPU and model are available, and validates the analytic estimate against the measured numbers. On a CPU box it degrades cleanly to the analytic estimate:

```python
from helix.inference import serving_benchmark, print_serving_result, ModelConfig

res = serving_benchmark(ModelConfig.from_preset("mistral-7b"),
                        batch=32, prompt_len=2048, gen_len=128,
                        device="H100", model="mistralai/Mistral-7B-v0.1")
print_serving_result(res)   # backend="vllm" if installed, else "analytic"
```

---

## Jupyter magic

```python
%load_ext helix.jupyter
%helix transformer_tiny x wq wk wv wo w1 w2 norm --bwd --devices 8
```

Renders inline SVG op-graph, log-log roofline chart, and HTML recommendations table — no server required.

---

## CLI

```bash
helix profile   examples/profile_demo.py
helix benchmark examples/profile_demo.py --iters 100
helix compare   examples/profile_demo.py
helix kernels                             # list available GPU kernels
helix hlo       examples/profile_demo.py  # dump StableHLO
helix serve                               # start FastAPI + WebSocket dashboard
```

---

## GPU kernels (Ampere A100/H100 only)

Three fused Pallas kernels ship with HelixIR. All require `jax[cuda12]` and compute capability ≥ 8.0.

| Kernel | What it fuses | Notes |
|---|---|---|
| `helix.kernels.rmsnorm_pallas` | variance + rsqrt + scale → 1 HBM pass | custom VJP for correct gradients |
| `helix.kernels.rope_pallas` | freq table + rotation → 1 HBM pass | avoids intermediate half-tensors |
| `helix.kernels.flash_attention` | online softmax, O(S) HBM | Pallas; causal mask supported |
| `helix.kernels.flash_attention_triton` | online softmax, O(S) HBM | **Triton** counterpart (torch); causal supported |

```python
from helix.kernels import rmsnorm_pallas, rope_pallas, flash_attention
from helix.kernels import flash_attention_triton   # Triton (torch+CUDA)
```

The Pallas and Triton flash-attention kernels implement the same online-softmax algorithm in the two languages GPU operators are actually written in — `helix compare attention` benchmarks both against the reference on your device.

> **T4 note:** T4 (compute capability 7.5) does not support Pallas/Triton. Use the reference implementations (`rmsnorm_ref`, `rope_ref`, `attention_ref`) or upgrade to A100/H100.

---

## Real benchmarks — NVIDIA T4 (Colab)

All numbers measured with `jax.block_until_ready()`, 10 warm-up + 50 timed iterations.

### RMSNorm — memory-bound as expected

| variant | latency |
|---|---|
| `rmsnorm_ref` (no JIT) | 0.994 ms |
| `rmsnorm` (JIT) | 1.000 ms |

Memory-bound ops see no speedup from JIT alone — the bottleneck is HBM bandwidth, not kernel launch overhead. The Pallas fused kernel eliminates two extra memory round-trips and shows improvement on Ampere+.

### Attention — quadratic scaling confirmed

Input: B=2, H=8, D=64. Sequence length swept from 128 → 1024.

| seq_len | latency |
|---|---|
| 128 | 0.609 ms |
| 256 | 1.479 ms |
| 512 | 3.557 ms |
| 1024 | 12.286 ms |

Scaling factor from S=128→1024 (8×): **20.2×** — consistent with O(S²) attention cost. Flash Attention reduces this to O(S) on Ampere+.

### GPT block — end-to-end throughput

LLaMA-style block (RMSNorm → QKV → attention → out-proj → FFN gate/up/down). B=4, S=512, D=1024, H=16.

| metric | value |
|---|---|
| Latency | 16.456 ms |
| Throughput | **3.26 TFLOPS** |
| T4 peak (FP16) | 65 TFLOPS |
| Utilization | ~5% |

Low utilization is expected: this is a single-batch inference run, not a fused training loop. Enabling Flash Attention, gradient checkpointing, and multi-device sharding (all of which HelixIR can generate) are the next levers.

### Runtime diagnostics — transformer attention block (T4)

`helix.runtime_diagnostics` on a 5-op QKV + attention + out-proj block (B=4, S=512, D=1024):

| metric | value |
|---|---|
| Compile latency (first JIT call) | 643.3 ms |
| Steady-state run latency | 7.2 ms |
| **JIT speedup** | **89×** |
| `helix.analyze()` overhead | 1.0 ms |
| HelixIR FLOP estimate | 23.6 GFLOPs |
| HelixIR memory estimate | 167.9 MB |
| Compute-bound ops | 31.2% |
| Bandwidth-bound ops | 68.8% |
| Recompile overhead (new shape) | 579.7 ms |
| Break-even calls at new shape | 81 calls |

The 89× JIT speedup means a single compile amortizes across any realistic inference run. The break-even of 81 calls quantifies exactly when padding inputs to a fixed shape is cheaper than eating a recompile — HelixIR surfaces this directly so you don't have to guess.

---

## Architecture

```
helix/
├── tracer/
│   ├── capture.py       jax.make_jaxpr + StableHLO dump + XLA cost model
│   └── graph.py         OpGraph / OpNode / OpEdge dataclasses
├── analyzer/
│   ├── memory.py        per-op FLOP + byte estimation
│   ├── bottleneck.py    roofline model — ridge point, compute/BW classification
│   └── fusion.py        XLA fusion group assignment
├── passes/
│   ├── fusion_advisor.py       XLA fusion hints with MB savings
│   ├── checkpoint_advisor.py   jax.checkpoint targets (tensors >10 MB)
│   └── sharding_advisor.py     data-parallel / tensor-parallel / FSDP advice
├── kernels/
│   ├── rmsnorm.py   reference + custom VJP + Pallas fused kernel
│   ├── rope.py      reference + Pallas fused kernel
│   └── attention.py reference + Flash Attention (Pallas, online softmax)
├── benchmark/
│   ├── runner.py            benchmark() / compare() with block_until_ready
│   └── workloads/           GPT block + UNet residual block fixtures
├── sharding.py     generate_sharding() → ShardingPlan with .code
├── backward.py     analyze_backward() / analyze_full() via jax.vjp
├── jupyter.py      %helix magic (SVG graph + roofline + HTML table)
├── server.py       FastAPI + WebSocket
└── cli.py          Typer CLI
```

---

## Development

```bash
git clone https://github.com/HackathonGroupMulti/HelixIR
cd HelixIR
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT
