# HelixIR

[![CI](https://github.com/ConmanTrialsOfKami/HelixIR/actions/workflows/ci.yml/badge.svg)](https://github.com/ConmanTrialsOfKami/HelixIR/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ConmanTrialsOfKami/HelixIR/blob/main/demo.ipynb)

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
| `helix.kernels.flash_attention` | online softmax, O(S) HBM | causal mask supported |

```python
from helix.kernels import rmsnorm_pallas, rope_pallas, flash_attention
```

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
git clone https://github.com/ConmanTrialsOfKami/HelixIR
cd HelixIR
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT
