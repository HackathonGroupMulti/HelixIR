"""
Roofline analysis of the two phases of LLM inference.

The whole point: **prefill and decode sit on opposite sides of the roofline.**

  * Prefill processes the entire prompt at once, so weight reads amortise over
    many tokens and the phase is *compute-bound* — it wants FLOPs.
  * Decode generates one token per step while re-reading every weight and the
    entire KV cache, so arithmetic intensity collapses and the phase is
    *memory-bandwidth-bound* — it wants bandwidth, not FLOPs.

``analyze_inference`` estimates FLOPs, bytes, arithmetic intensity, and the
roofline-predicted time for each phase, folds in KV-cache offload cost from
``kvcache.py``, and reports TTFT / decode throughput / tokens-per-second plus
ranked serving advice.  Everything is analytical — no GPU required.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .kvcache import (
    ModelConfig, kv_footprint, default_tiers, plan_kv_offload, OffloadPlan,
)


@dataclass
class PhaseResult:
    name: str                 # "prefill" | "decode"
    flops: int
    bytes: int
    arithmetic_intensity: float
    ridge_point: float
    is_compute_bound: bool
    time_ms: float

    @property
    def bound(self) -> str:
        return "compute-bound" if self.is_compute_bound else "memory-bound"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "flops": self.flops,
            "bytes": self.bytes,
            "arithmetic_intensity": self.arithmetic_intensity,
            "ridge_point": self.ridge_point,
            "is_compute_bound": self.is_compute_bound,
            "bound": self.bound,
            "time_ms": self.time_ms,
        }


def _layer_linear_flops(cfg: ModelConfig) -> int:
    """Per-token, per-layer FLOPs for the linear (matmul) projections."""
    d = cfg.d_model
    qkv = 2 * d * (d + 2 * cfg.kv_dim)      # q + k + v projections
    out = 2 * d * d                         # output projection
    ffn = 2 * 3 * d * cfg.ffn_dim           # SwiGLU: gate + up + down
    return qkv + out + ffn


def _prefill_phase(cfg: ModelConfig, batch: int, prompt_len: int,
                   peak_flops: float, peak_bw: float) -> PhaseResult:
    B, S, L = batch, prompt_len, cfg.num_layers
    ridge = peak_flops / peak_bw

    linear = B * S * L * _layer_linear_flops(cfg)
    # Attention scores QK^T + AV, both quadratic in S.
    attn = 2 * 2 * B * L * S * S * cfg.d_model
    flops = linear + attn

    # Weights read once for the whole prompt; KV cache written out.
    kv_written = kv_footprint(cfg, B, S).total_bytes
    bytes_moved = cfg.weight_bytes + kv_written

    ai = flops / bytes_moved if bytes_moved else 0.0
    time_s = max(flops / peak_flops, bytes_moved / peak_bw)
    return PhaseResult("prefill", flops, bytes_moved, ai, ridge,
                       ai >= ridge, time_s * 1e3)


def _decode_phase(cfg: ModelConfig, batch: int, context_len: int,
                  peak_flops: float, peak_bw: float,
                  offload: OffloadPlan) -> PhaseResult:
    B, L = batch, cfg.num_layers
    ridge = peak_flops / peak_bw

    # One new token per sequence per step.
    linear = B * 1 * L * _layer_linear_flops(cfg)
    # New query attends to the whole context.
    attn = 2 * 2 * B * L * context_len * cfg.d_model
    flops = linear + attn

    # Every step re-reads all weights and scans the entire KV cache.
    kv_bytes = kv_footprint(cfg, B, context_len).total_bytes
    bytes_moved = cfg.weight_bytes + kv_bytes

    ai = flops / bytes_moved if bytes_moved else 0.0
    # Compute term uses peak FLOPs; memory term charges the weights at HBM
    # bandwidth and the KV cache at its (possibly offloaded) blended rate.
    compute_s = flops / peak_flops
    weight_read_s = cfg.weight_bytes / peak_bw
    kv_read_s = offload.blended_read_s
    time_s = max(compute_s, weight_read_s + kv_read_s)
    return PhaseResult("decode", flops, bytes_moved, ai, ridge,
                       ai >= ridge, time_s * 1e3)


def analyze_inference(
    cfg: ModelConfig,
    batch: int = 1,
    prompt_len: int = 1024,
    gen_len: int = 128,
    device: str | None = None,
    activation_reserve_bytes: float = 2e9,
) -> dict:
    """
    Analyse one prefill + ``gen_len``-step decode run of ``cfg``.

    Returns a dict with ``config``, ``kv``, ``offload``, ``prefill``, ``decode``,
    serving throughput numbers, and ``advice``.  ``context_len`` for decode cost
    is taken at the end of generation (prompt_len + gen_len), the worst case.
    """
    from ..analyzer.bottleneck import _detect_device, _DEVICE_SPECS

    if device is None:
        device = _detect_device()
    specs = _DEVICE_SPECS.get(device, _DEVICE_SPECS["CPU"])
    peak_flops, peak_bw = specs["peak_flops"], specs["peak_bw"]

    context_len = prompt_len + gen_len
    kv = kv_footprint(cfg, batch, context_len)

    # Offload plan: reserve weights + activation scratch on the GPU first.
    reserve = cfg.weight_bytes + activation_reserve_bytes
    tiers = default_tiers(device, reserve)
    offload = plan_kv_offload(kv.total_bytes, tiers)

    prefill = _prefill_phase(cfg, batch, prompt_len, peak_flops, peak_bw)
    decode = _decode_phase(cfg, batch, context_len, peak_flops, peak_bw, offload)

    # Serving throughput.
    ttft_ms = prefill.time_ms
    ms_per_token = decode.time_ms                  # one step -> batch new tokens
    decode_tokens_per_s = (batch / (decode.time_ms / 1e3)) if decode.time_ms else 0.0
    total_ms = prefill.time_ms + gen_len * decode.time_ms

    advice = _serving_advice(cfg, batch, context_len, decode, offload, device)

    return {
        "config": {
            "name": cfg.name,
            "num_params": cfg.num_params,
            "weight_bytes": cfg.weight_bytes,
            "d_model": cfg.d_model,
            "num_kv_heads": cfg.num_kv_heads,
            "num_heads": cfg.num_heads,
            "dtype_bytes": cfg.dtype_bytes,
        },
        "device": device,
        "batch": batch,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "context_len": context_len,
        "kv": kv.to_dict(),
        "offload": offload.to_dict(),
        "prefill": prefill.to_dict(),
        "decode": decode.to_dict(),
        "ttft_ms": round(ttft_ms, 3),
        "ms_per_output_token": round(ms_per_token, 4),
        "decode_tokens_per_s": round(decode_tokens_per_s, 1),
        "total_latency_ms": round(total_ms, 3),
        "advice": advice,
    }


def _serving_advice(cfg: ModelConfig, batch: int, context_len: int,
                    decode: PhaseResult, offload: OffloadPlan,
                    device: str) -> list[str]:
    tips: list[str] = []

    if not offload.fits_in_gpu:
        spilled = [p for p in offload.placements if p["tier"] != "GPU HBM"]
        where = ", ".join(f"{p['bytes']/1e9:.1f} GB→{p['tier']}" for p in spilled)
        tips.append(
            f"KV cache ({offload.total_bytes/1e9:.1f} GB) exceeds free HBM; "
            f"spilled {where}. Blended KV read is {offload.slowdown_vs_hbm:.1f}× "
            f"slower than pure HBM — this is the decode bottleneck."
        )
        tips.append(
            "Reduce KV pressure: lower batch, cap context, or quantize the KV "
            "cache to fp8 (halves cache bytes and the offload penalty)."
        )
    else:
        headroom = _detect_headroom(cfg, batch, context_len, device)
        if headroom is not None:
            tips.append(
                f"KV cache fits in HBM with room for ~{headroom}× the current "
                f"batch before spilling. Raise batch size to improve decode "
                f"throughput (decode is memory-bound, so larger batches are "
                f"near-free until the cache overflows)."
            )

    if not decode.is_compute_bound:
        tips.append(
            f"Decode is memory-bound (AI {decode.arithmetic_intensity:.1f} < "
            f"ridge {decode.ridge_point:.0f} FLOP/byte). Extra FLOPs won't help; "
            f"the wins are KV-cache reuse (prefix/paged), fp8 KV, and batching."
        )

    if cfg.num_kv_heads == cfg.num_heads:
        tips.append(
            "Model uses full multi-head attention (num_kv_heads == num_heads). "
            "GQA/MQA checkpoints cut KV-cache bytes by "
            f"{cfg.num_heads // max(cfg.num_kv_heads, 1)}×+ and directly relieve "
            "the decode bottleneck."
        )

    tips.append(
        "Separate prefill (compute-bound) and decode (memory-bound) onto "
        "different schedules or devices — this is the basis of disaggregated / "
        "chunked-prefill serving in vLLM and SGLang."
    )
    return tips


def _detect_headroom(cfg: ModelConfig, batch: int, context_len: int,
                     device: str) -> int | None:
    """How many multiples of the current batch fit in free HBM before spilling."""
    from .kvcache import _GPU_HBM_BYTES
    from ..analyzer.bottleneck import _DEVICE_SPECS  # noqa: F401 (kept parallel)

    hbm = _GPU_HBM_BYTES.get(device)
    if hbm is None:
        return None
    reserve = cfg.weight_bytes + 2e9
    free = max(hbm - reserve, 0.0)
    per_batch = kv_footprint(cfg, 1, context_len).total_bytes
    if per_batch <= 0:
        return None
    return max(int(free / per_batch), 0)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_inference_report(report: dict, fn_name: str | None = None) -> None:
    """Print a formatted LLM inference roofline report."""
    w = 34
    sep = "═" * (w + 26)
    name = fn_name or report["config"]["name"]

    print(f"\n{sep}")
    print(f"  HelixIR Inference Roofline · {name}")
    print(sep)

    c = report["config"]
    print(f"\n  {'── Model / device':─<{w+24}}")
    print(f"  {'Parameters':<{w}} {c['num_params'] / 1e9:.2f} B")
    print(f"  {'Weights (' + str(c['dtype_bytes']) + ' B/param)':<{w}} "
          f"{c['weight_bytes'] / 1e9:.2f} GB")
    print(f"  {'Attention heads (q / kv)':<{w}} {c['num_heads']} / {c['num_kv_heads']}")
    print(f"  {'Device':<{w}} {report['device']}")
    print(f"  {'Workload (B / prompt / gen)':<{w}} "
          f"{report['batch']} / {report['prompt_len']} / {report['gen_len']}")

    kv = report["kv"]
    off = report["offload"]
    print(f"\n  {'── KV cache':─<{w+24}}")
    print(f"  {'Bytes / token (all layers)':<{w}} {kv['bytes_per_token'] / 1e3:.1f} KB")
    print(f"  {'Per sequence @ ctx ' + str(report['context_len']):<{w}} "
          f"{kv['bytes_per_sequence'] / 1e6:.1f} MB")
    print(f"  {'Total (batch ' + str(report['batch']) + ')':<{w}} "
          f"{kv['total_bytes'] / 1e9:.2f} GB")
    fits = "yes" if off["fits_in_gpu"] else "NO — offloaded"
    print(f"  {'Fits in free HBM':<{w}} {fits}")
    for p in off["placements"]:
        print(f"    {'· ' + p['tier']:<{w-2}} {p['bytes'] / 1e9:>7.2f} GB "
              f"@ {p['bandwidth_bytes_s'] / 1e9:.0f} GB/s")
    if not off["fits_in_gpu"]:
        print(f"  {'Blended KV bandwidth':<{w}} {off['blended_bandwidth'] / 1e9:.0f} GB/s "
              f"({off['slowdown_vs_hbm']:.1f}× slower than HBM)")

    for phase_key in ("prefill", "decode"):
        p = report[phase_key]
        print(f"\n  {'── ' + phase_key.capitalize():─<{w+24}}")
        print(f"  {'FLOPs':<{w}} {p['flops'] / 1e9:.2f} GFLOP")
        print(f"  {'Bytes moved':<{w}} {p['bytes'] / 1e9:.2f} GB")
        print(f"  {'Arithmetic intensity':<{w}} {p['arithmetic_intensity']:.1f} FLOP/byte")
        print(f"  {'Ridge point':<{w}} {p['ridge_point']:.0f} FLOP/byte")
        print(f"  {'Classification':<{w}} {p['bound'].upper()}")
        print(f"  {'Roofline time':<{w}} {p['time_ms']:.3f} ms")

    print(f"\n  {'── Serving':─<{w+24}}")
    print(f"  {'TTFT (prefill)':<{w}} {report['ttft_ms']:.2f} ms")
    print(f"  {'Per output token':<{w}} {report['ms_per_output_token']:.3f} ms")
    print(f"  {'Decode throughput':<{w}} {report['decode_tokens_per_s']:.0f} tok/s")
    print(f"  {'Total latency':<{w}} {report['total_latency_ms']:.1f} ms")

    print(f"\n  {'── Advice':─<{w+24}}")
    for i, tip in enumerate(report["advice"], 1):
        words, line, lines = tip.split(), [], []
        for word in words:
            if len(" ".join(line + [word])) > 74:
                lines.append(" ".join(line)); line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        print(f"  {i}. {lines[0]}")
        for cont in lines[1:]:
            print(f"     {cont}")

    print(f"\n{sep}\n")
