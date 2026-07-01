"""
Serving-side inference benchmark — measure a real engine, fall back to analysis.

``serving_benchmark`` always computes HelixIR's analytic prefill/decode estimate
(``analyze_inference``).  If a serving engine (vLLM) *and* a concrete model are
available it also measures TTFT (time-to-first-token) and TPOT (time-per-output-
token) on real hardware, then reports the two side by side — turning the analytic
model into something *validated* against measured numbers rather than asserted.

On a CPU-only box, or without vLLM installed, it degrades cleanly to the analytic
estimate and says so (``backend == "analytic"``).  This mirrors the Pallas kernel
fallback used elsewhere in HelixIR.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

from .kvcache import ModelConfig
from .analyze import analyze_inference


def _vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class ServingResult:
    backend: str            # "vllm" | "analytic"
    model: str
    batch: int
    prompt_len: int
    gen_len: int
    device: str

    ttft_ms: float
    tpot_ms: float          # time per output token (decode step)
    throughput_tok_s: float

    # Analytic reference (always present).  When backend == "vllm" these hold the
    # HelixIR prediction alongside the measured ttft_ms / tpot_ms above.
    analytic_ttft_ms: float = 0.0
    analytic_tpot_ms: float = 0.0
    ttft_error_pct: float | None = None
    tpot_error_pct: float | None = None

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model,
            "batch": self.batch,
            "prompt_len": self.prompt_len,
            "gen_len": self.gen_len,
            "device": self.device,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "throughput_tok_s": self.throughput_tok_s,
            "analytic_ttft_ms": self.analytic_ttft_ms,
            "analytic_tpot_ms": self.analytic_tpot_ms,
            "ttft_error_pct": self.ttft_error_pct,
            "tpot_error_pct": self.tpot_error_pct,
        }


def _pct_error(measured: float, predicted: float) -> float | None:
    if measured <= 0:
        return None
    return round(abs(measured - predicted) / measured * 100, 1)


def _measure_vllm(
    model: str, batch: int, prompt_len: int, gen_len: int,
) -> tuple[float, float, float]:
    """
    Measure (ttft_ms, tpot_ms, throughput_tok_s) with vLLM.

    TTFT is timed with a single-token generation; TPOT is derived from a longer
    generation with the prefill cost subtracted out.  Requires a GPU and a model
    weight vLLM can load — never reached on the CPU fallback path.
    """
    from vllm import LLM, SamplingParams  # imported lazily

    llm = LLM(model=model, enforce_eager=True)
    prompt = "word " * prompt_len
    prompts = [prompt] * batch

    # TTFT: generate exactly one token.
    t0 = time.perf_counter()
    llm.generate(prompts, SamplingParams(max_tokens=1))
    ttft_ms = (time.perf_counter() - t0) * 1e3

    # Full generation: subtract prefill to isolate decode.
    t0 = time.perf_counter()
    llm.generate(prompts, SamplingParams(max_tokens=gen_len))
    full_ms = (time.perf_counter() - t0) * 1e3

    decode_ms = max(full_ms - ttft_ms, 0.0)
    tpot_ms = decode_ms / max(gen_len - 1, 1)
    throughput = (batch * gen_len) / (full_ms / 1e3) if full_ms > 0 else 0.0
    return ttft_ms, tpot_ms, throughput


def serving_benchmark(
    cfg: ModelConfig,
    batch: int = 1,
    prompt_len: int = 1024,
    gen_len: int = 128,
    device: str | None = None,
    model: str | None = None,
    backend: str = "auto",
) -> ServingResult:
    """
    Benchmark serving latency for ``cfg``.

    Parameters
    ----------
    cfg      : model shape (drives the analytic estimate).
    model    : HF model id/path for vLLM.  Required to actually measure; if None
               or vLLM is unavailable, returns the analytic estimate.
    backend  : "auto" (measure if possible, else analytic), "vllm" (force
               measure — raises if unavailable), or "analytic" (never measure).
    """
    analytic = analyze_inference(cfg, batch=batch, prompt_len=prompt_len,
                                 gen_len=gen_len, device=device)
    a_ttft = analytic["ttft_ms"]
    a_tpot = analytic["ms_per_output_token"]
    dev = analytic["device"]

    want_measure = backend in ("auto", "vllm") and model is not None and _vllm_available()
    if backend == "vllm" and not want_measure:
        raise RuntimeError(
            "backend='vllm' requires vLLM installed and model=<id> "
            "(pip install vllm; needs a CUDA GPU)."
        )

    if want_measure:
        ttft_ms, tpot_ms, throughput = _measure_vllm(model, batch, prompt_len, gen_len)
        return ServingResult(
            backend="vllm", model=model, batch=batch, prompt_len=prompt_len,
            gen_len=gen_len, device=dev,
            ttft_ms=round(ttft_ms, 3), tpot_ms=round(tpot_ms, 4),
            throughput_tok_s=round(throughput, 1),
            analytic_ttft_ms=a_ttft, analytic_tpot_ms=a_tpot,
            ttft_error_pct=_pct_error(ttft_ms, a_ttft),
            tpot_error_pct=_pct_error(tpot_ms, a_tpot),
        )

    # Analytic fallback.
    return ServingResult(
        backend="analytic", model=model or cfg.name, batch=batch,
        prompt_len=prompt_len, gen_len=gen_len, device=dev,
        ttft_ms=a_ttft, tpot_ms=a_tpot,
        throughput_tok_s=analytic["decode_tokens_per_s"],
        analytic_ttft_ms=a_ttft, analytic_tpot_ms=a_tpot,
    )


def print_serving_result(res: ServingResult) -> None:
    w = 30
    sep = "═" * (w + 24)
    print(f"\n{sep}")
    print(f"  HelixIR Serving Benchmark · {res.model}")
    print(sep)
    print(f"  {'Backend':<{w}} {res.backend}"
          f"{'  (vLLM unavailable — estimate)' if res.backend == 'analytic' else ''}")
    print(f"  {'Device':<{w}} {res.device}")
    print(f"  {'Workload (B / prompt / gen)':<{w}} "
          f"{res.batch} / {res.prompt_len} / {res.gen_len}")
    print(f"  {'TTFT':<{w}} {res.ttft_ms:.2f} ms")
    print(f"  {'Time per output token':<{w}} {res.tpot_ms:.3f} ms")
    print(f"  {'Throughput':<{w}} {res.throughput_tok_s:.0f} tok/s")
    if res.backend == "vllm":
        print(f"\n  {'── Analytic vs measured':─<{w+22}}")
        print(f"  {'Predicted TTFT':<{w}} {res.analytic_ttft_ms:.2f} ms "
              f"({res.ttft_error_pct}% off)")
        print(f"  {'Predicted TPOT':<{w}} {res.analytic_tpot_ms:.3f} ms "
              f"({res.tpot_error_pct}% off)")
    print(f"\n{sep}\n")
