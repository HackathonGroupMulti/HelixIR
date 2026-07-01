"""
LLM inference analysis — KV-cache modeling, multi-tier offload planning, and
prefill/decode roofline classification.  All offline (no GPU required).

    from helix.inference import ModelConfig, analyze_inference, print_inference_report

    cfg = ModelConfig.from_preset("llama3-8b")
    report = analyze_inference(cfg, batch=16, prompt_len=1024, gen_len=128,
                               device="A100")
    print_inference_report(report)
"""
from .kvcache import (
    ModelConfig,
    KVFootprint,
    kv_footprint,
    MemoryTier,
    default_tiers,
    plan_kv_offload,
    OffloadPlan,
    available_presets,
)
from .analyze import (
    analyze_inference,
    print_inference_report,
    PhaseResult,
)
from .serving import serving_benchmark, print_serving_result, ServingResult
from .sweep import (
    sweep_inference, print_sweep, sweep_to_markdown, sweep_to_csv, SweepRow,
)

__all__ = [
    "ModelConfig",
    "KVFootprint",
    "kv_footprint",
    "MemoryTier",
    "default_tiers",
    "plan_kv_offload",
    "OffloadPlan",
    "available_presets",
    "analyze_inference",
    "print_inference_report",
    "PhaseResult",
    # Serving benchmark (vLLM w/ analytic fallback)
    "serving_benchmark",
    "print_serving_result",
    "ServingResult",
    # Sweep pipeline
    "sweep_inference",
    "print_sweep",
    "sweep_to_markdown",
    "sweep_to_csv",
    "SweepRow",
]
