"""
KV-cache memory model and multi-level offload planning for LLM inference.

Autoregressive decoding keeps a per-layer cache of keys and values for every
token generated so far.  That cache — not the weights — is what limits batch
size and context length at serving time, and reading it back every decode step
is what makes decode *memory-bandwidth-bound* (see ``analyze.py``).

This module models the cache analytically (no GPU required):

  * ``ModelConfig``  — transformer shape, GQA/MQA-aware, with real presets.
  * ``KVFootprint``  — bytes-per-token / per-sequence / total for a workload.
  * ``MemoryTier``   — a level of the memory hierarchy (HBM / CPU / SSD / remote).
  * ``plan_kv_offload`` — spill an oversized cache across tiers and report the
                          blended read bandwidth, i.e. the FlexKV-style question
                          of *where does the KV cache live and what does it cost*.
"""
from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Shape of a decoder-only transformer.

    ``num_kv_heads < num_heads`` expresses grouped-query attention (GQA); set it
    equal to ``num_heads`` for vanilla multi-head attention, or to 1 for
    multi-query attention (MQA).  KV-cache size scales with ``num_kv_heads``,
    which is exactly why GQA/MQA models serve longer contexts.
    """
    name: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_dim: int
    vocab_size: int = 32_000
    dtype_bytes: int = 2          # fp16 / bf16 = 2, fp8 = 1, fp32 = 4

    @property
    def d_model(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Combined key/value width per layer (GQA-reduced)."""
        return self.num_kv_heads * self.head_dim

    @property
    def num_params(self) -> int:
        """Approximate parameter count (SwiGLU FFN, tied-ish embeddings)."""
        d = self.d_model
        attn = d * (d + 2 * self.kv_dim) + d * d          # qkv + out proj
        ffn = 3 * d * self.ffn_dim                        # gate + up + down
        per_layer = attn + ffn
        embed = 2 * self.vocab_size * d                   # input embed + lm head
        return self.num_layers * per_layer + embed

    @property
    def weight_bytes(self) -> int:
        return self.num_params * self.dtype_bytes

    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache added for one token across all layers (K and V)."""
        return 2 * self.num_layers * self.kv_dim * self.dtype_bytes

    @classmethod
    def from_preset(cls, name: str, dtype_bytes: int = 2) -> "ModelConfig":
        if name not in _PRESETS:
            raise KeyError(
                f"unknown preset {name!r}; available: {sorted(_PRESETS)}"
            )
        kw = dict(_PRESETS[name])
        return cls(name=name, dtype_bytes=dtype_bytes, **kw)


# Real model shapes.  head_dim * num_heads == hidden_size.
_PRESETS: dict[str, dict] = {
    "llama2-7b":  dict(num_layers=32, num_heads=32, num_kv_heads=32,
                       head_dim=128, ffn_dim=11008, vocab_size=32_000),
    "llama2-13b": dict(num_layers=40, num_heads=40, num_kv_heads=40,
                       head_dim=128, ffn_dim=13824, vocab_size=32_000),
    "llama2-70b": dict(num_layers=80, num_heads=64, num_kv_heads=8,
                       head_dim=128, ffn_dim=28672, vocab_size=32_000),
    "llama3-8b":  dict(num_layers=32, num_heads=32, num_kv_heads=8,
                       head_dim=128, ffn_dim=14336, vocab_size=128_256),
    "llama3-70b": dict(num_layers=80, num_heads=64, num_kv_heads=8,
                       head_dim=128, ffn_dim=28672, vocab_size=128_256),
    "mistral-7b": dict(num_layers=32, num_heads=32, num_kv_heads=8,
                       head_dim=128, ffn_dim=14336, vocab_size=32_000),
    "qwen2-7b":   dict(num_layers=28, num_heads=28, num_kv_heads=4,
                       head_dim=128, ffn_dim=18944, vocab_size=152_064),
}


def available_presets() -> list[str]:
    return sorted(_PRESETS)


# ---------------------------------------------------------------------------
# KV-cache footprint for a concrete workload
# ---------------------------------------------------------------------------

@dataclass
class KVFootprint:
    batch: int
    context_len: int
    bytes_per_token: int
    bytes_per_sequence: int
    total_bytes: int

    def to_dict(self) -> dict:
        return {
            "batch": self.batch,
            "context_len": self.context_len,
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_sequence": self.bytes_per_sequence,
            "total_bytes": self.total_bytes,
        }


def kv_footprint(cfg: ModelConfig, batch: int, context_len: int) -> KVFootprint:
    """KV-cache size for ``batch`` sequences each holding ``context_len`` tokens."""
    per_tok = cfg.kv_bytes_per_token()
    per_seq = per_tok * context_len
    return KVFootprint(
        batch=batch,
        context_len=context_len,
        bytes_per_token=per_tok,
        bytes_per_sequence=per_seq,
        total_bytes=per_seq * batch,
    )


# ---------------------------------------------------------------------------
# Memory hierarchy + offload planning (FlexKV-style)
# ---------------------------------------------------------------------------

@dataclass
class MemoryTier:
    name: str
    capacity_bytes: float
    bandwidth_bytes_s: float


# GPU HBM capacity (bytes) per device — bandwidth comes from the roofline specs.
_GPU_HBM_BYTES: dict[str, float] = {
    "H100": 80e9, "H100 SXM": 80e9, "H100 SXM5 80GB": 80e9,
    "A100": 80e9, "A100-SXM4-80GB": 80e9, "A100-SXM4-40GB": 40e9,
    "V100": 32e9, "V100-SXM2-32GB": 32e9,
    "RTX 4090": 24e9, "RTX 3090": 24e9, "RTX 3080": 10e9,
    "L4": 24e9, "T4": 16e9,
    "CPU": 8e9,
}

# Off-GPU tiers, ordered fastest → slowest.  Bandwidths are per-direction
# steady-state numbers for a typical PCIe-gen5 server.
_CPU_BW = 55e9          # ~PCIe 5.0 x16 host<->device
_SSD_BW = 6e9           # ~NVMe gen4 sequential
_REMOTE_BW = 12.5e9     # ~100 GbE / RDMA


def default_tiers(device: str, gpu_reserve_bytes: float) -> list[MemoryTier]:
    """
    Build the default GPU → CPU → SSD → remote hierarchy for ``device``.

    ``gpu_reserve_bytes`` (weights + activation scratch) is subtracted from HBM
    so only the *free* HBM is offered to the KV cache.
    """
    from ..analyzer.bottleneck import _DEVICE_SPECS

    hbm = _GPU_HBM_BYTES.get(device, 40e9)
    gpu_bw = _DEVICE_SPECS.get(device, _DEVICE_SPECS["CPU"])["peak_bw"]
    free_hbm = max(hbm - gpu_reserve_bytes, 0.0)
    return [
        MemoryTier("GPU HBM", free_hbm, gpu_bw),
        MemoryTier("CPU RAM", 512e9, _CPU_BW),
        MemoryTier("NVMe SSD", 8e12, _SSD_BW),
        MemoryTier("Remote", float("inf"), _REMOTE_BW),
    ]


@dataclass
class OffloadPlan:
    total_bytes: float
    placements: list[dict] = field(default_factory=list)   # per-tier bytes
    fits_in_gpu: bool = True
    blended_read_s: float = 0.0        # time to scan the whole cache once
    blended_bandwidth: float = 0.0     # total_bytes / blended_read_s
    slowdown_vs_hbm: float = 1.0       # blended read time / all-in-HBM time

    def to_dict(self) -> dict:
        return {
            "total_bytes": self.total_bytes,
            "placements": self.placements,
            "fits_in_gpu": self.fits_in_gpu,
            "blended_read_s": self.blended_read_s,
            "blended_bandwidth": self.blended_bandwidth,
            "slowdown_vs_hbm": self.slowdown_vs_hbm,
        }


def plan_kv_offload(total_bytes: float, tiers: list[MemoryTier]) -> OffloadPlan:
    """
    Greedily place ``total_bytes`` of KV cache across ``tiers`` (fastest first)
    and compute the blended bandwidth for a full-cache read.

    Decode reads the entire live cache every step, so the time to scan it once
    across whatever tiers it spilled into is the quantity that governs the
    offload latency penalty.
    """
    remaining = total_bytes
    placements: list[dict] = []
    read_time = 0.0
    gpu_capacity = tiers[0].capacity_bytes if tiers else 0.0

    for tier in tiers:
        if remaining <= 0:
            break
        placed = min(remaining, tier.capacity_bytes)
        remaining -= placed
        read_time += placed / tier.bandwidth_bytes_s
        placements.append({
            "tier": tier.name,
            "bytes": placed,
            "bandwidth_bytes_s": tier.bandwidth_bytes_s,
        })

    fits = total_bytes <= gpu_capacity
    hbm_bw = tiers[0].bandwidth_bytes_s if tiers else 1.0
    ideal_read = total_bytes / hbm_bw if hbm_bw else 0.0
    blended_bw = total_bytes / read_time if read_time > 0 else hbm_bw
    slowdown = read_time / ideal_read if ideal_read > 0 else 1.0

    return OffloadPlan(
        total_bytes=total_bytes,
        placements=placements,
        fits_in_gpu=fits,
        blended_read_s=read_time,
        blended_bandwidth=blended_bw,
        slowdown_vs_hbm=slowdown,
    )
