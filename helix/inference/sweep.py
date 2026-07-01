"""
Automated inference sweep — grid over batch × context, one report out.

A serving engineer's first question is "how far can I push batch and context
before the KV cache spills and decode falls off a cliff?"  ``sweep_inference``
answers it by running ``analyze_inference`` across a grid and collecting the
numbers that matter (KV size, HBM fit, decode bound, TTFT, TPOT, throughput,
offload slowdown) into rows you can print, drop into a report, or save as CSV.

This is the reproducible-benchmark-pipeline piece: deterministic, offline, and
scriptable from the CLI (``helix infer-sweep``).
"""
from __future__ import annotations
from dataclasses import dataclass

from .kvcache import ModelConfig
from .analyze import analyze_inference


@dataclass
class SweepRow:
    batch: int
    prompt_len: int
    gen_len: int
    kv_gb: float
    fits_hbm: bool
    decode_bound: str
    ttft_ms: float
    tpot_ms: float
    throughput_tok_s: float
    offload_slowdown: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def sweep_inference(
    cfg: ModelConfig,
    batches: list[int] | tuple[int, ...] = (1, 8, 32, 64),
    prompt_lens: list[int] | tuple[int, ...] = (2048,),
    gen_len: int = 128,
    device: str | None = None,
) -> list[SweepRow]:
    """Run the analytic inference model across the batch × prompt_len grid."""
    rows: list[SweepRow] = []
    for prompt_len in prompt_lens:
        for batch in batches:
            r = analyze_inference(cfg, batch=batch, prompt_len=prompt_len,
                                  gen_len=gen_len, device=device)
            rows.append(SweepRow(
                batch=batch,
                prompt_len=prompt_len,
                gen_len=gen_len,
                kv_gb=round(r["kv"]["total_bytes"] / 1e9, 2),
                fits_hbm=r["offload"]["fits_in_gpu"],
                decode_bound=r["decode"]["bound"],
                ttft_ms=r["ttft_ms"],
                tpot_ms=r["ms_per_output_token"],
                throughput_tok_s=r["decode_tokens_per_s"],
                offload_slowdown=round(r["offload"]["slowdown_vs_hbm"], 2),
            ))
    return rows


_COLUMNS = [
    ("batch", "B", "{}"),
    ("prompt_len", "prompt", "{}"),
    ("kv_gb", "KV GB", "{:.2f}"),
    ("fits_hbm", "HBM", "{}"),
    ("tpot_ms", "TPOT ms", "{:.2f}"),
    ("throughput_tok_s", "tok/s", "{:.0f}"),
    ("offload_slowdown", "slow×", "{:.2f}"),
]


def _fmt(row: SweepRow, key: str, spec: str) -> str:
    val = getattr(row, key)
    if isinstance(val, bool):
        return "yes" if val else "no"
    return spec.format(val)


def print_sweep(rows: list[SweepRow], cfg_name: str = "", device: str = "") -> None:
    """Print the sweep as an aligned text table."""
    title = f"HelixIR Inference Sweep · {cfg_name} · {device}".strip(" ·")
    headers = [h for _, h, _ in _COLUMNS]
    table = [[_fmt(r, k, s) for k, _, s in _COLUMNS] for r in rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in table)) if table else len(headers[i])
              for i in range(len(headers))]

    line = "─" * (sum(widths) + 3 * len(widths) + 1)
    print(f"\n{line}\n  {title}\n{line}")
    print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("─" * widths[i] for i in range(len(headers))))
    for r in table:
        print("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(r))))
    print(f"{line}\n")


def sweep_to_markdown(rows: list[SweepRow]) -> str:
    """Render the sweep as a GitHub-flavored markdown table (for the README)."""
    headers = [h for _, h, _ in _COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r, k, s) for k, _, s in _COLUMNS) + " |")
    return "\n".join(lines)


def sweep_to_csv(rows: list[SweepRow]) -> str:
    """Render the full sweep (all fields) as CSV text."""
    if not rows:
        return ""
    fields = list(rows[0].to_dict().keys())
    out = [",".join(fields)]
    for r in rows:
        d = r.to_dict()
        out.append(",".join(str(d[f]) for f in fields))
    return "\n".join(out)
