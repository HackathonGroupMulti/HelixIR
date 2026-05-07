"""
HelixIR Jupyter integration.

Load the extension once per notebook:
    %load_ext helix.jupyter

Then analyse any JAX function inline:
    %helix my_fn x w          # uses variables from the notebook namespace
    %helix my_fn x w --bwd    # also analyse backward pass
    %helix my_fn x w --devices 8

The cell magic traces whatever function you define:
    %%helix x w               # x and w taken from namespace; fn defined in cell
    def fn(x, w):
        return jnp.dot(x, w)
"""
from __future__ import annotations
import io
import json
import textwrap
import traceback
import html as _html
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Graph layout helpers — layered DAG, pure Python → SVG, no networkx needed
# ---------------------------------------------------------------------------

_CATEGORY_COLOR = {
    "matmul":      "#3b82f6",
    "elementwise": "#22c55e",
    "reduction":   "#f97316",
    "memory":      "#ef4444",
    "collective":  "#a855f7",
    "other":       "#6b7280",
}

def _topo_layers(nodes, edges) -> list[list[int]]:
    """Assign each node to its longest-path layer from any source."""
    succ: dict[int, list[int]] = {n["id"]: [] for n in nodes}
    pred: dict[int, list[int]] = {n["id"]: [] for n in nodes}
    for e in edges:
        succ[e["src"]].append(e["dst"])
        pred[e["dst"]].append(e["src"])

    layer: dict[int, int] = {}
    queue = [n["id"] for n in nodes if not pred[n["id"]]]
    for nid in queue:
        layer[nid] = 0

    ordered = list(queue)
    while queue:
        nid = queue.pop(0)
        for s in succ[nid]:
            layer[s] = max(layer.get(s, 0), layer[nid] + 1)
            if s not in [q for q in queue]:
                queue.append(s)
                ordered.append(s)

    max_layer = max(layer.values(), default=0)
    buckets: list[list[int]] = [[] for _ in range(max_layer + 1)]
    for nid, l in layer.items():
        buckets[l].append(nid)
    return buckets


def _graph_svg(graph_dict: dict, width: int = 820, height: int = 340) -> str:
    nodes = graph_dict["nodes"]
    edges = graph_dict["edges"]
    if not nodes:
        return f'<svg width="{width}" height="{height}"><text x="10" y="20" fill="#9ca3af">No ops.</text></svg>'

    node_by_id = {n["id"]: n for n in nodes}
    layers = _topo_layers(nodes, edges)

    # Clamp to max 12 layers for display
    layers = layers[:12]
    all_shown = {nid for layer in layers for nid in layer}

    col_w = min(width // max(len(layers), 1), 100)
    positions: dict[int, tuple[float, float]] = {}
    for li, layer in enumerate(layers):
        x = 40 + li * col_w
        for ni, nid in enumerate(layer):
            row_h = height // (len(layer) + 1)
            y = (ni + 1) * row_h
            positions[nid] = (x, y)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#111827;border-radius:8px;border:1px solid #374151">'
    ]

    # Edges
    for e in edges:
        if e["src"] not in positions or e["dst"] not in positions:
            continue
        x1, y1 = positions[e["src"]]
        x2, y2 = positions[e["dst"]]
        svg_parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#374151" stroke-width="1.5" marker-end="url(#arr)"/>'
        )

    # Arrow marker
    svg_parts.insert(1,
        '<defs><marker id="arr" viewBox="0 -5 10 10" refX="14" refY="0" '
        'markerWidth="5" markerHeight="5" orient="auto">'
        '<path d="M0,-5L10,0L0,5" fill="#4b5563"/></marker></defs>'
    )

    # Nodes
    max_bytes = max((n["bytes_written"] for n in nodes), default=1) or 1
    for n in nodes:
        if n["id"] not in positions:
            continue
        x, y = positions[n["id"]]
        r = 6 + int(12 * (n["bytes_written"] / max_bytes) ** 0.5)
        color = _CATEGORY_COLOR.get(n["category"], "#6b7280")
        stroke = "#f9fafb" if n["is_compute_bound"] else "#f97316"
        dash = "none" if n["is_compute_bound"] else "4,2"
        label = n["name"][:9] + "…" if len(n["name"]) > 10 else n["name"]
        ai = f"{n['arithmetic_intensity']:.1f}"
        title = _html.escape(
            f"{n['name']} ({n['category']})\n"
            f"AI: {ai} FLOP/byte\n"
            f"FLOPs: {n['flops']/1e6:.2f}M\n"
            f"Bytes: {(n['bytes_read']+n['bytes_written'])/1e6:.2f}MB"
        )
        svg_parts.append(
            f'<g><title>{title}</title>'
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" fill-opacity="0.85" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-dasharray="{dash}"/>'
            f'<text x="{x}" y="{y+r+11}" text-anchor="middle" font-size="8" '
            f'fill="#9ca3af" font-family="monospace">{label}</text>'
            f'</g>'
        )

    # Overflow indicator
    remaining = len(nodes) - len(all_shown)
    if remaining > 0:
        svg_parts.append(
            f'<text x="{width-8}" y="16" text-anchor="end" font-size="9" fill="#6b7280">'
            f'+{remaining} more</text>'
        )

    # Legend
    lx = 8
    for cat, col in list(_CATEGORY_COLOR.items())[:4]:
        svg_parts.append(
            f'<circle cx="{lx+5}" cy="{height-10}" r="4" fill="{col}"/>'
            f'<text x="{lx+12}" y="{height-6}" font-size="8" fill="#6b7280" font-family="monospace">{cat}</text>'
        )
        lx += 70

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _roofline_svg(graph_dict: dict, roofline: dict, width: int = 400, height: int = 240) -> str:
    nodes = [n for n in graph_dict["nodes"] if n["arithmetic_intensity"] > 0 and n["flops"] > 0]
    peak_flops = roofline["peak_flops"]
    peak_bw = roofline["peak_bandwidth"]
    ridge = roofline["ridge_point"]
    peak_t = peak_flops / 1e12

    pad_l, pad_b, pad_t, pad_r = 52, 36, 16, 16
    W = width  - pad_l - pad_r
    H = height - pad_t - pad_b

    intensities = [n["arithmetic_intensity"] for n in nodes] + [ridge * 0.05, ridge * 10]
    x_min = max(min(intensities) * 0.5, 0.01)
    x_max = max(intensities) * 2

    import math
    def lx(v):
        v = max(v, x_min)
        return pad_l + W * (math.log10(v) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min))

    y_max = peak_t * 1.5
    def ly(v):
        return pad_t + H * (1 - v / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#111827;border-radius:8px;border:1px solid #374151">'
    ]

    # Axes
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+H}" stroke="#374151" stroke-width="1"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+H}" x2="{pad_l+W}" y2="{pad_t+H}" stroke="#374151" stroke-width="1"/>')

    # Roofline
    rx = lx(ridge)
    r1x, r1y = lx(x_min), ly((x_min * peak_bw) / 1e12)
    r2x, r2y = rx, ly(peak_t)
    r3x, r3y = lx(x_max), ly(peak_t)
    parts.append(f'<polyline points="{r1x:.1f},{r1y:.1f} {r2x:.1f},{r2y:.1f} {r3x:.1f},{r3y:.1f}" '
                 f'fill="none" stroke="#3b82f6" stroke-width="2" stroke-dasharray="6,3"/>')

    # Ridge marker
    parts.append(f'<line x1="{rx:.1f}" y1="{pad_t}" x2="{rx:.1f}" y2="{pad_t+H}" '
                 f'stroke="#f59e0b" stroke-width="1" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{rx+3:.1f}" y="{pad_t+12}" font-size="9" fill="#f59e0b" font-family="monospace">'
                 f'ridge={ridge:.1f}</text>')

    # Op dots
    for n in nodes:
        achievable = min(n["flops"] / 1e12, (n["arithmetic_intensity"] * peak_bw) / 1e12)
        cx, cy = lx(n["arithmetic_intensity"]), ly(max(achievable, 0.001))
        col = _CATEGORY_COLOR.get(n["category"], "#6b7280")
        title = _html.escape(f"{n['name']}\nAI: {n['arithmetic_intensity']:.2f}")
        parts.append(f'<g><title>{title}</title>'
                     f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{col}" '
                     f'fill-opacity="0.85" stroke="#111827" stroke-width="1"/></g>')

    # Axis labels
    parts.append(f'<text x="{pad_l+W//2}" y="{height-4}" text-anchor="middle" '
                 f'font-size="9" fill="#6b7280" font-family="monospace">Arithmetic Intensity (FLOP/byte)</text>')
    parts.append(f'<text x="10" y="{pad_t+H//2}" text-anchor="middle" '
                 f'font-size="9" fill="#6b7280" font-family="monospace" '
                 f'transform="rotate(-90,10,{pad_t+H//2})">TFLOPS</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _recommendations_html(passes: list[dict]) -> str:
    _TYPE_BADGE = {
        "fusion_opportunity":   ("bg:#1e3a5f", "blue",   "fusion"),
        "soft_barrier":         ("bg:#3f2a00", "orange", "barrier"),
        "checkpoint_candidate": ("bg:#3a1a00", "orange", "checkpoint"),
        "data_parallel":        ("bg:#14321a", "green",  "data∥"),
        "tensor_parallel":      ("bg:#003333", "cyan",   "tensor∥"),
        "fsdp":                 ("bg:#2a1a3a", "purple", "FSDP"),
    }
    rows = []
    for pr in passes:
        for rec in pr["recommendations"][:5]:
            t = rec.get("type", "")
            _, color, label = _TYPE_BADGE.get(t, ("", "gray", t))
            msg = _html.escape(rec.get("message", ""))
            hint = rec.get("code_hint", "")
            hint_html = (
                f'<br/><code style="color:#86efac;font-size:11px">{_html.escape(hint)}</code>'
                if hint else ""
            )
            rows.append(
                f'<tr>'
                f'<td style="padding:4px 8px;white-space:nowrap">'
                f'<span style="color:{color};font-size:10px;font-weight:bold">{label}</span></td>'
                f'<td style="padding:4px 8px;font-size:12px">{msg}{hint_html}</td>'
                f'</tr>'
            )

    if not rows:
        return '<p style="color:#6b7280;font-size:12px">No recommendations.</p>'

    return (
        '<table style="width:100%;border-collapse:collapse;font-family:monospace">'
        '<thead><tr>'
        '<th style="padding:4px 8px;color:#6b7280;font-size:10px;text-align:left">TYPE</th>'
        '<th style="padding:4px 8px;color:#6b7280;font-size:10px;text-align:left">RECOMMENDATION</th>'
        '</tr></thead>'
        '<tbody>' + "\n".join(rows) + '</tbody></table>'
    )


def _render_html(report: dict, fn_name: str, bwd_report: dict | None = None) -> str:
    r = report["roofline"]
    g = report["graph"].to_dict()
    passes_dicts = [p.to_dict() for p in report["passes"]]

    graph_svg    = _graph_svg(g)
    roofline_svg = _roofline_svg(g, r.to_dict())
    recs_html    = _recommendations_html(passes_dicts)

    bwd_section = ""
    if bwd_report:
        br = bwd_report["roofline"]
        bg = bwd_report["graph"].to_dict()
        bwd_section = f"""
        <div style="margin-top:16px">
          <h4 style="color:#a78bfa;margin:0 0 6px">Backward pass  ·  {bg['nodes'].__len__()} ops  ·
            {br.total_flops/1e9:.1f} GFLOPs  ·  {br.total_bytes/1e6:.1f} MB</h4>
          {_graph_svg(bg, width=820, height=200)}
        </div>
        """

    return f"""
<div style="background:#0f172a;color:#f1f5f9;font-family:monospace;padding:16px;border-radius:10px;border:1px solid #1e293b">
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:12px">
    <span style="color:#22d3ee;font-weight:bold;font-size:16px">HelixIR</span>
    <span style="color:#94a3b8;font-size:13px">{_html.escape(fn_name)}</span>
    <span style="color:#475569;font-size:11px">·  {r.device}</span>
  </div>

  <div style="display:grid;grid-template-columns:repeat(5,auto);gap:10px;margin-bottom:14px">
    {''.join(f'<div style="background:#1e293b;padding:6px 14px;border-radius:6px;text-align:center">'
             f'<div style="color:#64748b;font-size:9px;text-transform:uppercase">{lbl}</div>'
             f'<div style="color:#67e8f9;font-size:15px;font-weight:bold">{val}</div></div>'
             for lbl, val in [
                 ("ops",      report["num_ops"]),
                 ("GFLOPs",   f"{report['total_flops']/1e9:.2f}"),
                 ("MB",       f"{report['total_bytes']/1e6:.1f}"),
                 ("ridge",    f"{r.ridge_point:.1f}"),
                 ("device",   r.device),
             ])}
  </div>

  <div style="display:flex;gap:14px;flex-wrap:wrap">
    <div style="flex:1;min-width:420px">{graph_svg}</div>
    <div>{roofline_svg}</div>
  </div>

  <div style="margin-top:14px;background:#1e293b;border-radius:6px;padding:10px">
    <h4 style="color:#94a3b8;margin:0 0 8px;font-size:11px;text-transform:uppercase">Optimization Recommendations</h4>
    {recs_html}
  </div>

  {bwd_section}
</div>
"""


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------

def _error_html(title: str, detail: str = "", tb: str = "") -> str:
    detail_html = f'<div style="color:#fca5a5;font-size:12px;margin-top:6px">{_html.escape(detail)}</div>' if detail else ""
    tb_html = ""
    if tb:
        tb_html = (
            '<pre style="margin-top:10px;padding:8px;background:#1e293b;border-radius:4px;'
            'font-size:11px;color:#94a3b8;white-space:pre-wrap;overflow-x:auto">'
            f'{_html.escape(tb.strip())}</pre>'
        )
    return (
        '<div style="background:#0f172a;color:#f1f5f9;font-family:monospace;padding:12px 16px;'
        'border-radius:8px;border:1px solid #7f1d1d">'
        '<span style="color:#f87171;font-weight:bold">HelixIR error</span>'
        f'<span style="color:#94a3b8;font-size:12px;margin-left:10px">{_html.escape(title)}</span>'
        f'{detail_html}{tb_html}'
        '</div>'
    )


# ---------------------------------------------------------------------------
# IPython magic
# ---------------------------------------------------------------------------

def _parse_magic_line(line: str) -> tuple[str, list[str], dict]:
    """Parse '%helix fn_name arg1 arg2 --bwd --devices 8'."""
    parts = line.strip().split()
    fn_name = parts[0] if parts else ""
    args_names = []
    opts: dict[str, Any] = {"bwd": False, "devices": 8}
    i = 1
    while i < len(parts):
        if parts[i] == "--bwd":
            opts["bwd"] = True
        elif parts[i] == "--devices" and i + 1 < len(parts):
            opts["devices"] = int(parts[i + 1])
            i += 1
        else:
            args_names.append(parts[i])
        i += 1
    return fn_name, args_names, opts


def load_ipython_extension(ipython):
    """Called by %load_ext helix.jupyter."""
    from IPython.core.magic import register_line_magic, register_cell_magic
    from IPython.display import display, HTML

    def _show_error(title: str, detail: str = "", tb: str = "") -> None:
        display(HTML(_error_html(title, detail, tb)))

    @register_line_magic
    def helix(line):
        """
        %helix fn_name arg1 arg2 [--bwd] [--devices N]

        Analyse a JAX function using variables from the current namespace.
        """
        import helix as _helix

        fn_name, arg_names, opts = _parse_magic_line(line)
        ns = ipython.user_ns

        if fn_name not in ns:
            _show_error(f"'{fn_name}' not found in namespace")
            return

        fn   = ns[fn_name]
        args = []
        for a in arg_names:
            if a not in ns:
                _show_error(f"argument '{a}' not found in namespace")
                return
            args.append(ns[a])

        try:
            report = _helix.analyze(fn, *args, num_devices=opts["devices"])
        except Exception as exc:
            _show_error("Analysis failed", str(exc), traceback.format_exc())
            return

        bwd_report = None
        if opts["bwd"]:
            try:
                from helix.backward import analyze_backward
                bwd_report = analyze_backward(fn, *args)
            except Exception as exc:
                _show_error("Backward analysis failed", str(exc), traceback.format_exc())

        display(HTML(_render_html(report, fn_name, bwd_report)))

    @register_cell_magic
    def helix_cell(line, cell):
        """
        %%helix arg1 arg2 [--bwd]

        Define a function called 'fn' in the cell and analyse it immediately.
        """
        import helix as _helix

        arg_names, _, opts = _parse_magic_line("fn " + line)
        arg_names = arg_names  # fn was prepended
        ns = ipython.user_ns

        # Execute the cell to define the function
        ipython.run_cell(cell)
        if "fn" not in ipython.user_ns:
            _show_error("Define a function named 'fn' in the cell")
            return

        fn   = ipython.user_ns["fn"]
        args = []
        for a in arg_names:
            if a not in ns:
                _show_error(f"argument '{a}' not found in namespace")
                return
            args.append(ns[a])

        try:
            report = _helix.analyze(fn, *args, num_devices=opts.get("devices", 8))
        except Exception as exc:
            _show_error("Analysis failed", str(exc), traceback.format_exc())
            return

        display(HTML(_render_html(report, "fn")))

    # Register %%helix as the cell magic alias
    ipython.register_magic_function(helix_cell, magic_kind="cell", magic_name="helix")
