/**
 * Log-log roofline model chart.
 *
 * X axis: arithmetic intensity (FLOP / byte)
 * Y axis: achievable throughput (TFLOPS)
 *
 * The roofline is drawn as two line segments meeting at the ridge point:
 *   left  segment = bandwidth roof  (slope = peak_bandwidth / 1e12 TFLOPS/FLOP*byte)
 *   right segment = compute roof    (flat line at peak_flops)
 *
 * Individual ops are plotted as dots at their (intensity, achievable) position.
 */
import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { OpGraph, RooflineResult } from "../types";

interface Props {
  graph: OpGraph;
  roofline: RooflineResult;
  width?: number;
  height?: number;
}

export function RooflineChart({ graph, roofline, width = 600, height = 380 }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const margin = { top: 24, right: 24, bottom: 52, left: 64 };
    const W = width  - margin.left - margin.right;
    const H = height - margin.top  - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Ranges
    const intensities = graph.nodes.map((n) => n.arithmetic_intensity).filter((v) => v > 0);
    const xMin = Math.max(0.01, d3.min(intensities) ?? 0.01);
    const xMax = Math.max(roofline.ridge_point * 10, d3.max(intensities) ?? roofline.ridge_point * 10);

    const x = d3.scaleLog().domain([xMin, xMax]).range([0, W]);
    const yMax = (roofline.peak_flops / 1e12) * 1.5;
    const y = d3.scaleLog().domain([0.001, yMax]).range([H, 0]);

    // Axes
    g.append("g").attr("transform", `translate(0,${H})`)
      .call(d3.axisBottom(x).ticks(6, ".1e"))
      .selectAll("text").attr("fill", "#9ca3af");
    g.selectAll(".domain, .tick line").attr("stroke", "#374151");

    g.append("g")
      .call(d3.axisLeft(y).ticks(5, ".0e"))
      .selectAll("text").attr("fill", "#9ca3af");

    // Axis labels
    g.append("text")
      .attr("x", W / 2).attr("y", H + 44)
      .attr("text-anchor", "middle")
      .attr("fill", "#9ca3af").attr("font-size", 12)
      .text("Arithmetic Intensity (FLOP / byte)");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -H / 2).attr("y", -52)
      .attr("text-anchor", "middle")
      .attr("fill", "#9ca3af").attr("font-size", 12)
      .text("Achievable Throughput (TFLOPS)");

    // Roofline curve
    const peakT = roofline.peak_flops / 1e12;
    const ridgeX = roofline.ridge_point;

    const roofPoints: [number, number][] = [
      [xMin, (xMin * roofline.peak_bandwidth) / 1e12],
      [ridgeX, peakT],
      [xMax, peakT],
    ];

    const line = d3.line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(Math.max(d[1], 0.001)));

    g.append("path")
      .datum(roofPoints)
      .attr("d", line)
      .attr("fill", "none")
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 2.5)
      .attr("stroke-dasharray", "6,3");

    // Ridge point marker
    g.append("line")
      .attr("x1", x(ridgeX)).attr("x2", x(ridgeX))
      .attr("y1", 0).attr("y2", H)
      .attr("stroke", "#f59e0b").attr("stroke-width", 1).attr("stroke-dasharray", "4,3");

    g.append("text")
      .attr("x", x(ridgeX) + 4).attr("y", 16)
      .attr("fill", "#f59e0b").attr("font-size", 10)
      .text(`ridge ${ridgeX.toFixed(1)}`);

    // Op dots
    const tooltip = d3.select("body").append("div")
      .style("position", "absolute")
      .style("background", "#1f2937")
      .style("border", "1px solid #374151")
      .style("border-radius", "6px")
      .style("padding", "6px 10px")
      .style("font-size", "11px")
      .style("color", "#f9fafb")
      .style("pointer-events", "none")
      .style("opacity", 0);

    const COLORS: Record<string, string> = {
      matmul:      "#3b82f6",
      elementwise: "#22c55e",
      reduction:   "#f97316",
      memory:      "#ef4444",
      collective:  "#a855f7",
      other:       "#6b7280",
    };

    const dots = graph.nodes.filter((n) => n.arithmetic_intensity > 0 && n.flops > 0);

    g.selectAll("circle.op")
      .data(dots)
      .join("circle")
      .attr("class", "op")
      .attr("cx", (d) => x(Math.max(xMin, d.arithmetic_intensity)))
      .attr("cy", (d) => {
        const achievable = Math.min(d.flops / 1e12, (d.arithmetic_intensity * roofline.peak_bandwidth) / 1e12);
        return y(Math.max(0.001, achievable));
      })
      .attr("r", 5)
      .attr("fill", (d) => COLORS[d.category] ?? "#6b7280")
      .attr("fill-opacity", 0.8)
      .attr("stroke", "#111827").attr("stroke-width", 1)
      .on("mouseover", (ev, d) => {
        tooltip.transition().duration(100).style("opacity", 1);
        tooltip.html(
          `<strong>${d.name}</strong><br/>` +
          `AI: ${d.arithmetic_intensity.toFixed(2)} FLOP/byte<br/>` +
          `${d.is_compute_bound ? "compute-bound" : "bandwidth-bound"}`
        );
      })
      .on("mousemove", (ev) => tooltip.style("left", ev.pageX + 12 + "px").style("top", ev.pageY - 24 + "px"))
      .on("mouseout", () => tooltip.transition().duration(150).style("opacity", 0));

    return () => { tooltip.remove(); };
  }, [graph, roofline, width, height]);

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-400 mb-2">Roofline Model · {roofline.device}</h3>
      <svg ref={svgRef} width={width} height={height} className="rounded-lg bg-gray-900 border border-gray-700" />
    </div>
  );
}
