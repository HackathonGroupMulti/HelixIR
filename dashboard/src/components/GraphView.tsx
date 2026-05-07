/**
 * Force-directed op-graph rendered with D3.
 *
 * Node colour encodes category:
 *   blue    = matmul   (compute-heavy)
 *   green   = elementwise
 *   orange  = reduction
 *   red     = memory-intensive
 *   purple  = collective (cross-device)
 *   gray    = other
 *
 * Node size encodes total output bytes (clamped to [6, 24] px radius).
 * A dashed ring marks bandwidth-bound nodes; solid = compute-bound.
 */
import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { OpGraph, OpNode, OpEdge } from "../types";

const CATEGORY_COLOR: Record<string, string> = {
  matmul:      "#3b82f6",
  elementwise: "#22c55e",
  reduction:   "#f97316",
  memory:      "#ef4444",
  collective:  "#a855f7",
  other:       "#6b7280",
};

interface Props {
  graph: OpGraph;
  width?: number;
  height?: number;
}

interface D3Node extends d3.SimulationNodeDatum {
  id: number;
  node: OpNode;
}

interface D3Link extends d3.SimulationLinkDatum<D3Node> {
  edge: OpEdge;
}

export function GraphView({ graph, width = 900, height = 500 }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !graph.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const g = svg.append("g");

    // Zoom + pan
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.2, 4])
        .on("zoom", (ev) => g.attr("transform", ev.transform))
    );

    // Arrow marker
    svg.append("defs").append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 18).attr("refY", 0)
      .attr("markerWidth", 6).attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#4b5563");

    const nodes: D3Node[] = graph.nodes.map((n) => ({ id: n.id, node: n }));
    const nodeById = new Map(nodes.map((n) => [n.id, n]));

    const links: D3Link[] = graph.edges
      .map((e) => ({
        source: nodeById.get(e.src)!,
        target: nodeById.get(e.dst)!,
        edge: e,
      }))
      .filter((l) => l.source && l.target);

    const radiusScale = d3.scaleSqrt()
      .domain([0, d3.max(graph.nodes, (n) => n.bytes_written) ?? 1])
      .range([5, 22]);

    const sim = d3.forceSimulation(nodes)
      .force("link",   d3.forceLink<D3Node, D3Link>(links).id((d) => d.id).distance(60))
      .force("charge", d3.forceManyBody().strength(-180))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(28));

    const link = g.append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", "#374151")
      .attr("stroke-width", 1.5)
      .attr("marker-end", "url(#arrow)");

    const node = (g.append("g")
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("cursor", "pointer") as d3.Selection<SVGGElement, D3Node, SVGGElement, unknown>)
      .call(
        d3.drag<SVGGElement, D3Node>()
          .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
          .on("end",   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    node.append("circle")
      .attr("r", (d) => radiusScale(d.node.bytes_written))
      .attr("fill", (d) => CATEGORY_COLOR[d.node.category] ?? "#6b7280")
      .attr("fill-opacity", 0.85)
      .attr("stroke", (d) => d.node.is_compute_bound ? "#f9fafb" : "#f97316")
      .attr("stroke-width", (d) => d.node.is_compute_bound ? 1 : 2)
      .attr("stroke-dasharray", (d) => d.node.is_compute_bound ? "none" : "4,2");

    node.append("text")
      .text((d) => d.node.name.length > 12 ? d.node.name.slice(0, 11) + "…" : d.node.name)
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("font-size", 9)
      .attr("fill", "#f9fafb")
      .attr("pointer-events", "none");

    // Tooltip
    const tooltip = d3.select("body").append("div")
      .style("position", "absolute")
      .style("background", "#1f2937")
      .style("border", "1px solid #374151")
      .style("border-radius", "6px")
      .style("padding", "8px 12px")
      .style("font-size", "12px")
      .style("color", "#f9fafb")
      .style("pointer-events", "none")
      .style("opacity", 0);

    node
      .on("mouseover", (ev, d) => {
        const n = d.node;
        tooltip.transition().duration(100).style("opacity", 1);
        tooltip.html(
          `<strong>${n.name}</strong> (${n.category})<br/>` +
          `Out: ${n.output_shapes.map((s) => s.join("×")).join(", ")}  ${n.dtype}<br/>` +
          `FLOPs: ${(n.flops / 1e6).toFixed(2)} MFLOPs<br/>` +
          `Bytes R/W: ${(n.bytes_read / 1e6).toFixed(1)} / ${(n.bytes_written / 1e6).toFixed(1)} MB<br/>` +
          `AI: ${n.arithmetic_intensity.toFixed(2)} FLOP/byte<br/>` +
          `<em>${n.is_compute_bound ? "compute-bound" : "bandwidth-bound"}</em>`
        );
      })
      .on("mousemove", (ev) => {
        tooltip.style("left", (ev.pageX + 12) + "px").style("top", (ev.pageY - 24) + "px");
      })
      .on("mouseout", () => tooltip.transition().duration(150).style("opacity", 0));

    sim.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as D3Node).x!)
        .attr("y1", (d) => (d.source as D3Node).y!)
        .attr("x2", (d) => (d.target as D3Node).x!)
        .attr("y2", (d) => (d.target as D3Node).y!);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    return () => {
      sim.stop();
      tooltip.remove();
    };
  }, [graph, width, height]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="rounded-lg bg-gray-900 border border-gray-700"
    />
  );
}
