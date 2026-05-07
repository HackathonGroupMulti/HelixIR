export interface OpNode {
  id: number;
  name: string;
  category: "elementwise" | "reduction" | "matmul" | "memory" | "collective" | "other";
  input_shapes: number[][];
  output_shapes: number[][];
  dtype: string;
  flops: number;
  bytes_read: number;
  bytes_written: number;
  arithmetic_intensity: number;
  is_compute_bound: boolean;
  fusion_group: number | null;
}

export interface OpEdge {
  src: number;
  dst: number;
  shape: number[];
  dtype: string;
}

export interface OpGraph {
  nodes: OpNode[];
  edges: OpEdge[];
  input_shapes: number[][];
  output_shapes: number[][];
}

export interface RooflineResult {
  device: string;
  peak_flops: number;
  peak_bandwidth: number;
  ridge_point: number;
  compute_bound_ops: string[];
  bandwidth_bound_ops: string[];
  total_flops: number;
  total_bytes: number;
}

export interface Recommendation {
  type: string;
  message: string;
  code_hint?: string;
  ops?: string[];
  estimated_savings_mb?: number;
  size_mb?: number;
  shape?: number[];
  [key: string]: unknown;
}

export interface PassResult {
  pass_name: string;
  summary: string;
  recommendations: Recommendation[];
}

export interface ProfileReport {
  graph: OpGraph;
  roofline: RooflineResult;
  passes: PassResult[];
  num_ops: number;
  total_flops: number;
  total_bytes: number;
}

export interface BenchmarkResult {
  name: string;
  mean_ms: number;
  std_ms: number;
  min_ms: number;
  max_ms: number;
  iterations: number;
  flops: number;
  achieved_tflops: number;
  peak_tflops: number;
  efficiency_pct: number;
}

export type WsMessage =
  | { event: "report"; data: ProfileReport }
  | { event: "benchmark"; data: BenchmarkResult }
  | { event: "benchmarks_init"; data: BenchmarkResult[] }
  | { event: "benchmarks_cleared"; data: Record<string, never> };
