import { useState, useCallback } from "react";
import type { ProfileReport, BenchmarkResult, WsMessage } from "./types";
import { useWebSocket } from "./hooks/useWebSocket";
import { GraphView } from "./components/GraphView";
import { RooflineChart } from "./components/RooflineChart";
import { BenchmarkTable } from "./components/BenchmarkTable";
import { PassPanel } from "./components/PassPanel";

const WS_URL = `ws://${window.location.host}/ws`;

type Tab = "graph" | "roofline" | "passes" | "benchmarks";

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-800 rounded-lg px-4 py-2 text-center">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-lg font-bold text-cyan-300 tabular-nums">{value}</div>
    </div>
  );
}

export default function App() {
  const [report, setReport] = useState<ProfileReport | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([]);
  const [connected, setConnected] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("graph");

  const handleMessage = useCallback((msg: WsMessage) => {
    setConnected(true);
    if (msg.event === "report") {
      setReport(msg.data);
    } else if (msg.event === "benchmark") {
      setBenchmarks((prev) => [...prev, msg.data]);
    } else if (msg.event === "benchmarks_init") {
      setBenchmarks(msg.data);
    } else if (msg.event === "benchmarks_cleared") {
      setBenchmarks([]);
    }
  }, []);

  useWebSocket(WS_URL, handleMessage);

  const clearBenchmarks = useCallback(async () => {
    await fetch("/api/benchmarks", { method: "DELETE" });
  }, []);

  const r = report?.roofline;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-mono">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-cyan-400 font-bold text-lg tracking-tight">HelixIR</span>
          <span className="text-gray-600 text-xs">JAX Performance Optimizer</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-400" : "bg-red-500"}`} />
          <span className="text-gray-500">{connected ? "live" : "disconnected"}</span>
        </div>
      </header>

      <main className="px-6 py-4 space-y-4">
        {/* Stats row */}
        {report && r && (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <StatPill label="Device"    value={r.device} />
            <StatPill label="Ops"       value={String(report.num_ops)} />
            <StatPill label="GFLOPs"    value={(report.total_flops / 1e9).toFixed(1)} />
            <StatPill label="MB"        value={(report.total_bytes / 1e6).toFixed(1)} />
            <StatPill label="Ridge pt"  value={r.ridge_point.toFixed(1) + " F/B"} />
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 border-b border-gray-800 pb-0">
          {(["graph", "roofline", "passes", "benchmarks"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setActiveTab(t)}
              className={`px-4 py-2 text-sm rounded-t transition-colors ${
                activeTab === t
                  ? "bg-gray-800 text-cyan-300 border-b-2 border-cyan-400"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="min-h-[480px]">
          {!report && activeTab !== "benchmarks" && (
            <div className="flex items-center justify-center h-64 text-gray-600 text-sm">
              Waiting for profile data…
              <br />
              Run <code className="text-cyan-400 ml-1">helix profile &lt;script.py&gt; --push</code>
            </div>
          )}

          {report && activeTab === "graph" && (
            <GraphView graph={report.graph} width={960} height={520} />
          )}

          {report && activeTab === "roofline" && (
            <RooflineChart graph={report.graph} roofline={report.roofline} width={700} height={420} />
          )}

          {report && activeTab === "passes" && (
            <PassPanel passes={report.passes} />
          )}

          {activeTab === "benchmarks" && (
            <BenchmarkTable results={benchmarks} onClear={clearBenchmarks} />
          )}
        </div>
      </main>
    </div>
  );
}
