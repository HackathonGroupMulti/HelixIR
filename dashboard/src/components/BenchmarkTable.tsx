import type { BenchmarkResult } from "../types";

interface Props {
  results: BenchmarkResult[];
  onClear: () => void;
}

export function BenchmarkTable({ results, onClear }: Props) {
  if (!results.length) {
    return (
      <div className="text-gray-500 text-sm py-6 text-center">
        No benchmarks yet. Run <code className="text-cyan-400">helix benchmark</code> to populate.
      </div>
    );
  }

  const baseline = results[0];

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-400">Benchmark Results</h3>
        <button
          onClick={onClear}
          className="text-xs text-gray-500 hover:text-red-400 transition-colors"
        >
          Clear
        </button>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-700">
        <table className="w-full text-sm">
          <thead className="bg-gray-800 text-gray-400">
            <tr>
              {["Name", "Mean (ms)", "Std", "Min", "Max", "TFLOPS", "Eff%", "Speedup"].map((h) => (
                <th key={h} className="px-3 py-2 text-left font-medium text-xs">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => {
              const speedup = baseline.mean_ms / r.mean_ms;
              const isBaseline = i === 0;
              return (
                <tr key={i} className="border-t border-gray-800 hover:bg-gray-800/50 transition-colors">
                  <td className="px-3 py-2 font-medium text-cyan-300">{r.name}</td>
                  <td className="px-3 py-2 tabular-nums">{r.mean_ms.toFixed(3)}</td>
                  <td className="px-3 py-2 tabular-nums text-gray-500">±{r.std_ms.toFixed(3)}</td>
                  <td className="px-3 py-2 tabular-nums text-gray-500">{r.min_ms.toFixed(3)}</td>
                  <td className="px-3 py-2 tabular-nums text-gray-500">{r.max_ms.toFixed(3)}</td>
                  <td className="px-3 py-2 tabular-nums">
                    {r.achieved_tflops > 0 ? r.achieved_tflops.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {r.efficiency_pct > 0 ? (
                      <span
                        className={
                          r.efficiency_pct > 60 ? "text-green-400" :
                          r.efficiency_pct > 30 ? "text-yellow-400" : "text-red-400"
                        }
                      >
                        {r.efficiency_pct.toFixed(1)}%
                      </span>
                    ) : "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {isBaseline ? (
                      <span className="text-gray-500">baseline</span>
                    ) : (
                      <span className={speedup >= 1 ? "text-green-400" : "text-red-400"}>
                        {speedup.toFixed(2)}×
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
