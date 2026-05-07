import type { PassResult, Recommendation } from "../types";

const TYPE_COLOR: Record<string, string> = {
  fusion_opportunity: "text-blue-400",
  soft_barrier:       "text-yellow-400",
  checkpoint_candidate: "text-orange-400",
  data_parallel:      "text-green-400",
  tensor_parallel:    "text-cyan-400",
  fsdp:               "text-purple-400",
};

function RecCard({ rec }: { rec: Recommendation }) {
  const color = TYPE_COLOR[rec.type] ?? "text-gray-400";
  return (
    <div className="border border-gray-700 rounded-md p-3 bg-gray-800/60 space-y-1">
      <div className={`text-xs font-semibold uppercase tracking-wide ${color}`}>{rec.type}</div>
      <p className="text-sm text-gray-200">{rec.message}</p>
      {rec.code_hint && (
        <pre className="mt-1 text-xs bg-gray-900 text-green-300 rounded p-2 overflow-x-auto">
          {rec.code_hint}
        </pre>
      )}
    </div>
  );
}

interface Props {
  passes: PassResult[];
}

export function PassPanel({ passes }: Props) {
  return (
    <div className="space-y-6">
      {passes.map((pr) => (
        <div key={pr.pass_name}>
          <div className="flex items-baseline gap-3 mb-2">
            <h3 className="text-sm font-bold text-gray-200">{pr.pass_name}</h3>
            <span className="text-xs text-gray-500">{pr.summary}</span>
          </div>
          <div className="grid grid-cols-1 gap-2">
            {pr.recommendations.slice(0, 6).map((rec, i) => (
              <RecCard key={i} rec={rec} />
            ))}
            {pr.recommendations.length > 6 && (
              <p className="text-xs text-gray-500 pl-1">
                +{pr.recommendations.length - 6} more recommendations
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
