import type { MetricResult } from "@/lib/types";
import { Meter } from "../ui/primitives";
import { scoreColor } from "@/lib/utils";

const ORDER = [
  "relevance",
  "completeness",
  "specificity",
  "structure",
  "clarity",
  "confidence",
  "conciseness",
];

const rank = (k: string) => (ORDER.indexOf(k) === -1 ? 99 : ORDER.indexOf(k));

export function MetricGrid({ metrics }: { metrics: Record<string, MetricResult> }) {
  const entries = Object.entries(metrics).sort((a, b) => rank(a[0]) - rank(b[0]));

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {entries.map(([key, m]) => (
        <div key={key} className="rounded-lg surface-2 p-4">
          <div className="flex items-baseline justify-between">
            <span className="text-sm font-medium">{m.label}</span>
            {m.computed ? (
              <span className={`text-lg font-bold tabular-nums ${scoreColor(m.value)}`}>
                {Math.round(m.value)}
              </span>
            ) : (
              <span className="text-xs text-muted">not measured</span>
            )}
          </div>
          {m.computed && <Meter value={m.value} className="mt-2" />}
          {m.detail && <p className="mt-2 text-xs text-muted">{m.detail}</p>}
        </div>
      ))}
    </div>
  );
}
