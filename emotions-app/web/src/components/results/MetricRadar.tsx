import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
} from "recharts";
import type { MetricResult } from "@/lib/types";

export function MetricRadar({ metrics }: { metrics: Record<string, MetricResult> }) {
  const data = Object.values(metrics)
    .filter((m) => m.computed)
    .map((m) => ({ metric: m.label.replace(/ \(.*\)/, ""), value: Math.round(m.value) }));

  if (data.length < 3) return null;

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <RadarChart data={data} outerRadius="68%" margin={{ top: 8, right: 24, bottom: 8, left: 24 }}>
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis
            dataKey="metric"
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          />
          <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
          <Radar
            dataKey="value"
            stroke="#0080ff"
            fill="#0080ff"
            fillOpacity={0.35}
            isAnimationActive={false}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
