import type { DeliveryMetrics, EmotionScore } from "@/lib/types";
import { Card, CardBody, CardHeader, CardTitle, Meter } from "../ui/primitives";

export function TranscriptCard({ transcript }: { transcript: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Transcript</CardTitle>
      </CardHeader>
      <CardBody>
        <p className="text-sm leading-relaxed text-muted">{transcript || "(nothing transcribed)"}</p>
      </CardBody>
    </Card>
  );
}

export function DeliveryStats({ delivery }: { delivery: DeliveryMetrics }) {
  const rows: [string, string, string][] = [
    ["Pace", `${Math.round(delivery.words_per_minute)} wpm`, wpmHint(delivery.words_per_minute)],
    ["Pauses", `${Math.round(delivery.pause_ratio * 100)}% of the time`, ""],
    ["Fillers", `${delivery.filler_rate_per_100.toFixed(1)} per 100 words`, ""],
    ["Pitch variation", delivery.pitch_variation.toFixed(2), pitchHint(delivery.pitch_variation)],
    ["Length", `${Math.round(delivery.duration_seconds)}s`, ""],
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Delivery</CardTitle>
      </CardHeader>
      <CardBody>
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
          {rows.map(([label, value, hint]) => (
            <div key={label} className="flex items-baseline justify-between gap-3 border-b border-[var(--border)] py-1.5 last:border-0">
              <dt className="text-sm text-muted">{label}</dt>
              <dd className="text-right text-sm font-medium">
                {value}
                {hint && <span className="ml-1 text-xs font-normal text-muted">({hint})</span>}
              </dd>
            </div>
          ))}
        </dl>
      </CardBody>
    </Card>
  );
}

export function EmotionBars({ emotions }: { emotions: EmotionScore[] }) {
  if (emotions.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Emotional tone</CardTitle>
      </CardHeader>
      <CardBody className="space-y-2">
        {emotions.map((e) => (
          <div key={e.label}>
            <div className="flex justify-between text-sm">
              <span className="capitalize">{e.label}</span>
              <span className="tabular-nums text-muted">{Math.round(e.score * 100)}%</span>
            </div>
            <Meter value={e.score * 100} className="mt-1" />
          </div>
        ))}
        <p className="pt-1 text-xs text-muted">
          GoEmotions classification of the answer text.
        </p>
      </CardBody>
    </Card>
  );
}

function wpmHint(w: number) {
  if (w < 110) return "a little slow";
  if (w > 180) return "quite fast";
  return "good range";
}
function pitchHint(p: number) {
  if (p < 0.08) return "monotone";
  if (p > 0.35) return "very varied";
  return "expressive";
}
