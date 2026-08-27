import { scoreColor, scoreStroke } from "@/lib/utils";

export function ScoreRing({ value }: { value: number }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - Math.max(0, Math.min(100, value)) / 100);

  return (
    <div className="relative grid size-32 place-items-center">
      <svg viewBox="0 0 120 120" className="size-32 -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke="var(--surface-2, #f1f5f9)" strokeWidth="10" />
        <circle
          cx="60"
          cy="60"
          r={r}
          fill="none"
          stroke={scoreStroke(value)}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          className="transition-[stroke-dashoffset] duration-700"
        />
      </svg>
      <div className="absolute text-center">
        <div className={`text-3xl font-bold ${scoreColor(value)}`}>{Math.round(value)}</div>
        <div className="text-[10px] uppercase tracking-wide text-muted">out of 100</div>
      </div>
    </div>
  );
}
