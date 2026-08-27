import { Link } from "react-router-dom";
import { ArrowLeft, Info } from "lucide-react";
import { useSession } from "@/store/session";
import { Badge, Card, CardBody, CardHeader, CardTitle } from "@/components/ui/primitives";
import { ScoreRing } from "@/components/results/ScoreRing";
import { MetricRadar } from "@/components/results/MetricRadar";
import { MetricGrid } from "@/components/results/MetricGrid";
import { FeedbackList } from "@/components/results/FeedbackList";
import { DeliveryStats, EmotionBars, TranscriptCard } from "@/components/results/VoicePanels";

export function Results() {
  const { result, question, answer } = useSession();

  if (!result) {
    return (
      <Card>
        <CardBody className="space-y-3 py-10 text-center">
          <p className="text-muted">No evaluation to show yet.</p>
          <Link
            to="/"
            className="mx-auto inline-flex h-10 w-fit items-center rounded-lg bg-brand-500 px-4 text-sm font-medium text-white hover:bg-brand-600"
          >
            Start a session
          </Link>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-5">
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-muted hover:text-[var(--text)]">
        <ArrowLeft className="size-4" /> New session
      </Link>

      <Card>
        <CardBody className="flex flex-col items-center gap-5 py-6 sm:flex-row sm:items-center sm:gap-8">
          <ScoreRing value={result.score} />
          <div className="flex-1 space-y-2 text-center sm:text-left">
            <div className="flex flex-wrap items-center justify-center gap-2 sm:justify-start">
              {result.matched_question ? (
                <Badge tone="brand">{result.matched_question.category}</Badge>
              ) : (
                <Badge tone="warn">scored generically</Badge>
              )}
              {result.flags
                .filter((f) => f !== "scored_generically")
                .map((f) => (
                  <Badge key={f} tone="warn">
                    {f.replace(/_/g, " ")}
                  </Badge>
                ))}
            </div>
            <p className="text-sm font-medium">{question}</p>
            <p className="line-clamp-3 text-sm text-muted">{answer}</p>
          </div>
        </CardBody>
      </Card>

      {result.scored_generically && (
        <div className="flex items-start gap-2 rounded-lg surface-2 px-3 py-2 text-xs text-muted">
          <Info className="mt-0.5 size-4 shrink-0" />
          This question isn’t in the reference set, so completeness was scored against
          points derived from the question itself rather than a curated list.
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Metric profile</CardTitle>
          </CardHeader>
          <CardBody>
            <MetricRadar metrics={result.metrics} />
          </CardBody>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Breakdown</CardTitle>
          </CardHeader>
          <CardBody>
            <MetricGrid metrics={result.metrics} />
          </CardBody>
        </Card>
      </div>

      <FeedbackList feedback={result.feedback} />

      {result.transcript !== null && <TranscriptCard transcript={result.transcript} />}
      {result.delivery && <DeliveryStats delivery={result.delivery} />}
      <EmotionBars emotions={result.emotions} />
    </div>
  );
}
