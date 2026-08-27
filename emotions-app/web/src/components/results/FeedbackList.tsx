import { Check, Lightbulb } from "lucide-react";
import type { FeedbackBlock } from "@/lib/types";
import { Badge, Card, CardBody, CardHeader, CardTitle } from "../ui/primitives";

export function FeedbackList({ feedback }: { feedback: FeedbackBlock }) {
  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Feedback</CardTitle>
        <Badge tone={feedback.source === "template" ? "neutral" : "brand"}>
          {feedback.source === "template" ? "rule-based" : `${feedback.source} judge`}
        </Badge>
      </CardHeader>
      <CardBody className="space-y-4">
        <p className="text-sm leading-relaxed">{feedback.summary}</p>

        {feedback.strengths.length > 0 && (
          <div>
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
              What worked
            </h4>
            <ul className="space-y-1.5">
              {feedback.strengths.map((s, i) => (
                <li key={i} className="flex gap-2 text-sm">
                  <Check className="mt-0.5 size-4 shrink-0 text-emerald-500" />
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}

        {feedback.improvements.length > 0 && (
          <div>
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-amber-600 dark:text-amber-400">
              To improve
            </h4>
            <ul className="space-y-1.5">
              {feedback.improvements.map((s, i) => (
                <li key={i} className="flex gap-2 text-sm">
                  <Lightbulb className="mt-0.5 size-4 shrink-0 text-amber-500" />
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardBody>
    </Card>
  );
}
