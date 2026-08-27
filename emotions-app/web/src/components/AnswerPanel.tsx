import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { AlertCircle, Loader2, PenLine, Send } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { blobToWav } from "@/lib/audio";
import { session } from "@/store/session";
import type { Mode } from "@/lib/types";
import { Button, Card, CardBody, Textarea } from "./ui/primitives";
import { Recorder } from "./Recorder";
import { cn } from "@/lib/utils";

interface Props {
  question: string;
  mode: Mode;
  answer: string;
  onModeChange: (m: Mode) => void;
  onAnswerChange: (a: string) => void;
}

export function AnswerPanel({ question, mode, answer, onModeChange, onAnswerChange }: Props) {
  const navigate = useNavigate();
  const [recording, setRecording] = useState<Blob | null>(null);

  const evaluate = useMutation({
    mutationFn: async () => {
      if (mode === "text") return api.evaluateText(question, answer.trim());
      const wav = await blobToWav(recording!);
      return api.evaluateAudio(question, wav);
    },
    onSuccess: (result) => {
      session.setResult(result, result.transcript || answer);
      navigate("/results");
    },
  });

  const canSubmit =
    question.trim().length > 0 &&
    (mode === "text" ? answer.trim().length >= 10 : recording !== null) &&
    !evaluate.isPending;

  return (
    <Card>
      <CardBody className="space-y-4 pt-5">
        <div className="flex gap-1 rounded-lg surface-2 p-1">
          {(["text", "voice"] as const).map((m) => (
            <button
              key={m}
              onClick={() => onModeChange(m)}
              className={cn(
                "flex-1 rounded-md px-3 py-1.5 text-sm font-medium capitalize transition-colors",
                mode === m ? "surface shadow-sm" : "text-muted hover:text-[var(--text)]",
              )}
            >
              {m === "text" ? "Type answer" : "Record answer"}
            </button>
          ))}
        </div>

        {mode === "text" ? (
          <>
            <Textarea
              placeholder="Answer as you would in the interview…"
              value={answer}
              onChange={(e) => onAnswerChange(e.target.value)}
              className="min-h-[12rem]"
            />
            <div className="flex items-center gap-1 text-xs text-muted">
              <PenLine className="size-3" />
              {answer.trim().split(/\s+/).filter(Boolean).length} words
            </div>
          </>
        ) : (
          <Recorder disabled={evaluate.isPending} onRecorded={setRecording} />
        )}

        {evaluate.isError && (
          <div className="flex items-start gap-2 rounded-lg bg-rose-100 px-3 py-2 text-sm text-rose-800 dark:bg-rose-950/50 dark:text-rose-300">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <span>
              {evaluate.error instanceof ApiError
                ? evaluate.error.message
                : "Something went wrong. Please try again."}
            </span>
          </div>
        )}

        <Button size="lg" disabled={!canSubmit} onClick={() => evaluate.mutate()} className="w-full sm:w-auto">
          {evaluate.isPending ? (
            <>
              <Loader2 className="size-4 animate-spin" />
              {mode === "voice" ? "Transcribing & scoring…" : "Scoring…"}
            </>
          ) : (
            <>
              <Send className="size-4" /> Evaluate answer
            </>
          )}
        </Button>
        {evaluate.isPending && (
          <p className="text-xs text-muted">
            First run can take 20-40s while the models load. Later answers are fast.
          </p>
        )}
      </CardBody>
    </Card>
  );
}
