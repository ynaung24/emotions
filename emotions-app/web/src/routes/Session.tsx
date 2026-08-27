import { useState } from "react";
import { QuestionPicker } from "@/components/QuestionPicker";
import { AnswerPanel } from "@/components/AnswerPanel";
import { session, useSession } from "@/store/session";

export function Session() {
  const s = useSession();
  const [question, setQuestion] = useState(s.question);
  const [questionId, setQuestionId] = useState(s.questionId);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold">Practice an interview answer</h1>
        <p className="mt-1 text-sm text-muted">
          Pick a question, answer by typing or recording, and get a scored breakdown
          across relevance, completeness, specificity, structure and delivery.
        </p>
      </div>

      <QuestionPicker
        value={question}
        questionId={questionId}
        onChange={(text, id) => {
          setQuestion(text);
          setQuestionId(id);
          session.setQuestion(text, id);
        }}
      />

      {question.trim().length > 0 && (
        <AnswerPanel
          question={question}
          mode={s.mode}
          answer={s.answer}
          onModeChange={session.setMode}
          onAnswerChange={session.setAnswer}
        />
      )}
    </div>
  );
}
