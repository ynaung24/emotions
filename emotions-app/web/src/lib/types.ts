// Mirrors app/schemas/evaluation.py. Keep in sync with the backend OpenAPI schema.

export interface Question {
  id: string;
  text: string;
  category: string;
}

export interface MetricResult {
  value: number;
  computed: boolean;
  label: string;
  detail: string | null;
}

export interface EmotionScore {
  label: string;
  score: number;
}

export interface DeliveryMetrics {
  words_per_minute: number;
  pause_ratio: number;
  filler_rate_per_100: number;
  pitch_variation: number;
  duration_seconds: number;
}

export interface FeedbackBlock {
  summary: string;
  strengths: string[];
  improvements: string[];
  source: "template" | "claude" | "openai";
}

export interface EvaluationResult {
  score: number;
  metrics: Record<string, MetricResult>;
  feedback: FeedbackBlock;
  matched_question: Question | null;
  scored_generically: boolean;
  emotions: EmotionScore[];
  transcript: string | null;
  delivery: DeliveryMetrics | null;
  flags: string[];
}

export type Mode = "text" | "voice";

export interface SessionState {
  question: string;
  questionId: string | null;
  mode: Mode;
  answer: string;
  result: EvaluationResult | null;
}
