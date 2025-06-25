export interface EvaluationMetrics {
  relevance: number;
  clarity: number;
  completeness: number;
  confidence?: number;
  conciseness?: number;
  [key: string]: number | undefined;
}

export interface EmotionScores {
  confidence: number;
  enthusiasm: number;
  professionalism: number;
  clarity: number;
  [key: string]: number;
}

export interface EvaluationResponse {
  score: number;
  metrics: EvaluationMetrics;
  feedback: string | string[];
  emotions?: EmotionScores;
  transcribedText?: string;
}

export interface EvaluationResult extends EvaluationResponse {
  question: string;
  timestamp: string;
  isVoiceEvaluation?: boolean;
}
