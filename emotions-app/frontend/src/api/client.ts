import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Question {
  id: string;
  text: string;
}

export interface EvaluationMetrics {
  relevance: number;
  clarity: number;
  completeness: number;
  [key: string]: number;
}

export interface EvaluationResponse {
  score: number;
  metrics: EvaluationMetrics;
  feedback: string;
  emotions?: Record<string, number>;
}

// API functions
export const fetchQuestions = async (): Promise<{ questions: string[] }> => {
  const response = await apiClient.get('/questions');
  return response.data;
};

export const evaluateText = async (
  text: string,
  question: string
): Promise<EvaluationResponse> => {
  const response = await apiClient.post('/evaluate/text', { text, question });
  return response.data;
};

export const evaluateAudio = async (
  audioFile: File,
  question: string
): Promise<EvaluationResponse> => {
  const formData = new FormData();
  formData.append('file', audioFile);
  formData.append('question', question);

  const response = await apiClient.post('/evaluate/audio', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};
