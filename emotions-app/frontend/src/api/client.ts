import axios, { AxiosError } from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Log API requests and responses
const requestLogger = (config: any) => {
  console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data || '');
  return config;
};

const responseLogger = (response: any) => {
  console.log(`[API] ${response.status} ${response.config.url}`, response.data);
  return response;
};

const errorLogger = (error: any) => {
  if (axios.isAxiosError(error)) {
    console.error(
      `[API Error] ${error.config?.method?.toUpperCase()} ${error.config?.url}`,
      {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message
      }
    );
  } else {
    console.error('[API Error]', error);
  }
  return Promise.reject(error);
};

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout
});

// Add request and response interceptors
apiClient.interceptors.request.use(requestLogger);
apiClient.interceptors.response.use(responseLogger, errorLogger);

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
