import type { EvaluationResult, Question } from "./types";

// dev: Vite proxies /api -> http://localhost:8000 (see vite.config.ts)
// prod: set VITE_API_BASE to the backend origin
const BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly kind: "timeout" | "network" | "http",
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit, timeoutMs: number): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(
        "The server took too long to respond. It may still be warming up its models - try again in a moment.",
        0,
        "timeout",
      );
    }
    throw new ApiError("Could not reach the server. Is the backend running?", 0, "network");
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status, "http");
  }
  return res.json() as Promise<T>;
}

export interface ReadyResponse {
  ready: boolean;
  models: Record<string, boolean>;
}

export const api = {
  ready: () => request<ReadyResponse>("/ready", { method: "GET" }, 5_000),

  questions: () =>
    request<{ questions: Question[] }>("/questions", { method: "GET" }, 10_000).then(
      (r) => r.questions,
    ),

  evaluateText: (question: string, answer: string) =>
    request<EvaluationResult>(
      "/evaluate/text",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, answer }),
      },
      // generous: first call may cold-start the LLM judge / emotion model
      60_000,
    ),

  evaluateAudio: (question: string, audio: Blob) => {
    const form = new FormData();
    form.append("file", audio, "answer.wav");
    form.append("question", question);
    return request<EvaluationResult>(
      "/evaluate/audio",
      { method: "POST", body: form },
      120_000, // transcription + scoring
    );
  },
};
