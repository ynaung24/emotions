import { useSyncExternalStore } from "react";
import type { EvaluationResult, Mode, SessionState } from "@/lib/types";

// sessionStorage-backed so a refresh on /results doesn't dead-end at
// "no evaluation data" the way the old router-state-only approach did.

const KEY = "ire.session.v1";

const empty: SessionState = {
  question: "",
  questionId: null,
  mode: "text",
  answer: "",
  result: null,
};

function read(): SessionState {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? { ...empty, ...JSON.parse(raw) } : empty;
  } catch {
    return empty;
  }
}

let state = read();
const listeners = new Set<() => void>();

function write(next: SessionState) {
  state = next;
  try {
    sessionStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* private mode / quota - in-memory still works */
  }
  listeners.forEach((l) => l());
}

export const session = {
  subscribe(fn: () => void) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
  get: () => state,
  setQuestion(question: string, questionId: string | null) {
    write({ ...state, question, questionId, result: null });
  },
  setMode(mode: Mode) {
    write({ ...state, mode });
  },
  setAnswer(answer: string) {
    write({ ...state, answer });
  },
  setResult(result: EvaluationResult, answer: string) {
    write({ ...state, result, answer });
  },
  reset() {
    write(empty);
  },
};

export function useSession(): SessionState {
  return useSyncExternalStore(session.subscribe, session.get, session.get);
}
