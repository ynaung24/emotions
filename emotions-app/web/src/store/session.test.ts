import { beforeEach, describe, expect, it } from "vitest";
import { session } from "./session";
import type { EvaluationResult } from "@/lib/types";

const fakeResult = {
  score: 72,
  metrics: {},
  feedback: { summary: "ok", strengths: [], improvements: [], source: "template" },
  matched_question: null,
  scored_generically: false,
  emotions: [],
  transcript: null,
  delivery: null,
  flags: [],
} as EvaluationResult;

beforeEach(() => {
  sessionStorage.clear();
  session.reset();
});

describe("session store", () => {
  it("persists across a simulated reload", () => {
    session.setQuestion("Tell me about yourself.", "q1");
    session.setResult(fakeResult, "my answer");

    // a fresh module read would re-hydrate from sessionStorage
    const raw = JSON.parse(sessionStorage.getItem("ire.session.v1")!);
    expect(raw.question).toBe("Tell me about yourself.");
    expect(raw.result.score).toBe(72);
    expect(raw.answer).toBe("my answer");
  });

  it("clears the stale result when the question changes", () => {
    session.setResult(fakeResult, "a");
    session.setQuestion("new question", null);
    expect(session.get().result).toBeNull();
  });
});
