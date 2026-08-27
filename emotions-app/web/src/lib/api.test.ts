import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";

const realFetch = global.fetch;
afterEach(() => {
  global.fetch = realFetch;
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("surfaces the backend's detail string on a 4xx", async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Empty audio upload." }), { status: 400 }),
    );
    await expect(api.evaluateText("q", "a")).rejects.toMatchObject({
      message: "Empty audio upload.",
      status: 400,
      kind: "http",
    });
  });

  it("maps an aborted request to a timeout ApiError, not a generic failure", async () => {
    global.fetch = vi.fn().mockImplementation(() => {
      const e = new DOMException("aborted", "AbortError");
      return Promise.reject(e);
    });
    const err = await api.ready().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.kind).toBe("timeout");
  });

  it("maps a network failure to a network ApiError", async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    const err = await api.questions().catch((e) => e);
    expect(err.kind).toBe("network");
  });
});
