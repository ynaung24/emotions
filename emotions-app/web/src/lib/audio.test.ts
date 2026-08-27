import { describe, expect, it } from "vitest";
import { wavArrayBuffer } from "./audio";

function fakeBuffer(channels: number[][], sampleRate: number): AudioBuffer {
  return {
    numberOfChannels: channels.length,
    length: channels[0].length,
    sampleRate,
    getChannelData: (c: number) => Float32Array.from(channels[c]),
  } as unknown as AudioBuffer;
}

describe("wavArrayBuffer", () => {
  it("writes a valid mono 16-bit PCM header", () => {
    const view = new DataView(wavArrayBuffer(fakeBuffer([[0, 0.5, -0.5, 1]], 16_000)));
    expect(String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3))).toBe("RIFF");
    expect(view.getUint16(20, true)).toBe(1); // PCM
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint16(34, true)).toBe(16); // bits per sample
  });

  it("downmixes stereo so it isn't garbled (the old encoder's bug)", () => {
    const stereo = fakeBuffer(
      [
        [1, 1, 1, 1],
        [-1, -1, -1, -1],
      ],
      16_000,
    );
    const buf = wavArrayBuffer(stereo);
    const view = new DataView(buf);
    expect(view.getUint16(22, true)).toBe(1);
    for (let o = 44; o < buf.byteLength; o += 2) {
      expect(Math.abs(view.getInt16(o, true))).toBeLessThan(10); // L+R average ~ 0
    }
  });

  it("resamples to 16 kHz", () => {
    const view = new DataView(
      wavArrayBuffer(fakeBuffer([[0, 0, 0, 0, 0, 0, 0, 0]], 48_000)),
    );
    expect(view.getUint32(24, true)).toBe(16_000);
  });
});
