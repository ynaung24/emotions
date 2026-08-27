// Encode a decoded AudioBuffer to a mono 16-bit PCM WAV.
//
// The old implementation declared an interleaved-stereo header but wrote channels
// contiguously, so any stereo recording came out garbled and half-silent. This
// downmixes to mono up front, so there is only one channel to lay down.

export function encodeWavMono(buffer: AudioBuffer, targetRate = 16_000): Blob {
  return new Blob([wavArrayBuffer(buffer, targetRate)], { type: "audio/wav" });
}

/** The raw WAV bytes - split out so it's testable without a working Blob/Response. */
export function wavArrayBuffer(buffer: AudioBuffer, targetRate = 16_000): ArrayBuffer {
  const mono = downmix(buffer);
  const resampled =
    buffer.sampleRate === targetRate ? mono : resample(mono, buffer.sampleRate, targetRate);

  const bytesPerSample = 2;
  const dataSize = resampled.length * bytesPerSample;
  const arrayBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(arrayBuffer);

  writeStr(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(view, 8, "WAVE");
  writeStr(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, targetRate, true);
  view.setUint32(28, targetRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < resampled.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, resampled[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return arrayBuffer;
}

function downmix(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0);
  const out = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < data.length; i++) out[i] += data[i] / buffer.numberOfChannels;
  }
  return out;
}

function resample(data: Float32Array, from: number, to: number): Float32Array {
  const ratio = from / to;
  const out = new Float32Array(Math.round(data.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const src = i * ratio;
    const lo = Math.floor(src);
    const hi = Math.min(lo + 1, data.length - 1);
    out[i] = data[lo] + (data[hi] - data[lo]) * (src - lo);
  }
  return out;
}

function writeStr(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

export async function blobToWav(blob: Blob): Promise<Blob> {
  const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
  try {
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
    return encodeWavMono(decoded);
  } finally {
    void ctx.close();
  }
}
