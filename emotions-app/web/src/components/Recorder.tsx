import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, RotateCcw, Square } from "lucide-react";
import { Button } from "./ui/primitives";

interface Props {
  disabled?: boolean;
  onRecorded: (blob: Blob) => void;
}

type Phase = "idle" | "recording" | "recorded";

export function Recorder({ disabled, onRecorded }: Props) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const cleanup = useCallback(() => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    void audioCtxRef.current?.close();
    timerRef.current = rafRef.current = null;
    streamRef.current = audioCtxRef.current = analyserRef.current = null;
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const drawWave = useCallback(() => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext("2d")!;
    const data = new Uint8Array(analyser.frequencyBinCount);
    const render = () => {
      rafRef.current = requestAnimationFrame(render);
      analyser.getByteTimeDomainData(data);
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#0080ff";
      ctx.beginPath();
      const slice = width / data.length;
      for (let i = 0; i < data.length; i++) {
        const y = (data[i] / 128) * (height / 2);
        if (i === 0) ctx.moveTo(0, y);
        else ctx.lineTo(i * slice, y);
      }
      ctx.stroke();
    };
    render();
  }, []);

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      ctx.createMediaStreamSource(stream).connect(analyser);
      analyserRef.current = analyser;
      drawWave();

      const rec = new MediaRecorder(stream);
      mediaRef.current = rec;
      rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      rec.onstop = () => {
        // browser-native container (webm/opus or mp4); server-side decoding
        // + our own WAV re-encode handle the conversion
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        onRecorded(blob);
        setPhase("recorded");
        cleanup();
      };
      rec.start();

      setSeconds(0);
      timerRef.current = window.setInterval(() => setSeconds((s) => s + 1), 1000);
      setPhase("recording");
    } catch {
      setError("Microphone access was denied. Check your browser's site permissions.");
    }
  }

  function stop() {
    if (mediaRef.current?.state === "recording") mediaRef.current.stop();
  }

  function redo() {
    setPhase("idle");
    setSeconds(0);
  }

  const mmss = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 rounded-lg surface-2 p-4">
        <canvas
          ref={canvasRef}
          width={320}
          height={56}
          className="h-14 flex-1 rounded bg-[var(--surface)]"
        />
        <span className="font-mono text-lg tabular-nums">{mmss}</span>
      </div>

      {error && <p className="text-sm text-rose-600 dark:text-rose-400">{error}</p>}

      <div className="flex gap-2">
        {phase === "idle" && (
          <Button onClick={start} disabled={disabled}>
            <Mic className="size-4" /> Start recording
          </Button>
        )}
        {phase === "recording" && (
          <Button onClick={stop} variant="outline">
            <Square className="size-4 fill-current" /> Stop
          </Button>
        )}
        {phase === "recorded" && (
          <Button onClick={redo} variant="ghost">
            <RotateCcw className="size-4" /> Re-record
          </Button>
        )}
      </div>
    </div>
  );
}
