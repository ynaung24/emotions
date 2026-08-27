# Interview Response Evaluator - Web

Vite + React + TypeScript + Tailwind. Talks to the FastAPI backend.

## Dev

```bash
npm install
npm run dev          # http://localhost:5173
```

`vite.config.ts` proxies `/api/*` to `http://localhost:8000`, so run the backend
(`uvicorn app.main:app` in `../backend`) alongside. For a deployed build, set
`VITE_API_BASE` to the backend origin at build time.

## Scripts

```bash
npm run build        # tsc -b && vite build
npm run typecheck
npm run lint
npm test             # vitest (audio encoder, api client, session store)
npm run e2e          # playwright - needs the backend running on :8000
```

## Structure

```
src/
  lib/          api.ts (fetch client, typed errors, per-call timeouts)
                types.ts (mirrors the backend schema)
                query.ts (one QueryClient)
                audio.ts (mono 16-bit WAV encoder)
  store/        session.ts (sessionStorage-backed - survives a refresh on /results)
  components/   QuestionPicker, AnswerPanel, Recorder, WarmupBanner, Layout
                results/  ScoreRing, MetricRadar, MetricGrid, FeedbackList, VoicePanels
                ui/       primitives (Button, Card, Textarea, Meter, ...)
  routes/       Session, Results
```

## Notes

- One flow: pick a question, answer via the Text / Voice tabs on the same screen,
  results render inline. Session state is persisted so a page reload on `/results`
  keeps the result.
- The Results page renders a metric only when the backend marks it `computed` -
  no placeholder numbers.
- `Recorder` draws a live `AnalyserNode` waveform; `audio.ts` downmixes to mono
  before encoding (the old CRA encoder garbled stereo).
- `WarmupBanner` polls `/ready`; the first evaluation can take 20-40s while the
  models load.
