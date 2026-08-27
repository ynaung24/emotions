# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An **Interview Response Evaluator**: pick an interview question, answer by typing or
recording, get a 0–100 score with a per-metric breakdown and written feedback. Two parts
under `emotions-app/`:

- `emotions-app/backend/` — FastAPI service, the scoring engine (Python 3.12)
- `emotions-app/web/` — Vite + React + TypeScript + Tailwind SPA

`data/corpus.json` (20 questions with keywords + `evaluation_criteria` weights +
`inappropriate_words`) is the one hand-authored asset. It is gitignored by a blanket
`data/` rule and exists only locally — back it up.

`demo/` (screencasts) is legacy. The old CRA frontend, the Streamlit app, a random-
projection emotion RNN (`models/emotion_model.pt`), and the fabricated-metrics notebooks
have all been removed — see git history and `EVALUATION.md`.

## Commands

Backend (`emotions-app/backend/`):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,judge]"
uvicorn app.main:app --reload          # :8000, docs at /docs
pytest -m "not slow"                    # fast: pure scoring, fake encoder (~3s)
pytest -m slow                          # loads real models: golden discrimination, API, registry
ruff check app tests && mypy app
python -m scripts.diagnose_scores       # per-tier score breakdown (tuning aid)
```
Use `/usr/local/bin/python3` (clean python.org 3.12) for the venv — the default
`python3` is Anaconda with a broken NumPy ABI.

Web (`emotions-app/web/`):
```bash
npm install
npm run dev            # :5173, proxies /api -> :8000
npm run build          # tsc -b && vite build
npm test               # vitest
npm run e2e            # playwright (needs backend on :8000)
npm run typecheck && npm run lint
```

Full stack: `docker compose up --build` (web on :8080, backend on :8000; `data/` and the
model cache are mounted as volumes so models download once).

## Backend architecture

Layered `app/` package. **Models load exactly once**, in the lifespan (`app/main.py`), into
a `ModelRegistry` that also precomputes corpus question + keyword embeddings. Requests do
**one spaCy pass and ≤2 encoder passes** (guarded by a test).

```
app/
  main.py                 create_app() + lifespan
  core/config.py          pydantic-settings, env prefix EVAL_ (+ ANTHROPIC/OPENAI keys)
  api/routes/             health (/health, /ready), questions, evaluate (/evaluate/{text,audio})
  schemas/evaluation.py   EvaluationResult, MetricResult{value, computed, ...}
  services/
    registry.py           ModelRegistry - every heavy model, built once; precompute()
    corpus.py             load + validate corpus.json -> Corpus (frozen dataclasses)
    evaluator.py           the orchestrator (guard -> features -> encode -> score -> aggregate -> feedback)
    scoring/              one file per metric + base.py (piecewise calibration) + adjust.py
    feedback/             templates.py (always) + judge.py (Claude/OpenAI/Null)
    transcription.py      faster-whisper   delivery.py  prosody   emotion.py  GoEmotions
```

### Scoring model (the core of the rebuild)

The old evaluator rescaled every cosine `(sim+1)/2`, flooring relevance/completeness at 50
and pinning clarity at 100 — every answer scored ≥ 65 regardless of quality, and two
keyword-feedback branches were unreachable. The rebuild:

- **`scoring/base.py::piecewise`** — anchored piecewise-linear calibration replaces
  `(sim+1)/2`. Anchor constants (`RELEVANCE_COSINE`, `KEYWORD_SIM`, …) are tuning knobs;
  `tests/golden/` is the guard on them.
- **relevance** = bi-encoder cosine of the answer vs. a *target*. For a matched corpus
  question the target is its `expected_keywords` (not the question wording), so an
  open-ended prompt still scores a specific answer as on-point and a question-parroting
  vague answer doesn't. (An ms-marco cross-encoder was tried and removed — it inverted on
  "on-topic but empty" answers.)
- **completeness** = hybrid: keyword lemma appears in the answer (full credit) OR semantic
  max-sentence similarity (partial). Coverage curve treats keyword lists as aspirational.
- **`scoring/adjust.py`** — cross-metric fixes the orchestrator applies: `damp_relevance`
  (a low-completeness answer loses ~⅓ of its relevance), behavioural relevance/completeness
  floors from the STAR `structure` score.
- **weights** come from `corpus.json::evaluation_criteria`, extended with the new criteria.
  `structure` only applies to behavioural/project/communication questions.
- An unknown question is still scored (`scored_generically: true`), never hard-failed.

### The LLM judge

`feedback/judge.py` — a `Judge` protocol with `ClaudeJudge`, `OpenAIJudge`, `NullJudge`.
Auto-selected from whichever API key is set; **never scores**, only writes prose; with no
key the API works unchanged. Default Claude model is `claude-haiku-4-5` (runs per request).
When touching judge code, load the `claude-api` skill for current model IDs.

## Frontend architecture

One flow: `Session` route (question picker + Text/Voice tabs) → `Results` route. State in
`store/session.ts` is **sessionStorage-backed**, so a refresh or pasted URL on `/results`
doesn't dead-end (the old app kept it only in router state). One `QueryClient`
(`lib/query.ts`). `lib/api.ts` is a typed fetch client with per-endpoint timeouts and typed
errors. `lib/audio.ts` encodes mono 16-bit WAV (the old CRA encoder declared interleaved
stereo but wrote channels contiguously → garbled). Results render a metric only when
`computed` is true. `WarmupBanner` polls `/ready`.

## Gotchas

- First evaluation after startup takes 20–40s (model load). `/ready` reports readiness.
- `faster-whisper` + `soundfile`/`audioread` handle audio decoding — **ffmpeg is not
  required** (the old code shelled out to it via pydub).
- `.env` at the repo root may hold a stale `OPENAI_API_KEY`; the judge tries it and falls
  back to templates on a 401. Set `EVAL_JUDGE_PROVIDER=none` to skip.
- `model_cache/` (gitignored) holds HF-hub-format caches; ~560 MB after pruning, +~500 MB
  when the emotion model first loads.
