# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An "Interview Response Evaluator" web app. A user picks an interview question, submits a text or voice answer, and gets a 0–100 score plus per-metric breakdown and written feedback. Two parts, both under `emotions-app/`:

- `emotions-app/backend/` — FastAPI service (the real logic lives here)
- `emotions-app/frontend/` — Create React App + TypeScript + Chakra UI SPA

Everything at the repo root outside `emotions-app/` (`notebooks/`, `demo/`, `models/emotion_model.pt`, `data/go_emotions/`, root `requirements.txt`) is legacy from an earlier voice-emotion RNN / Streamlit experiment and is **not used by the running app**. Most of it is gitignored. The README's "Acknowledgments" section still mentions Streamlit — that frontend was removed in `fa4f1c6d`.

## Commands

Backend (run from `emotions-app/backend/`):
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md        # required; not in requirements.txt
uvicorn main:app --reload                       # serves http://localhost:8000, docs at /docs
```
First startup downloads the SentenceTransformer model into `emotions-app/backend/model_cache/` and NLTK `punkt`/`stopwords` — needs internet and takes a minute. Model loads eagerly at import time (see `main.py`), so import errors surface on server start.

Frontend (run from `emotions-app/frontend/`):
```bash
npm install
npm start        # http://localhost:3000
npm run build
npm test         # react-scripts/Jest — no test files exist yet
```

There is no backend test suite (`tests/` is gitignored), no linter wired up despite `backend/README.md` mentioning black/isort/flake8, and no Docker setup despite path references to `/app/data/`.

## How evaluation works

`emotions-app/backend/evaluate/evaluate.py` is the core. `evaluate_response(response, question, role_context)` is called by both `/evaluate/text` and `/evaluate/audio`.

Flow:
1. `load_corpus()` reads `data/corpus.json` (repo root). It tries several hardcoded paths in order, including a machine-specific absolute path — keep the relative `../../data/corpus.json` resolution working.
2. The incoming `question` is matched against corpus questions by SentenceTransformer cosine similarity with a **0.7 threshold**. No match → the response scores 0 with "Question not found in evaluation corpus". This is why the frontend must send question text straight from `GET /questions` rather than free text.
3. Metrics, each 0–100:
   - `relevance` — cosine similarity of response vs. question
   - `clarity` — heuristic on average sentence length (10–20 words = 100)
   - `completeness` — mean similarity of the response against the matched question's `expected_keywords`
   - `confidence`, `conciseness` — **not computed**; hardcoded in `main.py` (0 / 75, and 0.7 for audio)
4. Overall score = weighted sum using weights from `corpus.json` → `evaluation_criteria` (default 0.4 / 0.3 / 0.3).
5. `generate_feedback()` turns the three real metrics + keyword hits into a single feedback string.

`corpus.json` shape: `questions[]` (`id`, `text`, `category`, `expected_keywords[]`), `evaluation_criteria` (per-criterion `weight`), `inappropriate_words[]` (exact-word match → instant 0).

`analyze_response_structure()` and `get_embedding()` (with its `embedding_cache`) exist but are currently unused by the request path. `openai` is imported and keyed from `OPENAI_API_KEY` in `.env` but not actually called.

## Audio path specifics (`/evaluate/audio`)

Multipart upload (`file` + `question` form field). The handler shells out through `pydub`/`AudioSegment`, so **ffmpeg must be installed** to transcode to 16 kHz mono WAV. Transcription uses `SpeechRecognition` → `recognize_google`, which requires internet and has no API key. Returned `emotions` scores are a hardcoded placeholder dict — there is no real voice emotion model in the request path.

## Frontend notes

- API base URL is hardcoded to `http://localhost:8000` in `src/api/client.ts` (10s axios timeout). No `proxy` in `package.json`.
- Backend CORS is `allow_origins=["*"]`.
- Routing: `react-router-dom` v6 in `App.tsx` (`/`, `/evaluate/text`, `/evaluate/voice`, `/results`). Data fetching: `react-query` **v3** (`react-query` package, not `@tanstack/react-query`).
- Voice recording uses `react-audio-voice-recorder`.
- `GET /questions` returns `{ questions: string[] }` (plain strings, not objects) — the frontend `Question` interface doesn't match the wire format.

## Gotchas

- `emotions-app/backend/evaluate.py` is a **broken symlink** to a deleted `../../../evaluate.py`. Ignore it; the live code is the `evaluate/` package. Don't `import evaluate` expecting the file.
- `main.py` and `evaluate/evaluate.py` both call `initialize_models()` / `load_corpus()` at module import — the models get loaded more than once. Slow but harmless.
- `.env` at the repo root contains a real `OPENAI_API_KEY` and is gitignored — don't commit it or echo it.
- `data/` is gitignored except `data/corpus.json`, which is force-added. New corpus edits must be `git add -f` or the ignore rule adjusted.
