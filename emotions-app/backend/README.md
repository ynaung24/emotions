# Interview Response Evaluator - Backend

FastAPI service that scores interview answers (text or voice) on a 0-100 scale
with a per-metric breakdown and written feedback.

## What it measures

Every answer is scored on deterministic, offline metrics - each returns a value
**and** a `computed` flag, so the UI never shows a number that wasn't actually
measured:

| Metric | Signal |
|---|---|
| relevance | bi-encoder cosine of the answer against the question, damped by how much substance is present |
| completeness | hybrid lexical + semantic coverage of the points a strong answer would hit |
| specificity | named tools / proper nouns / numerals density + lexical variety |
| clarity | Flesch reading ease + sentence-length distribution |
| confidence | inverse hedge ("I think", "maybe") and filler density; blends prosody for voice |
| conciseness | content-word ratio, lexical variety, repetition, length band |
| structure | STAR arc (situation/task/action/result) - behavioural questions only |

An **optional LLM judge** (Claude or OpenAI, auto-selected from whichever API key
is set) adds narrative feedback. It never scores; with no key it's skipped.

Voice answers additionally get a Whisper transcript, delivery metrics (WPM, pause
ratio, filler rate, pitch variation), and GoEmotions emotion classification.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,judge]"          # drop [judge] to skip the anthropic/openai SDKs
python -m spacy download en_core_web_md # (installed automatically via the wheel dep)
cp .env.example .env                    # optional
```

## Run

```bash
uvicorn app.main:app --reload           # http://localhost:8000, docs at /docs
```

Models load once during startup (`warmup`). `GET /ready` reports when they're warm.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | models warm? |
| GET | `/questions` | `[{id, text, category}]` |
| POST | `/evaluate/text` | `{question, answer, role?}` -> `EvaluationResult` |
| POST | `/evaluate/audio` | multipart `file` + `question` -> `EvaluationResult` (with transcript + delivery) |

A question that isn't in `corpus.json` is still scored - the response carries
`scored_generically: true` and `flags: ["scored_generically"]`.

## Tests

```bash
pytest -m "not slow"     # fast: pure scoring logic with a fake encoder (~3s)
pytest -m slow           # loads real models: golden discrimination, API, registry
pytest                   # everything
ruff check app tests && mypy app
```

`tests/golden/` is the regression guard for the core defect: it asserts a strong
answer clears an off-topic one by >=25 points and that the tiers stay ordered.
The previous implementation could not do this.

## Layout

```
app/
  main.py              create_app() + lifespan (models load once here)
  core/                config (pydantic-settings), logging
  api/routes/          health, questions, evaluate
  schemas/             pydantic request/response
  services/
    registry.py        owns every heavy model; builds each once; precomputes corpus embeddings
    corpus.py          load + validate corpus.json
    evaluator.py       orchestrator: one spaCy pass, one encoder pass, then scoring
    scoring/           one file per metric + calibration + cross-metric adjustments
    feedback/          templates.py (always) + judge.py (Claude/OpenAI/Null)
    transcription.py   faster-whisper
    delivery.py        prosody metrics from word timestamps
    emotion.py         GoEmotions
```
