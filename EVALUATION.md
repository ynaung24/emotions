# Evaluation

This replaces `notebooks/emotion_detection_showcase.md`, whose numbers (78.3%
accuracy, per-class F1 for "Happy/Sad/Angry/Neutral", a "120 MB" model that was
1.6 MB) described a training run that does not exist in this repo and could not be
reproduced. Everything below is produced by `pytest -m slow` and the scripts in
`emotions-app/backend/scripts/`.

## What "good" means here

The scorer is not a trained classifier with a labelled test set - it is a
composite of deterministic linguistic and embedding metrics. The thing worth
measuring is **discrimination**: does it rank a strong answer above a weak one for
the same question, and does it push an off-topic answer down?

## Golden discrimination set

`emotions-app/backend/tests/data/golden.jsonl` - 5 questions across categories
(background, evaluation, behavioural, deployment, ethics), each with four answers:

| tier | what it is |
|---|---|
| `strong` | detailed, specific, correct |
| `mediocre` | on topic, thin, no specifics |
| `weak` | on topic in wording only - platitudes |
| `offtopic` | a good answer, but to a different question |

Scores (0-100), current `main`:

| tier | mean | min | max |
|---|---|---|---|
| strong | 75.3 | 67.1 | 85.1 |
| mediocre | 47.2 | 40.0 | 61.0 |
| weak | 30.3 | 22.7 | 42.7 |
| offtopic | 22.5 | 16.5 | 30.0 |

- **strong − offtopic gap:** mean 52.8, min 37.1 (test guard: ≥ 25)
- **strong > mediocre > weak** for every question (test guard)
- **offtopic < 45 and weak < 55** for every question (test guard)

The previous implementation scored every normal-length answer ≥ 65 regardless of
quality and could not separate a strong answer from an off-topic one; that is the
regression `tests/golden/test_discrimination.py` guards against.

## Per-metric checks

`tests/unit/` verifies each metric in isolation with a deterministic fake encoder:
specificity rewards concrete detail, confidence penalises hedging, conciseness
penalises repetition, completeness's missing-keyword list is populated (the old
code's equivalent branch was mathematically unreachable), STAR structure detects
all four components when present.

## Voice path

`tests/test_api.py` covers the audio endpoint: a silent WAV is flagged
`no_speech_detected`; a real recording returns a transcript plus delivery metrics
(WPM, pause ratio, filler rate, pitch variation) and GoEmotions classification.
Transcription is `faster-whisper` (`base.en`); emotion is
`SamLowe/roberta-base-go_emotions`, the same 28-label taxonomy as the GoEmotions
CSV that ships in `data/`.

## Reproducing

```bash
cd emotions-app/backend
pytest -m slow -q
python -m scripts.diagnose_scores      # per-tier breakdown, all 5 questions
python -m scripts.diagnose_corpus      # a strong + fluff answer for all 20 questions
```
