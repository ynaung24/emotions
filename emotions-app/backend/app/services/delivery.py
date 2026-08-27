"""Spoken-delivery metrics from Whisper word timestamps + librosa pitch.

This is what makes the voice path genuinely different from the text path rather
than strictly worse - it measures *how* something was said.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

from app.schemas.evaluation import DeliveryMetrics
from app.services.scoring.text_features import FILLER_PATTERNS
from app.services.transcription import Transcription

_FILLER_RE = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)
_PAUSE_THRESHOLD = 0.6  # seconds of silence between words that counts as a pause


def analyse_delivery(tr: Transcription) -> tuple[DeliveryMetrics, float]:
    """Return the metrics plus a 0-1 'steadiness' signal for the confidence scorer."""
    words = tr.words
    n_words = len(words)
    duration = max(tr.duration, 1e-6)

    wpm = n_words / duration * 60.0 if n_words else 0.0

    gaps = [
        b.start - a.end for a, b in itertools.pairwise(words) if b.start > a.end
    ]
    pause_time = sum(g for g in gaps if g >= _PAUSE_THRESHOLD)
    pause_ratio = min(1.0, pause_time / duration)

    fillers = len(_FILLER_RE.findall(tr.text))
    filler_rate = fillers / max(n_words, 1) * 100.0

    pitch_var = _pitch_variation(tr.audio, tr.sample_rate)

    # Steadiness: penalise very fast/slow speech, heavy pausing, and monotone or
    # wildly erratic pitch.
    pace_ok = _bell(wpm, ideal=150, spread=55)
    pause_ok = 1.0 - min(1.0, pause_ratio / 0.35)
    pitch_ok = _bell(pitch_var, ideal=0.18, spread=0.16)
    steadiness = float(np.clip(0.45 * pace_ok + 0.3 * pause_ok + 0.25 * pitch_ok, 0.0, 1.0))

    metrics = DeliveryMetrics(
        words_per_minute=round(wpm, 1),
        pause_ratio=round(pause_ratio, 3),
        filler_rate_per_100=round(filler_rate, 2),
        pitch_variation=round(pitch_var, 3),
        duration_seconds=round(duration, 1),
    )
    return metrics, steadiness


def _pitch_variation(audio: np.ndarray, sr: int) -> float:
    if audio.size < sr // 2:
        return 0.0
    try:
        import librosa

        f0, _voiced, _ = librosa.pyin(
            audio, sr=sr, fmin=65, fmax=400, frame_length=2048
        )
    except Exception:
        return 0.0
    voiced_f0 = f0[np.isfinite(f0)]
    if voiced_f0.size < 10:
        return 0.0
    # coefficient of variation of pitch, in semitone-ish terms
    return float(np.std(np.log2(voiced_f0)) * 2.0)


def _bell(x: float, *, ideal: float, spread: float) -> float:
    return float(np.exp(-((x - ideal) ** 2) / (2 * spread**2)))
