"""Speech-to-text via faster-whisper. Offline, word-level timestamps.

Replaces the old `recognize_google` path (undocumented endpoint, rate limited,
needs network) and its pydub->ffmpeg transcode dance.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from app.core.logging import get_logger

logger = get_logger(__name__)

TARGET_SR = 16_000


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcription:
    text: str
    words: list[Word]
    duration: float
    audio: np.ndarray  # mono float32 @ 16 kHz, kept for prosody analysis
    sample_rate: int = TARGET_SR


def load_audio(raw: bytes) -> tuple[np.ndarray, float]:
    """Decode arbitrary container bytes to mono float32 @ 16 kHz.

    soundfile handles WAV/FLAC/OGG natively; librosa (audioread) covers the rest
    without shelling out to ffmpeg.
    """
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
    except (sf.LibsndfileError, RuntimeError):
        import librosa

        mono, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)

    if sr != TARGET_SR:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SR)
    return mono.astype(np.float32), len(mono) / TARGET_SR


def transcribe(raw: bytes, model: object) -> Transcription:
    audio, duration = load_audio(raw)
    segments, _info = model.transcribe(  # type: ignore[attr-defined]
        audio, language="en", word_timestamps=True, vad_filter=True
    )

    words: list[Word] = []
    chunks: list[str] = []
    for seg in segments:
        chunks.append(seg.text)
        for w in seg.words or []:
            words.append(Word(text=w.word.strip(), start=w.start, end=w.end))

    text = " ".join(c.strip() for c in chunks).strip()
    logger.info("transcribed %.1fs of audio into %d words", duration, len(words))
    return Transcription(text=text, words=words, duration=duration, audio=audio)
