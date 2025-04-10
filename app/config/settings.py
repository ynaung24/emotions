from pathlib import Path
from typing import Dict, List

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Video settings
VIDEO_SETTINGS = {
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "face_detection_confidence": 0.5,
    "emotion_detection_confidence": 0.7,
}

# Audio settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "chunk_size": 1024,
    "channels": 1,
    "max_duration": 10,  # seconds
}

# Text settings
TEXT_SETTINGS = {
    "max_length": 512,
    "batch_size": 32,
    "language": "en",
}

# Emotion labels (Ekman's 6 basic emotions + neutral)
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# Model paths
MODEL_PATHS = {
    "video": MODELS_DIR / "video_emotion_cnn.pt",
    "audio": MODELS_DIR / "audio_emotion_rnn.pt",
    "text": MODELS_DIR / "text_emotion_bert.pt",
    "fusion_weights": MODELS_DIR / "fusion_weights.json",
}

# Ensemble weights (can be adjusted based on model performance)
ENSEMBLE_WEIGHTS = {
    "video": 0.4,
    "audio": 0.3,
    "text": 0.3,
}

# API settings
API_SETTINGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
}

# Logging settings
LOG_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": BASE_DIR / "logs" / "app.log",
}

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR, LOG_SETTINGS["log_file"].parent]:
    directory.mkdir(parents=True, exist_ok=True)
