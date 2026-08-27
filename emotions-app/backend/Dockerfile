FROM python:3.12-slim AS base

# libsndfile for soundfile; ffmpeg is NOT required (soundfile + audioread cover decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/model_cache \
    EVAL_MODEL_CACHE_DIR=/app/model_cache

WORKDIR /app/emotions-app/backend

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e ".[judge]"

COPY app ./app

# The corpus ships in app/data/; the model cache is a volume (see docker-compose.yml)
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
