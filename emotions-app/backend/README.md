# Emotion Detection API

This is the backend service for the Emotion Detection application, built with FastAPI. It provides endpoints for evaluating text and audio responses for interviews, including emotion detection and response analysis.

## Features

- Evaluate text responses for interview questions
- Process and evaluate audio responses
- Detect emotions from voice
- Provide detailed feedback and scores

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file):
   ```
   # API Configuration
   PORT=8000
   DEBUG=True
   ```

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- Interactive API docs (Swagger UI): `http://localhost:8000/docs`
- Alternative API docs (ReDoc): `http://localhost:8000/redoc`

## Endpoints

### GET /
Basic health check endpoint.

### GET /questions
Get a list of available interview questions.

### POST /evaluate/text
Evaluate a text response to an interview question.

**Request Body:**
```json
{
  "text": "Your response here",
  "question": "Interview question"
}
```

### POST /evaluate/audio
Evaluate an audio response to an interview question.

**Form Data:**
- `file`: Audio file (WAV, MP3, etc.)
- `question`: Interview question

## Development

### Code Style
This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

### Pre-commit Hooks
To set up pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## License
MIT
