# Interview Emotion Analyzer

A multimodal emotion analysis system that combines facial expressions, vocal tone, and speech content to provide real-time emotion analysis during interviews.

## Features

- Real-time facial expression analysis using CNN
- Voice tone emotion detection using RNN/CNN
- Speech content analysis using BERT
- Ensemble fusion of multiple modalities
- Live web interface using Streamlit
- REST API using FastAPI

## Project Structure

```
interview-emotion-analyzer/
├── app/                # Core application modules
├── frontend/          # Streamlit UI
├── notebooks/         # Model experiments
├── data/             # Datasets
├── models/           # Trained models
└── tests/            # Unit tests
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ynaung24/emotions.git
cd emotions
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
# Start the FastAPI backend
uvicorn app.api.main:app --reload

# In a separate terminal, start the Streamlit frontend
streamlit run frontend/streamlit_app.py
```

## Docker Setup

Alternatively, you can run the application using Docker:

```bash
docker-compose up
```

## License

MIT License 