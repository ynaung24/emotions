# Interview Response Evaluator

A modern, responsive web application for evaluating interview responses with both text and voice analysis capabilities. Built with React, Chakra UI, and FastAPI.

## Features

- **Modern Web Interface**
  - Clean, responsive design using Chakra UI
  - Intuitive navigation and user experience
  - Real-time feedback and scoring

- **Text Response Evaluation**
  - Semantic analysis using BERT for text understanding
  - Relevance, clarity, and completeness metrics
  - Detailed feedback and improvement suggestions
  - Professional tone assessment

- **Voice Response Analysis**
  - Browser-based audio recording
  - Speech-to-text transcription
  - Emotion detection and analysis
  - Real-time feedback on speaking clarity and tone

- **Evaluation Criteria**
  - Relevance to the question
  - Clarity of expression
  - Completeness of response
  - Professional tone assessment
  - Emotional intelligence analysis

## Prerequisites

- Node.js (v16 or later)
- Python 3.8+
- npm or yarn

## Installation

### Backend Setup

1. Navigate to the backend directory and create a virtual environment:

```bash
cd emotions-app/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory and install dependencies:

```bash
cd ../frontend
npm install
# or
yarn install
```

## Running the Application

### Start the Backend

In the backend directory:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Start the Frontend

In the frontend directory:

```bash
npm start
# or
yarn start
```

The application will open in your default browser at `http://localhost:3000`

## Usage

1. **Home Page**
   - View available evaluation options
   - Choose between text or voice evaluation

2. **Text Evaluation**
   - Select a question from the dropdown
   - Type your response in the text area
   - Click "Submit" to get detailed feedback

3. **Voice Evaluation**
   - Select a question
   - Click the microphone button to start recording
   - Speak your response clearly
   - Click stop when finished
   - Submit for analysis

4. **Results**
   - View your overall score
   - See detailed metrics for different evaluation criteria
   - Review feedback and suggestions for improvement
   - (For voice) View emotion analysis

## Technical Stack

### Frontend
- React 18 with TypeScript
- Chakra UI for styling and components
- React Query for data fetching
- Axios for HTTP requests
- Web Audio API for voice recording

### Backend
- FastAPI for RESTful API
- Python 3.8+
- PyTorch for ML models
- HuggingFace Transformers for NLP
- SpeechRecognition for audio processing

### Key Features
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Feedback**: Immediate scoring and analysis
- **Secure**: JWT-based authentication (coming soon)
- **Scalable**: Microservices-ready architecture

## Project Structure

```
emotions-app/
├── backend/               # FastAPI backend
│   ├── main.py            # Main application entry point
│   ├── evaluate/          # Evaluation logic
│   ├── models/            # ML models
│   └── requirements.txt   # Python dependencies
├── frontend/              # React frontend
│   ├── public/            # Static files
│   ├── src/               # Source code
│   │   ├── components/    # Reusable components
│   │   ├── pages/         # Page components
│   │   ├── api/           # API client
│   │   └── types/         # TypeScript types
│   └── package.json       # Node.js dependencies
└── README.md              # This file
├── README.md            # Project documentation
├── .gitignore           # Git ignore file
└── data/                # Data directory
    └── corpus.json/     # General interview prompts
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Go Emotions dataset for emotion detection
- SentenceTransformer for semantic analysis
- Streamlit for the web interface 