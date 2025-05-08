# Interview Response Evaluator

A comprehensive interview response evaluation system that analyzes both text and voice responses using advanced NLP and emotion detection techniques.

## Features

- **Text Response Evaluation**
  - Semantic analysis using BERT for text understanding
  - Keyword matching and scoring
  - Relevance, clarity, and completeness metrics
  - Detailed feedback and improvement suggestions
  - Professional tone assessment

- **Voice Response Analysis**
  - Multi-modal emotion detection using RNN
  - BERT-based text feature extraction
  - Mel spectrogram for audio feature extraction
  - Real-time emotion visualization and scoring
  - Support for 29 distinct emotions including:
    - Basic emotions (joy, sadness, anger, fear, surprise)
    - Complex emotions (pride, gratitude, admiration, contentment)
    - Professional emotions (confidence, enthusiasm, optimism)

- **Evaluation Criteria**
  - Relevance to the question
  - Clarity of expression
  - Completeness of response
  - Professional tone assessment
  - Emotional intelligence analysis with weighted scoring
  - Positive emotion boost system (up to 30% score enhancement)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/interview-evaluator.git
cd interview-evaluator
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Using the Evaluator:
   - Select an interview question from the dropdown
   - Choose between text or voice response
   - For text: Type your response in the text area
   - For voice: Record or upload your response
   - Click "Evaluate" to get feedback

3. Understanding the Results:
   - Overall score (0-100)
   - Individual metrics for relevance, clarity, and completeness
   - Detailed feedback for improvement
   - Emotion analysis (for voice responses)

## Technical Details

### Text Evaluation
- BERT (bert-base-uncased) for semantic analysis
- Custom scoring system for interview responses
- Multi-criteria evaluation framework
- Professional tone assessment

### Voice Processing
- RNN-based emotion detection model
- BERT for text feature extraction
- Mel spectrogram for audio features
- Multi-modal fusion for comprehensive analysis
- Real-time emotion detection and scoring

### Dataset
- Go Emotions dataset for emotion detection training
- Emotion Stimulus dataset for additional training data
- Custom corpus for interview questions
- Synthetic audio generation for training

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── evaluate.py           # Response evaluation logic
├── voice_processor.py    # Voice processing and emotion detection
├── prepare_dataset.py    # Dataset preparation script
├── requirements.txt      # Project dependencies
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