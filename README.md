# Interview Evaluator

A tool for evaluating interview responses using natural language processing and semantic similarity.

## Features

- Question-response evaluation using sentence transformers
- Scoring based on relevance, clarity, and completeness
- Detailed feedback generation
- Interactive web interface using Streamlit

## Project Structure

```
interview_evaluator/
├── app.py              # Main Streamlit application
├── evaluate.py         # Core evaluation logic
├── data/
│   └── corpus.json    # Training and evaluation data
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ynaung24/emotions.git
cd emotions
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run interview_evaluator/app.py
```

## Usage

1. Select a question from the dropdown menu
2. Enter your response in the text area
3. Click "Evaluate" to get your score and feedback
4. Review your scores for:
   - Overall score (out of 100)
   - Relevance (out of 100)
   - Clarity (out of 100)
   - Completeness (out of 100)
5. Read the detailed feedback to improve your responses

## License

MIT License 