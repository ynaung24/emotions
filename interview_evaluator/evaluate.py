import json
import os
from typing import Dict, Any
from sentence_transformers import SentenceTransformer, util

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_corpus() -> Dict[str, Any]:
    """Load the evaluation corpus from JSON file."""
    corpus_path = os.path.join(os.path.dirname(__file__), 'data', 'corpus.json')
    with open(corpus_path, 'r') as f:
        return json.load(f)

def evaluate_response(response: str, question: str) -> Dict[str, Any]:
    """
    Evaluate an interview response against a question.
    
    Args:
        response: The candidate's response
        question: The interview question
        
    Returns:
        Dictionary containing evaluation metrics
    """
    corpus = load_corpus()
    
    # Find matching question in corpus
    matching_question = None
    for q in corpus['questions']:
        if util.pytorch_cos_sim(
            model.encode(question, convert_to_tensor=True),
            model.encode(q['text'], convert_to_tensor=True)
        ).item() > 0.8:
            matching_question = q
            break
    
    if not matching_question:
        return {
            'score': 0,
            'feedback': 'Question not found in evaluation corpus',
            'metrics': {
                'relevance': 0,
                'clarity': 0,
                'completeness': 0
            }
        }
    
    # Calculate keyword coverage
    keywords = matching_question['expected_keywords']
    response_embedding = model.encode(response.lower(), convert_to_tensor=True)
    keyword_scores = []
    
    for keyword in keywords:
        keyword_embedding = model.encode(keyword, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(response_embedding, keyword_embedding).item()
        keyword_scores.append(similarity)
    
    # Calculate metrics (scaled to 100)
    completeness = int(sum(score > 0.5 for score in keyword_scores) / len(keywords) * 100)
    relevance = int(max(0.0, min(1.0, sum(keyword_scores) / len(keywords))) * 100)
    
    # Simple clarity score based on response length and average sentence length
    sentences = response.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    clarity = int(max(0.0, min(1.0, 2.0 - abs(avg_sentence_length - 15) / 15)) * 100)
    
    # Calculate overall score
    criteria = corpus['evaluation_criteria']
    overall_score = int(
        relevance * criteria['relevance']['weight'] +
        clarity * criteria['clarity']['weight'] +
        completeness * criteria['completeness']['weight']
    )
    
    return {
        'score': overall_score,
        'feedback': generate_feedback(relevance/100, clarity/100, completeness/100),
        'metrics': {
            'relevance': relevance,
            'clarity': clarity,
            'completeness': completeness
        }
    }

def generate_feedback(relevance: float, clarity: float, completeness: float) -> str:
    """Generate feedback based on metrics."""
    feedback = []
    
    if relevance < 0.5:
        feedback.append("Try to focus more on addressing the specific question asked.")
    elif relevance < 0.8:
        feedback.append("Good attempt at addressing the question, but could be more focused.")
    else:
        feedback.append("Excellent job addressing the question directly.")
        
    if clarity < 0.5:
        feedback.append("Try to make your response more clear and concise.")
    elif clarity < 0.8:
        feedback.append("Your response is fairly clear, but could be more well-structured.")
    else:
        feedback.append("Your response is very clear and well-structured.")
        
    if completeness < 0.5:
        feedback.append("Include more key points in your response.")
    elif completeness < 0.8:
        feedback.append("Good coverage of key points, but some important elements are missing.")
    else:
        feedback.append("Excellent coverage of all key points.")
        
    return " ".join(feedback)