from sentence_transformers import SentenceTransformer, util
import json
import os
import re
from typing import Dict, Any, List
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the model
model = SentenceTransformer('all-mpnet-base-v2')

def load_corpus() -> Dict[str, Any]:
    """Load the evaluation corpus from JSON file."""
    corpus_path = os.path.join(os.path.dirname(__file__), 'data', 'corpus.json')
    with open(corpus_path, 'r') as f:
        return json.load(f)

def contains_inappropriate_content(text: str, inappropriate_words: List[str]) -> bool:
    """
    Check if text contains inappropriate content.
    Returns True only if an exact match of an inappropriate word is found.
    """
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words and check each word
    words = set(text.split())  # Use set for O(1) lookup
    inappropriate_set = set(inappropriate_words)  # Convert list to set for O(1) lookup
    
    # Check for exact matches only
    return any(word in inappropriate_set for word in words)

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
    
    # Check for inappropriate content first
    if contains_inappropriate_content(response, corpus['inappropriate_words']):
        return {
            'score': 0,
            'feedback': 'Your response contains inappropriate language. Please maintain a professional tone in your interview responses.',
            'metrics': {
                'relevance': 0,
                'clarity': 0,
                'completeness': 0
            }
        }
    
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
    
    # Calculate keyword coverage with more granular scoring
    keywords = matching_question['expected_keywords']
    response_embedding = model.encode(response.lower(), convert_to_tensor=True)
    keyword_scores = []
    keyword_feedback = []
    
    for keyword in keywords:
        keyword_embedding = model.encode(keyword, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(response_embedding, keyword_embedding).item()
        # Scale similarity to be more sensitive
        scaled_similarity = (similarity + 1) / 2  # Convert from [-1,1] to [0,1]
        keyword_scores.append(scaled_similarity)
        keyword_feedback.append((keyword, scaled_similarity))
    
    # Calculate metrics with more granular scoring
    # Completeness: Weighted average of keyword coverage
    completeness = int(sum(keyword_scores) / len(keywords) * 100)
    
    # Relevance: Compare response to question directly
    question_embedding = model.encode(question, convert_to_tensor=True)
    relevance_score = util.pytorch_cos_sim(response_embedding, question_embedding).item()
    relevance = int((relevance_score + 1) / 2 * 100)  # Convert from [-1,1] to [0,100]
    
    # Clarity: More sophisticated sentence analysis
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    if not sentences:
        clarity = 0
    else:
        # Calculate average sentence length
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        # Ideal sentence length is between 10-20 words
        if avg_length < 10:
            clarity = int((avg_length / 10) * 100)
        elif avg_length > 20:
            clarity = int((20 / avg_length) * 100)
        else:
            clarity = 100
    
    # Calculate overall score with adjusted weights
    criteria = corpus['evaluation_criteria']
    overall_score = int(
        relevance * criteria['relevance']['weight'] +
        clarity * criteria['clarity']['weight'] +
        completeness * criteria['completeness']['weight']
    )
    
    return {
        'score': overall_score,
        'feedback': generate_feedback(
            relevance/100, 
            clarity/100, 
            completeness/100,
            keyword_feedback,
            sentences,
            response
        ),
        'metrics': {
            'relevance': relevance,
            'clarity': clarity,
            'completeness': completeness
        }
    }

def generate_feedback(
    relevance: float, 
    clarity: float, 
    completeness: float,
    keyword_feedback: List[tuple],
    sentences: List[str],
    response: str
) -> str:
    """Generate detailed and actionable feedback based on metrics."""
    feedback = []
    
    # Relevance feedback
    if relevance < 0.5:
        feedback.append("Your response doesn't directly address the question. Try to focus on what was specifically asked.")
    elif relevance < 0.8:
        feedback.append("You're on the right track, but try to be more specific to the question asked.")
    else:
        feedback.append("Excellent job staying focused on the question!")
    
    # Clarity feedback
    if clarity < 0.5:
        feedback.append("Your response is difficult to follow. Try breaking it into shorter, clearer sentences.")
    elif clarity < 0.8:
        feedback.append("Your response is understandable, but could be more concise and well-structured.")
    else:
        feedback.append("Your response is very clear and well-structured!")
    
    # Keyword-specific feedback
    missing_keywords = [kw for kw, score in keyword_feedback if score < 0.5]
    strong_keywords = [kw for kw, score in keyword_feedback if score > 0.8]
    
    if missing_keywords:
        feedback.append(f"Consider including more about: {', '.join(missing_keywords)}")
    if strong_keywords:
        feedback.append(f"Good coverage of: {', '.join(strong_keywords)}")
    
    # Structure feedback
    if len(sentences) < 3:
        feedback.append("Try to provide more detailed examples or explanations.")
    elif len(sentences) > 8:
        feedback.append("Your response might be too long. Consider being more concise.")
    
    # Specific improvement suggestions
    if len(response.split()) < 50:
        feedback.append("Try to provide more detail and examples to support your points.")
    elif len(response.split()) > 200:
        feedback.append("Consider being more concise while maintaining key points.")
    
    # Format the feedback
    formatted_feedback = " ".join(feedback)
    return formatted_feedback