from sentence_transformers import SentenceTransformer, util
import json
import os
import re
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import openai
from dotenv import load_dotenv
from collections import defaultdict
import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Model configuration
MODEL_CONFIG = {
    'default': 'all-mpnet-base-v2',  # Best balance of speed and accuracy
    'large': 'all-MiniLM-L12-v2',    # Faster alternative with good performance
    'multilingual': 'paraphrase-multilingual-mpnet-base-v2',  # For non-English text
    'latest': 'sentence-t5-xxl'  # Best performance but requires more resources
}

# Initialize models with optimizations
def initialize_models(model_name: str = 'default'):
    """Initialize the language models with the specified configuration.
    
    Args:
        model_name: Key from MODEL_CONFIG to select the model
    """
    model_name = MODEL_CONFIG.get(model_name, model_name)
    
    # Configure device (use GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize sentence transformer model with optimizations
    model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder='./model_cache'  # Cache models in a local directory
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize spaCy for NLP processing
    try:
        nlp = spacy.load(
            "en_core_web_md",  # Medium model for better NER and POS tagging
            disable=["parser", "textcat"]  # Disable unused components for speed
        )
    except OSError:
        # If the model is not downloaded, download it
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
        nlp = spacy.load("en_core_web_md", disable=["parser", "textcat"])
    
    return model, nlp

# Initialize with default model
model, nlp = initialize_models('default')

# Enable mixed precision for faster inference on compatible GPUs
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
    autocast_enabled = True
else:
    autocast_enabled = False

# Cache for storing embeddings to improve performance
embedding_cache = {}

def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding from cache or compute it if not available.
    Uses mixed precision when GPU is available for better performance.
    """
    if not text or not isinstance(text, str) or not text.strip():
        # Return zero vector with correct dimensions for the current model
        return np.zeros(model.get_sentence_embedding_dimension())
    
    if text not in embedding_cache:
        try:
            # Use mixed precision if available
            if autocast_enabled:
                with autocast():
                    embedding_cache[text] = model.encode(
                        text,
                        convert_to_tensor=False,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
            else:
                embedding_cache[text] = model.encode(
                    text,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(model.get_sentence_embedding_dimension())
    
    return embedding_cache[text]

def load_corpus() -> Dict[str, Any]:
    """Load the evaluation corpus from JSON file."""
    corpus_path = os.path.join('data', 'corpus.json')
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

def evaluate_response(response: str, question: str, role_context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Evaluate an interview response against a question with enhanced metrics.
    
    Args:
        response: The candidate's response
        question: The interview question
        role_context: Optional context about the role being interviewed for
        
    Returns:
        Dictionary containing comprehensive evaluation metrics and feedback
    """
    corpus = load_corpus()
    
    # Check for inappropriate content first
    if contains_inappropriate_content(response, corpus.get('inappropriate_words', [])):
        return {
            'inappropriate': True,
            'score': 0,
            'metrics': {
                'relevance': 0,
                'clarity': 0,
                'completeness': 0,
                'specificity': 0,
                'professionalism': 0,
                'conciseness': 0
            },
            'feedback': 'Response contains inappropriate content',
            'suggestions': ['Please maintain a professional tone in your response']
        }
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

def generate_feedback(relevance: float, clarity: float, completeness: float, 
                    keyword_feedback: List[tuple], sentences: List, response: str) -> str:
    """
    Generate feedback based on evaluation metrics.
    
    Args:
        relevance: Relevance score (0-1)
        clarity: Clarity score (0-1)
        completeness: Completeness score (0-1)
        keyword_feedback: List of (keyword, score) tuples
        sentences: List of sentences in the response
        response: The original response text
        
    Returns:
        str: Formatted feedback string
    """
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
    
    # Completeness feedback
    if completeness < 0.5:
        feedback.append("Your response seems incomplete. Try to provide more details and examples.")
    elif completeness < 0.8:
        feedback.append("Your response covers the main points but could benefit from more depth or examples.")
    else:
        feedback.append("Great job providing a thorough and complete response!")
    
    # Keyword-specific feedback
    if keyword_feedback:
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
    word_count = len(response.split())
    if word_count < 50:
        feedback.append("Try to provide more detail and examples to support your points.")
    elif word_count > 200:
        feedback.append("Consider being more concise while maintaining key points.")
    
    return " ".join(feedback)

def analyze_response_structure(response: str, relevance: float, clarity: float, keyword_feedback: List[tuple] = None) -> Dict[str, float]:
    """
    Analyze the structure and readability of the response.
    
    Args:
        response: The text response to analyze
        relevance: The relevance score (0-1)
        clarity: The clarity score (0-1)
        keyword_feedback: List of tuples containing (keyword, score) pairs
        
    Returns:
        str: Formatted feedback on the response structure
    """
    if keyword_feedback is None:
        keyword_feedback = []
        
    doc = nlp(response)
    sentences = list(doc.sents)
    
    # Calculate sentence length statistics
    sentence_lengths = [len(sent) for sent in sentences]
    avg_sentence_length = sum(sentence_lengths) / max(1, len(sentence_lengths))
    
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
    word_count = len(response.split())
    if word_count < 50:
        feedback.append("Try to provide more detail and examples to support your points.")
    elif word_count > 200:
        feedback.append("Consider being more concise while maintaining key points.")
    
    # Format the feedback
    formatted_feedback = " ".join(feedback)
    
    return formatted_feedback