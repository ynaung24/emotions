import re
from typing import List

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TranscriptPreprocessor:
    """Preprocessor for cleaning and preparing transcribed text."""
    
    def __init__(self):
        """Initialize the transcript preprocessor."""
        # Common abbreviations and their expansions
        self.abbreviations = {
            "mr.": "mister",
            "mrs.": "missus",
            "dr.": "doctor",
            "prof.": "professor",
            "e.g.": "for example",
            "i.e.": "that is",
            "etc.": "and so forth",
        }
        logger.info("Initialized TranscriptPreprocessor")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Expand abbreviations
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(abbr, expansion)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        # Basic sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def remove_filler_words(self, text: str) -> str:
        """
        Remove common filler words.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with filler words removed
        """
        filler_words = {
            "um", "uh", "ah", "er", "like", "you know",
            "sort of", "kind of", "basically", "literally"
        }
        
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in filler_words]
        return " ".join(filtered_words)
    
    def preprocess(self, text: str) -> List[str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of preprocessed sentences
        """
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Remove filler words
            cleaned_text = self.remove_filler_words(cleaned_text)
            
            # Split into sentences
            sentences = self.split_into_sentences(cleaned_text)
            
            logger.debug(f"Preprocessed {len(sentences)} sentences")
            return sentences
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return []
