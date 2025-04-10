import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict, List, Tuple

from app.config.settings import EMOTION_LABELS, MODEL_PATHS, TEXT_SETTINGS
from app.utils.file_utils import load_model, save_model
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmotionBERT:
    """BERT model for emotion classification from text."""
    
    def __init__(self):
        """Initialize the BERT model and tokenizer."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(EMOTION_LABELS)
            )
            self.model = load_model(self.model, MODEL_PATHS["text"], "text_emotion")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Initialized EmotionBERT model on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing BERT model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for BERT model.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized inputs
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            max_length=TEXT_SETTINGS["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict emotion from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[str, float]: Predicted emotion and confidence
        """
        try:
            with torch.no_grad():
                # Preprocess text
                inputs = self.preprocess_text(text)
                
                # Get model predictions
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                # Get predicted emotion and confidence
                confidence, predicted = torch.max(probabilities, 1)
                emotion = EMOTION_LABELS[predicted.item()]
                confidence = confidence.item()
                
                return emotion, confidence
                
        except Exception as e:
            logger.error(f"Error predicting emotion from text: {str(e)}")
            return "neutral", 0.0
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict emotions from a batch of texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[Tuple[str, float]]: List of (emotion, confidence) pairs
        """
        predictions = []
        for text in texts:
            emotion, confidence = self.predict(text)
            predictions.append((emotion, confidence))
        return predictions
    
    def get_emotion_scores(self, text: str) -> Dict[str, float]:
        """
        Get emotion scores for all emotions.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Dictionary of emotion scores
        """
        try:
            with torch.no_grad():
                # Preprocess text
                inputs = self.preprocess_text(text)
                
                # Get model predictions
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                # Convert to dictionary
                scores = {
                    emotion: prob.item()
                    for emotion, prob in zip(EMOTION_LABELS, probabilities[0])
                }
                
                return scores
                
        except Exception as e:
            logger.error(f"Error getting emotion scores: {str(e)}")
            return {emotion: 0.0 for emotion in EMOTION_LABELS}
