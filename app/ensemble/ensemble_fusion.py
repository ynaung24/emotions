import numpy as np
from typing import Dict, List, Tuple

from app.config.settings import ENSEMBLE_WEIGHTS
from app.ensemble.emotion_labels import get_all_emotions
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EnsembleFusion:
    """Fusion of predictions from multiple modalities."""
    
    def __init__(self):
        """Initialize the ensemble fusion."""
        self.emotions = get_all_emotions()
        self.weights = ENSEMBLE_WEIGHTS
        logger.info("Initialized EnsembleFusion")
    
    def normalize_predictions(self, predictions: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Normalize predictions to a probability distribution.
        
        Args:
            predictions (List[Tuple[str, float]]): List of (emotion, confidence) pairs
            
        Returns:
            Dict[str, float]: Normalized probability distribution
        """
        # Initialize probabilities for all emotions
        probabilities = {emotion: 0.0 for emotion in self.emotions}
        
        # Sum up probabilities for each emotion
        for emotion, confidence in predictions:
            if emotion in probabilities:
                probabilities[emotion] += confidence
        
        # Normalize
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v/total for k, v in probabilities.items()}
        
        return probabilities
    
    def weighted_fusion(self, 
                       video_pred: Dict[str, float],
                       audio_pred: Dict[str, float],
                       text_pred: Dict[str, float]) -> Dict[str, float]:
        """
        Perform weighted fusion of predictions from different modalities.
        
        Args:
            video_pred (Dict[str, float]): Video modality predictions
            audio_pred (Dict[str, float]): Audio modality predictions
            text_pred (Dict[str, float]): Text modality predictions
            
        Returns:
            Dict[str, float]: Fused predictions
        """
        # Initialize fused probabilities
        fused_prob = {emotion: 0.0 for emotion in self.emotions}
        
        # Weighted sum of probabilities
        for emotion in self.emotions:
            fused_prob[emotion] = (
                self.weights["video"] * video_pred.get(emotion, 0.0) +
                self.weights["audio"] * audio_pred.get(emotion, 0.0) +
                self.weights["text"] * text_pred.get(emotion, 0.0)
            )
        
        # Normalize
        total = sum(fused_prob.values())
        if total > 0:
            fused_prob = {k: v/total for k, v in fused_prob.items()}
        
        return fused_prob
    
    def get_final_prediction(self, fused_prob: Dict[str, float]) -> Tuple[str, float]:
        """
        Get the final emotion prediction from fused probabilities.
        
        Args:
            fused_prob (Dict[str, float]): Fused probability distribution
            
        Returns:
            Tuple[str, float]: (predicted emotion, confidence)
        """
        # Get emotion with highest probability
        emotion = max(fused_prob.items(), key=lambda x: x[1])
        return emotion[0], emotion[1]
    
    def get_emotion_intensity(self, emotion: str, confidence: float) -> str:
        """
        Determine emotion intensity based on confidence.
        
        Args:
            emotion (str): Predicted emotion
            confidence (float): Prediction confidence
            
        Returns:
            str: Intensity level
        """
        if confidence < 0.4:
            return "mild"
        elif confidence < 0.7:
            return "moderate"
        else:
            return "intense"
    
    def analyze_emotions(self,
                        video_preds: List[Tuple[str, float]],
                        audio_preds: List[Tuple[str, float]],
                        text_preds: List[Tuple[str, float]]) -> Dict:
        """
        Analyze emotions from all modalities.
        
        Args:
            video_preds (List[Tuple[str, float]]): Video predictions
            audio_preds (List[Tuple[str, float]]): Audio predictions
            text_preds (List[Tuple[str, float]]): Text predictions
            
        Returns:
            Dict: Analysis results including fused prediction and modality-specific results
        """
        # Normalize predictions from each modality
        video_prob = self.normalize_predictions(video_preds)
        audio_prob = self.normalize_predictions(audio_preds)
        text_prob = self.normalize_predictions(text_preds)
        
        # Perform fusion
        fused_prob = self.weighted_fusion(video_prob, audio_prob, text_prob)
        
        # Get final prediction
        emotion, confidence = self.get_final_prediction(fused_prob)
        intensity = self.get_emotion_intensity(emotion, confidence)
        
        return {
            "final_prediction": {
                "emotion": emotion,
                "confidence": confidence,
                "intensity": intensity
            },
            "modality_predictions": {
                "video": video_prob,
                "audio": audio_prob,
                "text": text_prob
            },
            "fused_probabilities": fused_prob
        }
