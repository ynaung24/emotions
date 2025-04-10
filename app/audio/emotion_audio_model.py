import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from app.config.settings import EMOTION_LABELS, MODEL_PATHS
from app.utils.file_utils import load_model, save_model
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmotionRNN(nn.Module):
    """RNN model for emotion classification from audio features."""
    
    def __init__(self, input_size: int = 13, hidden_size: int = 128):
        """
        Initialize the RNN model.
        
        Args:
            input_size (int): Size of input features (MFCC coefficients)
            hidden_size (int): Size of hidden state
        """
        super(EmotionRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, len(EMOTION_LABELS))
        
        logger.info("Initialized EmotionRNN model")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_emotions)
        """
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output layer
        output = self.fc(attended)
        
        return output

class AudioEmotionClassifier:
    """Wrapper class for audio emotion classification."""
    
    def __init__(self):
        """Initialize the audio emotion classifier."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionRNN().to(self.device)
        self.model = load_model(self.model, MODEL_PATHS["audio"], "audio_emotion")
        self.model.eval()
        logger.info(f"Initialized AudioEmotionClassifier on {self.device}")
    
    def preprocess_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Preprocess features for emotion classification.
        
        Args:
            features (torch.Tensor): Input features tensor
            
        Returns:
            torch.Tensor: Preprocessed tensor
        """
        # Add batch dimension if not present
        if len(features.shape) == 2:
            features = features.unsqueeze(0)
        
        return features.to(self.device)
    
    def predict(self, features: torch.Tensor) -> Tuple[str, float]:
        """
        Predict emotion from audio features.
        
        Args:
            features (torch.Tensor): Input features tensor
            
        Returns:
            Tuple[str, float]: Predicted emotion and confidence
        """
        with torch.no_grad():
            # Preprocess features
            features = self.preprocess_features(features)
            
            # Get model predictions
            outputs = self.model(features)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted emotion and confidence
            confidence, predicted = torch.max(probabilities, 1)
            emotion = EMOTION_LABELS[predicted.item()]
            confidence = confidence.item()
            
            return emotion, confidence
    
    def predict_batch(self, features_list: List[torch.Tensor]) -> List[Tuple[str, float]]:
        """
        Predict emotions from a batch of feature tensors.
        
        Args:
            features_list (List[torch.Tensor]): List of input feature tensors
            
        Returns:
            List[Tuple[str, float]]: List of (emotion, confidence) pairs
        """
        predictions = []
        for features in features_list:
            emotion, confidence = self.predict(features)
            predictions.append((emotion, confidence))
        return predictions
