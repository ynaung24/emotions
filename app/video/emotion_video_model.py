import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from app.config.settings import EMOTION_LABELS, MODEL_PATHS
from app.utils.file_utils import load_model, save_model
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmotionCNN(nn.Module):
    """CNN model for emotion classification from facial images."""
    
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, len(EMOTION_LABELS))
        
        logger.info("Initialized EmotionCNN model")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_emotions)
        """
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class VideoEmotionClassifier:
    """Wrapper class for video emotion classification."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionCNN().to(self.device)
        self.model = load_model(self.model, MODEL_PATHS["video"], "video_emotion")
        self.model.eval()
        logger.info(f"Initialized VideoEmotionClassifier on {self.device}")
    
    def preprocess_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a frame for emotion classification.
        
        Args:
            frame (torch.Tensor): Input frame tensor
            
        Returns:
            torch.Tensor: Preprocessed tensor
        """
        # Normalize to [0, 1]
        frame = frame.float() / 255.0
        
        # Add batch dimension if not present
        if len(frame.shape) == 3:
            frame = frame.unsqueeze(0)
        
        return frame.to(self.device)
    
    def predict(self, frame: torch.Tensor) -> Tuple[str, float]:
        """
        Predict emotion from a frame.
        
        Args:
            frame (torch.Tensor): Input frame tensor
            
        Returns:
            Tuple[str, float]: Predicted emotion and confidence
        """
        with torch.no_grad():
            # Preprocess frame
            frame = self.preprocess_frame(frame)
            
            # Get model predictions
            outputs = self.model(frame)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted emotion and confidence
            confidence, predicted = torch.max(probabilities, 1)
            emotion = EMOTION_LABELS[predicted.item()]
            confidence = confidence.item()
            
            return emotion, confidence
    
    def predict_batch(self, frames: List[torch.Tensor]) -> List[Tuple[str, float]]:
        """
        Predict emotions from a batch of frames.
        
        Args:
            frames (List[torch.Tensor]): List of input frame tensors
            
        Returns:
            List[Tuple[str, float]]: List of (emotion, confidence) pairs
        """
        predictions = []
        for frame in frames:
            emotion, confidence = self.predict(frame)
            predictions.append((emotion, confidence))
        return predictions
