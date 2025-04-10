from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class EmotionPrediction(BaseModel):
    """Schema for emotion prediction."""
    emotion: str = Field(..., description="Predicted emotion")
    confidence: float = Field(..., description="Prediction confidence", ge=0.0, le=1.0)
    intensity: str = Field(..., description="Emotion intensity level")

class ModalityPredictions(BaseModel):
    """Schema for modality-specific predictions."""
    video: Dict[str, float] = Field(..., description="Video modality predictions")
    audio: Dict[str, float] = Field(..., description="Audio modality predictions")
    text: Dict[str, float] = Field(..., description="Text modality predictions")

class AnalysisResponse(BaseModel):
    """Schema for emotion analysis response."""
    final_prediction: EmotionPrediction = Field(..., description="Final emotion prediction")
    modality_predictions: ModalityPredictions = Field(..., description="Modality-specific predictions")
    fused_probabilities: Dict[str, float] = Field(..., description="Fused probability distribution")

class SessionSummary(BaseModel):
    """Schema for session summary."""
    session_id: str = Field(..., description="Session identifier")
    duration: Dict[str, str] = Field(..., description="Session duration")
    total_predictions: int = Field(..., description="Total number of predictions")
    emotion_distribution: Dict[str, int] = Field(..., description="Distribution of emotions")
    average_confidence: Dict[str, float] = Field(..., description="Average confidence per emotion")

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

class StreamResponse(BaseModel):
    """Schema for streaming response."""
    frame_id: str = Field(..., description="Frame identifier")
    timestamp: str = Field(..., description="Frame timestamp")
    analysis: AnalysisResponse = Field(..., description="Frame analysis results")
