from datetime import datetime
from typing import Dict, Optional, List, Tuple

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.api.schemas import (
    AnalysisResponse,
    ErrorResponse,
    SessionSummary,
    StreamResponse
)
from app.ensemble.ensemble_fusion import EnsembleFusion
from app.ensemble.decision_logger import DecisionLogger
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize router
router = APIRouter()

# Initialize components
ensemble = EnsembleFusion()
decision_logger = DecisionLogger(log_dir="logs/sessions")

@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_emotions(
    video_preds: List[Tuple[str, float]],
    audio_preds: List[Tuple[str, float]],
    text_preds: List[Tuple[str, float]]
) -> AnalysisResponse:
    """
    Analyze emotions from multiple modalities.
    
    Args:
        video_preds: Video modality predictions
        audio_preds: Audio modality predictions
        text_preds: Text modality predictions
        
    Returns:
        AnalysisResponse: Analysis results
    """
    try:
        # Perform ensemble analysis
        results = ensemble.analyze_emotions(video_preds, audio_preds, text_preds)
        
        # Log prediction
        decision_logger.log_prediction(results)
        
        return AnalysisResponse(**results)
        
    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/sessions/{session_id}", response_model=SessionSummary)
async def get_session_summary(session_id: str) -> SessionSummary:
    """
    Get summary of a session's predictions.
    
    Args:
        session_id: Session identifier
        
    Returns:
        SessionSummary: Session summary
    """
    try:
        summary = decision_logger.get_session_summary(session_id)
        return SessionSummary(**summary)
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    except Exception as e:
        logger.error(f"Error getting session summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.websocket("/ws/stream")
async def stream_analysis(websocket: WebSocket):
    """
    WebSocket endpoint for real-time emotion analysis.
    
    Args:
        websocket: WebSocket connection
    """
    await websocket.accept()
    
    try:
        # Start new session
        session_id = decision_logger.start_session()
        
        while True:
            # Receive data
            data = await websocket.receive_json()
            
            # Extract predictions
            video_preds = data.get("video_predictions", [])
            audio_preds = data.get("audio_predictions", [])
            text_preds = data.get("text_predictions", [])
            
            # Perform analysis
            results = ensemble.analyze_emotions(video_preds, audio_preds, text_preds)
            
            # Log prediction
            decision_logger.log_prediction(results)
            
            # Prepare response
            response = StreamResponse(
                frame_id=data.get("frame_id", ""),
                timestamp=datetime.now().isoformat(),
                analysis=results
            )
            
            # Send response
            await websocket.send_json(response.dict())
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        decision_logger.save_session()
        
    except Exception as e:
        logger.error(f"Error in WebSocket stream: {str(e)}")
        await websocket.send_json(
            ErrorResponse(
                error="Stream processing error",
                detail=str(e)
            ).dict()
        )
        decision_logger.save_session()
