import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DecisionLogger:
    """Logger for tracking emotion predictions and decisions."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize the decision logger.
        
        Args:
            log_dir (Path): Directory to store log files
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = None
        self.session_data = []
        logger.info(f"Initialized DecisionLogger in {log_dir}")
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new logging session.
        
        Args:
            session_id (Optional[str]): Custom session ID
            
        Returns:
            str: Session ID
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.current_session = session_id
        self.session_data = []
        logger.info(f"Started new logging session: {session_id}")
        return session_id
    
    def log_prediction(self, prediction_data: Dict) -> None:
        """
        Log a prediction decision.
        
        Args:
            prediction_data (Dict): Prediction data to log
        """
        if self.current_session is None:
            self.start_session()
        
        # Add timestamp
        entry = {
            "timestamp": datetime.now().isoformat(),
            **prediction_data
        }
        
        self.session_data.append(entry)
        logger.debug(f"Logged prediction for session {self.current_session}")
    
    def save_session(self) -> None:
        """Save the current session data to a file."""
        if not self.session_data:
            logger.warning("No data to save in current session")
            return
        
        try:
            # Create log file
            log_file = self.log_dir / f"session_{self.current_session}.json"
            
            # Save data
            with open(log_file, 'w') as f:
                json.dump({
                    "session_id": self.current_session,
                    "start_time": self.session_data[0]["timestamp"],
                    "end_time": self.session_data[-1]["timestamp"],
                    "predictions": self.session_data
                }, f, indent=2)
            
            logger.info(f"Saved session data to {log_file}")
            
        except Exception as e:
            logger.error(f"Error saving session data: {str(e)}")
    
    def load_session(self, session_id: str) -> Dict:
        """
        Load a previous session's data.
        
        Args:
            session_id (str): Session ID to load
            
        Returns:
            Dict: Session data
        """
        try:
            log_file = self.log_dir / f"session_{session_id}.json"
            
            if not log_file.exists():
                raise FileNotFoundError(f"Session {session_id} not found")
            
            with open(log_file, 'r') as f:
                session_data = json.load(f)
            
            logger.info(f"Loaded session data from {log_file}")
            return session_data
            
        except Exception as e:
            logger.error(f"Error loading session data: {str(e)}")
            raise
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict:
        """
        Get a summary of a session's predictions.
        
        Args:
            session_id (Optional[str]): Session ID to summarize
            
        Returns:
            Dict: Session summary
        """
        try:
            # Load session data
            if session_id is None:
                session_id = self.current_session
            
            if session_id is None:
                raise ValueError("No session ID provided")
            
            session_data = self.load_session(session_id)
            predictions = session_data["predictions"]
            
            # Calculate summary statistics
            emotion_counts = {}
            total_confidence = {}
            
            for pred in predictions:
                emotion = pred["final_prediction"]["emotion"]
                confidence = pred["final_prediction"]["confidence"]
                
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_confidence[emotion] = total_confidence.get(emotion, 0) + confidence
            
            # Calculate averages
            emotion_averages = {
                emotion: total_confidence[emotion] / count
                for emotion, count in emotion_counts.items()
            }
            
            return {
                "session_id": session_id,
                "duration": {
                    "start": session_data["start_time"],
                    "end": session_data["end_time"]
                },
                "total_predictions": len(predictions),
                "emotion_distribution": emotion_counts,
                "average_confidence": emotion_averages
            }
            
        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")
            raise
