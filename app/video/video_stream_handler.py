import cv2
import numpy as np
from typing import Generator, Optional, Tuple

from app.config.settings import VIDEO_SETTINGS
from app.utils.logger import setup_logger
from app.video.face_detector import FaceDetector
from app.video.emotion_video_model import VideoEmotionClassifier

logger = setup_logger(__name__)

class VideoStreamHandler:
    """Handler for video stream capture and processing."""
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize video stream handler.
        
        Args:
            camera_id (int): Camera device ID
        """
        self.camera_id = camera_id
        self.cap = None
        self.face_detector = FaceDetector()
        self.emotion_classifier = VideoEmotionClassifier()
        logger.info(f"Initialized VideoStreamHandler with camera_id={camera_id}")
    
    def start_stream(self) -> None:
        """Start the video stream."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_SETTINGS["frame_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_SETTINGS["frame_height"])
            self.cap.set(cv2.CAP_PROP_FPS, VIDEO_SETTINGS["fps"])
            
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open video stream")
            
            logger.info("Started video stream")
        except Exception as e:
            logger.error(f"Error starting video stream: {str(e)}")
            raise
    
    def stop_stream(self) -> None:
        """Stop the video stream."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Stopped video stream")
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get a frame from the video stream.
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: 
                (original frame, processed frame with face detection and emotion)
        """
        if self.cap is None or not self.cap.isOpened():
            return None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        # Detect faces
        face_boxes = self.face_detector.detect_faces(frame)
        
        # Process frame with face detection visualization
        processed_frame = self.face_detector.draw_faces(frame, face_boxes)
        
        # Add emotion predictions if faces are detected
        if face_boxes:
            for bbox in face_boxes:
                face = self.face_detector.crop_face(frame, bbox)
                if face is not None:
                    # Convert to tensor and predict emotion
                    face_tensor = torch.from_numpy(face).permute(2, 0, 1)
                    emotion, confidence = self.emotion_classifier.predict(face_tensor)
                    
                    # Add emotion label to frame
                    x, y, w, h = bbox
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(processed_frame, label, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, processed_frame
    
    def stream_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generator for streaming frames.
        
        Yields:
            Tuple[np.ndarray, np.ndarray]: (original frame, processed frame)
        """
        while True:
            frame, processed_frame = self.get_frame()
            if frame is not None and processed_frame is not None:
                yield frame, processed_frame
    
    def __enter__(self):
        """Context manager entry."""
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_stream()
