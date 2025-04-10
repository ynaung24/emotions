import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

from app.config.settings import VIDEO_SETTINGS
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FaceDetector:
    """Face detection using MediaPipe Face Detection."""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=VIDEO_SETTINGS["face_detection_confidence"]
        )
        logger.info("Initialized FaceDetector")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        face_boxes = []
        if results.detections:
            height, width = frame.shape[:2]
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(width - x, w)
                h = min(height - y, h)
                
                face_boxes.append((x, y, w, h))
        
        return face_boxes
    
    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop a face from the frame using its bounding box.
        
        Args:
            frame (np.ndarray): Input frame
            bbox (Tuple[int, int, int, int]): Bounding box (x, y, w, h)
            
        Returns:
            Optional[np.ndarray]: Cropped face image or None if invalid bbox
        """
        x, y, w, h = bbox
        
        # Check if bbox is valid
        if w <= 0 or h <= 0:
            return None
        
        # Crop face
        face = frame[y:y+h, x:x+w]
        
        # Resize to standard size
        face = cv2.resize(face, (VIDEO_SETTINGS["frame_width"], VIDEO_SETTINGS["frame_height"]))
        
        return face
    
    def draw_faces(self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            frame (np.ndarray): Input frame
            face_boxes (List[Tuple[int, int, int, int]]): List of bounding boxes
            
        Returns:
            np.ndarray: Frame with drawn bounding boxes
        """
        output_frame = frame.copy()
        
        for x, y, w, h in face_boxes:
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return output_frame
