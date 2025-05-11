"""
Shared data types for face detection and tracking
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np


@dataclass
class DetectedFace:
    """Data class to store detected face information"""
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Dict[str, Tuple[int, int]] = None  # Optional facial landmarks
    
    def crop_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract face region from frame based on bounding box"""
        x1, y1, x2, y2 = self.box
        # Ensure box coordinates are within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            return None
        # Extract face region
        return frame[y1:y2, x1:x2] 