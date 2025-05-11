"""
Face tracking module to optimize performance by tracking faces between detection frames.
"""
import time
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.models.face_types import DetectedFace
import src.config as config

class FaceTracker:
    """
    Face tracker for optimizing GPU performance by minimizing expensive face detection operations.
    Uses OpenCV's KCF tracker to follow faces between detection frames.
    """
    
    def __init__(self):
        """Initialize the face tracker"""
        self.trackers = []
        self.tracked_faces = []
        self.last_track_time = time.time()
        self.track_id_counter = 0
        
    def update(self, frame: np.ndarray, detected_faces: Optional[List[DetectedFace]] = None) -> List[DetectedFace]:
        """
        Update trackers with new frame and/or detected faces.
        
        Args:
            frame: Current video frame
            detected_faces: New faces detected in this frame, or None if using existing trackers
            
        Returns:
            List of tracked face positions (DetectedFace objects)
        """
        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]
        
        # If we have new detections, reset all trackers
        if detected_faces is not None and len(detected_faces) > 0:
            # Clear existing trackers
            self.trackers = []
            self.tracked_faces = []
            
            # Initialize new trackers for each detected face
            for face in detected_faces:
                # Get bounding box
                x1, y1, x2, y2 = face.box
                
                # Create tracker (KCF is a good balance of accuracy and speed)
                tracker = cv2.TrackerKCF_create()
                
                # Initialize tracker with current frame and bounding box
                tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                
                # Assign a unique ID to this face
                face_id = self.track_id_counter
                self.track_id_counter += 1
                
                # Store tracker with ID and original face data
                self.trackers.append({
                    'tracker': tracker,
                    'id': face_id,
                    'last_seen': current_time,
                    'confidence': face.confidence,
                    'landmarks': face.landmarks
                })
                
                # Add to tracked faces
                self.tracked_faces.append(face)
            
            # Update last track time
            self.last_track_time = current_time
            
            return self.tracked_faces
        
        # If no new detections, update existing trackers
        elif self.trackers:
            # Reset tracked faces
            self.tracked_faces = []
            
            # Track each face in the new frame
            updated_trackers = []
            for tracker_info in self.trackers:
                tracker = tracker_info['tracker']
                
                # Update tracker with new frame
                success, bbox = tracker.update(frame)
                
                # If tracking successful, update face information
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    
                    # Ensure box is within frame bounds
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(frame_width, x + w)
                    y2 = min(frame_height, y + h)
                    
                    # Skip if box is invalid
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Create face object
                    face = DetectedFace(
                        box=(x1, y1, x2, y2),
                        confidence=tracker_info['confidence'] * 0.95,  # Slightly reduce confidence over time
                        landmarks=tracker_info['landmarks']  # Keep original landmarks
                    )
                    
                    # Add to tracked faces
                    self.tracked_faces.append(face)
                    
                    # Keep tracker if it's still valid
                    # Only keep trackers that are recent enough
                    if current_time - tracker_info['last_seen'] < config.TRACKER_PERSIST_TIME:
                        tracker_info['last_seen'] = current_time
                        updated_trackers.append(tracker_info)
            
            # Update trackers list
            self.trackers = updated_trackers
            
            # Update last track time
            self.last_track_time = current_time
            
            return self.tracked_faces
        
        # If no trackers and no detections, return empty list
        return []
    
    def get_tracked_faces(self) -> List[DetectedFace]:
        """Get the currently tracked faces"""
        return self.tracked_faces
    
    def reset(self):
        """Reset all trackers"""
        self.trackers = []
        self.tracked_faces = [] 