"""
GPU-optimized face detection module using MediaPipe Face Detection model
"""
import time
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Dict, Any

from src.utils.gpu_check import log_memory_usage
from src.models.face_tracker import FaceTracker
from src.models.face_types import DetectedFace
import src.config as config

class FaceDetector:
    def __init__(self, device="cpu"):
        """
        Initialize the MediaPipe face detector.
        
        Args:
            device: Ignored in MediaPipe implementation, kept for compatibility
        """
        # Set up MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        
        # Initialize MediaPipe Face Detection with model selection and minimum confidence
        self.detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range (2m), 1 for full-range (5m)
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
        )
        
        # MediaPipe uses CPU but is highly optimized
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face tracker if enabled
        self.face_tracker = FaceTracker() if config.FACE_TRACKING_ENABLED else None
        
        # Log memory usage after model loading
        log_memory_usage("After face detector loaded")
        
        # Frame counter for detection interval
        self.frame_count = 0
        
        # Cache for face detections
        self.cached_faces = []
        self.last_detection_time = time.time()
    
    def detect_faces(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in a given frame using MediaPipe.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of DetectedFace objects
        """
        # Increment frame counter
        self.frame_count += 1
        current_time = time.time()
        
        # Check if we should perform detection or use tracking
        perform_detection = (
            self.frame_count % config.DETECTION_INTERVAL == 0 or
            current_time - self.last_detection_time > 1.0  # Force detect at least every second
        )
        
        # If tracking is enabled and we don't need to detect, use tracking
        if self.face_tracker and not perform_detection:
            # Update tracker with current frame
            tracked_faces = self.face_tracker.update(frame)
            
            # Use tracked faces if available
            if tracked_faces and len(tracked_faces) > 0:
                self.cached_faces = tracked_faces
                return self.cached_faces
        
        # Perform face detection if needed
        if perform_detection:
            # Resize frame for detection if needed (to save memory)
            detection_frame = frame
            if config.FACE_DETECTION_RESIZE_FACTOR < 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * config.FACE_DETECTION_RESIZE_FACTOR), int(w * config.FACE_DETECTION_RESIZE_FACTOR)
                detection_frame = cv2.resize(frame, (new_w, new_h))
                scale_factor = 1.0 / config.FACE_DETECTION_RESIZE_FACTOR
            else:
                scale_factor = 1.0
            
            # MediaPipe requires RGB input
            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            h, w = detection_frame.shape[:2]
            
            # Process the frame
            results = self.detector.process(rgb_frame)
            
            # Store timestamp of this detection
            self.last_detection_time = current_time
            
            # Clear cached faces
            self.cached_faces = []
            
            # Check if faces were detected
            if results.detections:
                # Limit number of faces
                for detection in results.detections[:config.MAX_FACES]:
                    # Get detection confidence
                    confidence = detection.score[0]
                    
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w * scale_factor)
                    y1 = int(bbox.ymin * h * scale_factor)
                    width = int(bbox.width * w * scale_factor)
                    height = int(bbox.height * h * scale_factor)
                    x2 = x1 + width
                    y2 = y1 + height
                    
                    # Get facial landmarks (if needed later)
                    landmarks = {}
                    landmarks_proto = detection.location_data.relative_keypoints
                    
                    # Mapping of keypoint indices to names
                    keypoint_names = ["right_eye", "left_eye", "nose_tip", "mouth_center", "right_ear", "left_ear"]
                    
                    # Extract keypoints
                    for i, keypoint in enumerate(landmarks_proto):
                        if i < len(keypoint_names):
                            # Convert relative coordinates to absolute
                            x = int(keypoint.x * w * scale_factor)
                            y = int(keypoint.y * h * scale_factor)
                            landmarks[keypoint_names[i]] = (x, y)
                    
                    # Create DetectedFace object
                    face = DetectedFace(
                        box=(x1, y1, x2, y2),
                        confidence=confidence,
                        landmarks=landmarks
                    )
                    
                    # Add to cache
                    self.cached_faces.append(face)
                
                # Update face tracker with new detections if enabled
                if self.face_tracker:
                    self.face_tracker.update(frame, self.cached_faces)
        
        return self.cached_faces
    
    def create_face_from_box(self, box, landmarks=None, confidence=0.8):
        """
        Create a DetectedFace object from a bounding box and optional landmarks.
        Useful for reusing face detections across frames for better performance.
        
        Args:
            box: Tuple of (x1, y1, x2, y2)
            landmarks: Optional dictionary of facial landmarks
            confidence: Detection confidence
            
        Returns:
            DetectedFace object
        """
        return DetectedFace(
            box=box,
            confidence=confidence,
            landmarks=landmarks
        )
    
    def draw_detections(self, frame: np.ndarray, faces: List[DetectedFace] = None) -> np.ndarray:
        """
        Draw face detections on the frame.
        
        Args:
            frame: Input frame
            faces: List of DetectedFace objects (if None, uses cached faces)
            
        Returns:
            Frame with detection annotations
        """
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Use provided faces or cached ones
        faces_to_draw = faces if faces is not None else self.cached_faces
        
        # Draw each face
        for face in faces_to_draw:
            x1, y1, x2, y2 = face.box
            
            # Draw bounding box
            cv2.rectangle(
                annotated_frame, 
                (x1, y1), 
                (x2, y2), 
                config.BOX_COLOR, 
                config.BOX_THICKNESS
            )
            
            # Optionally draw confidence
            confidence_text = f"Conf: {face.confidence:.2f}"
            cv2.putText(
                annotated_frame,
                confidence_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.TEXT_COLOR,
                config.FONT_THICKNESS
            )
            
            # Optionally draw facial landmarks
            if face.landmarks:
                for landmark_name, (x, y) in face.landmarks.items():
                    cv2.circle(
                        annotated_frame,
                        (x, y),
                        2,
                        (0, 0, 255),
                        -1
                    )
        
        return annotated_frame
    
    def release(self):
        """
        Release resources used by the detector.
        """
        # Close the detector
        self.detector.close()
        
        # Reset the tracker if enabled
        if self.face_tracker:
            self.face_tracker.reset() 