"""
GPU-optimized video frame processing utility
"""
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Any
import torch

import src.config as config
from src.utils.gpu_check import log_memory_usage

class FrameProcessor:
    def __init__(self, face_detector, emotion_classifier):
        """
        Initialize the frame processor with detector and classifier.
        
        Args:
            face_detector: Face detector instance
            emotion_classifier: Emotion classifier instance
        """
        self.face_detector = face_detector
        self.emotion_classifier = emotion_classifier
        self.processed_frames = 0
        self.start_time = time.time()
        self.fps = 0
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame: detect faces and classify emotions.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (processed frame with annotations, list of face results)
        """
        # Check that frame is not None
        if frame is None:
            raise ValueError("Input frame is None")
            
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Detect faces - now returns DetectedFace objects
        detected_faces = self.face_detector.detect_faces(frame)
        
        # Extract face regions and prepare for classification
        face_crops = []
        valid_faces = []
        
        for face in detected_faces:
            # Extract face crop using the helper method
            face_crop = face.crop_from_frame(frame)
            
            # Skip invalid crops
            if face_crop is None:
                continue
                
            face_crops.append(face_crop)
            valid_faces.append(face)
        
        # Perform batch emotion classification if we have faces
        face_results = []
        if face_crops:
            # Run emotion prediction
            emotion_results = self.emotion_classifier.predict_batch(face_crops)
            
            # Combine face data with emotion results
            for i, (face, emotions) in enumerate(zip(valid_faces[:len(emotion_results)], emotion_results)):
                if emotions is None:
                    continue
                    
                # Get top emotion
                top_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_name, confidence = top_emotion
                
                # Add result to list
                face_results.append({
                    "box": face.box,
                    "landmarks": face.landmarks,
                    "detection_confidence": face.confidence,
                    "emotions": emotions,
                    "top_emotion": emotion_name,
                    "emotion_confidence": confidence
                })
                
                # Draw bounding box and emotion label
                x1, y1, x2, y2 = face.box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), config.BOX_COLOR, config.BOX_THICKNESS)
                
                # Draw emotion label
                label = f"{emotion_name}: {confidence:.2f}"
                cv2.putText(
                    display_frame, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    config.FONT_SCALE, 
                    config.TEXT_COLOR, 
                    config.FONT_THICKNESS
                )
                
                # Optionally draw landmarks
                if face.landmarks:
                    for landmark_name, (x, y) in face.landmarks.items():
                        cv2.circle(
                            display_frame,
                            (x, y),
                            2,
                            (0, 0, 255),  # Red color for landmarks
                            -1
                        )
        
        # Calculate and display FPS
        self.processed_frames += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            self.fps = self.processed_frames / elapsed_time
            self.processed_frames = 0
            self.start_time = time.time()
            
            # Log memory usage every 5 seconds
            if int(time.time()) % 5 == 0:
                log_memory_usage("During processing")
        
        # Display FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            display_frame, 
            fps_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            config.FONT_SCALE, 
            config.TEXT_COLOR, 
            config.FONT_THICKNESS
        )
        
        return display_frame, face_results 