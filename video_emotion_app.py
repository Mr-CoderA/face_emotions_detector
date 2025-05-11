#!/usr/bin/env python
"""
Video File Facial Emotion Detection

Processes a video file, detects faces, and performs emotion recognition.
Saves the processed video with emotion annotations.
"""
import os
import sys
import cv2
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import application modules
from src.utils.gpu_check import check_gpu_availability, cleanup_gpu_memory, log_memory_usage
from src.models.face_detector import FaceDetector
from src.models.emotion_classifier import EmotionClassifier
import src.config as config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Video File Facial Emotion Detection")
    
    parser.add_argument(
        "input_video",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Path to output video file (default: input_filename_emotions.mp4)"
    )
    
    parser.add_argument(
        "--resolution", 
        type=str, 
        default="640x480", 
        help="Output resolution (default: 640x480)"
    )
    
    parser.add_argument(
        "--show_landmarks", 
        action="store_true", 
        help="Show facial landmarks"
    )
    
    parser.add_argument(
        "--preview", 
        action="store_true", 
        help="Show preview window during processing"
    )
    
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Speed up or slow down factor (e.g., 2.0 for double speed)"
    )
    
    return parser.parse_args()

def draw_emotions(frame, faces, emotions_list, show_landmarks=True):
    """Draw emotion labels and face boxes on the frame"""
    result_frame = frame.copy()
    
    for face, emotions in zip(faces, emotions_list):
        if emotions is None:
            continue
            
        # Get bounding box
        x1, y1, x2, y2 = face.box
        
        # Get top emotion
        top_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion_name, confidence = top_emotion
        
        # Draw bounding box
        cv2.rectangle(
            result_frame,
            (x1, y1),
            (x2, y2),
            config.BOX_COLOR,
            config.BOX_THICKNESS
        )
        
        # Draw emotion label
        label = f"{emotion_name}: {confidence:.2f}"
        cv2.putText(
            result_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            config.TEXT_COLOR,
            config.FONT_THICKNESS
        )
        
        # Draw facial landmarks if available and enabled
        if show_landmarks and face.landmarks:
            for landmark_name, (x, y) in face.landmarks.items():
                cv2.circle(
                    result_frame,
                    (x, y),
                    3,  # Slightly larger radius
                    (0, 0, 255),  # Red color
                    -1  # Filled circle
                )
    
    # Add processing info
    info_text = f"Processing: {current_fps:.1f} FPS | Frame: {current_frame}/{total_frames}"
    cv2.putText(
        result_frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # Larger font
        (0, 255, 255),  # Yellow color
        2
    )
    
    return result_frame

def main():
    """Main function"""
    global current_fps, current_frame, total_frames
    current_fps = 0
    current_frame = 0
    total_frames = 0
    
    args = parse_args()
    
    # Check if input video exists
    input_path = Path(args.input_video)
    if not input_path.exists():
        print(f"Error: Input video not found: {input_path}")
        return 1
    
    # Set output path if not specified
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_emotions.mp4"
    else:
        output_path = Path(args.output)
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}. Using default 640x480.")
        width, height = 640, 480
    
    # Check GPU availability
    try:
        check_gpu_availability()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("This application requires CUDA GPU acceleration.")
        return 1
    
    try:
        # Initialize models
        print(f"Initializing MediaPipe face detector...")
        face_detector = FaceDetector()
        
        print(f"Loading emotion recognition model: {config.EMOTION_MODEL_NAME}")
        emotion_classifier = EmotionClassifier(device="cuda")
        
        # Open input video
        print(f"Opening video: {input_path}")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video: {input_path}")
            return 1
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {original_width}x{original_height}, {fps} FPS, {total_frames} frames")
        
        # Calculate processing FPS based on speedup factor
        target_fps = fps * args.speedup
        frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        
        # Initialize output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Variables for processing
        frame_count = 0
        fps_start_time = time.time()
        processing_times = []
        
        print(f"Processing video... Output will be saved to: {output_path}")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = frame_count + 1
            
            # Resize frame if needed
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # Start measuring processing time
            start_time = time.time()
            
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            # Extract face crops
            face_crops = []
            valid_faces = []
            
            for face in faces:
                face_crop = face.crop_from_frame(frame)
                if face_crop is not None:
                    face_crops.append(face_crop)
                    valid_faces.append(face)
            
            # Classify emotions
            emotions_list = []
            if face_crops:
                emotions_list = emotion_classifier.predict_batch(face_crops)
            
            # Draw results on frame
            result_frame = draw_emotions(frame, valid_faces, emotions_list, show_landmarks=args.show_landmarks)
            
            # Write frame to output video
            out.write(result_frame)
            
            # Display preview if enabled
            if args.preview:
                cv2.imshow("Processing Video", result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("Processing canceled by user")
                    break
            
            # Calculate processing time for this frame
            elapsed_time = time.time() - start_time
            processing_times.append(elapsed_time)
            if len(processing_times) > 30:
                processing_times.pop(0)  # Keep only the last 30 frames
            
            # Calculate FPS
            frame_count += 1
            fps_elapsed_time = time.time() - fps_start_time
            
            # Print progress every second
            if fps_elapsed_time >= 1.0:
                current_fps = frame_count / fps_elapsed_time
                frame_count = 0
                fps_start_time = time.time()
                
                # Log memory usage every 5 seconds
                if int(time.time()) % 5 == 0:
                    log_memory_usage("During processing")
                
                # Print progress
                progress = (current_frame / total_frames) * 100
                if processing_times:
                    avg_time = sum(processing_times) / len(processing_times)
                    estimated_remaining = avg_time * (total_frames - current_frame)
                    print(f"Progress: {progress:.1f}% ({current_frame}/{total_frames}) | "
                          f"FPS: {current_fps:.1f} | "
                          f"Est. remaining: {estimated_remaining/60:.1f} minutes")
            
            # Limit processing speed to match target FPS (if preview is enabled)
            if args.preview and frame_interval > 0:
                processing_time = time.time() - start_time
                if processing_time < frame_interval:
                    time.sleep(frame_interval - processing_time)
        
        # Finalize video
        cap.release()
        out.release()
        
        if args.preview:
            cv2.destroyAllWindows()
            
        print(f"Processing complete! Output saved to: {output_path}")
    
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory! Please reduce the resolution or close other applications.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        if 'face_detector' in locals():
            face_detector.release()
        if 'emotion_classifier' in locals():
            emotion_classifier.release()
        if args.preview:
            cv2.destroyAllWindows()
        cleanup_gpu_memory()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 