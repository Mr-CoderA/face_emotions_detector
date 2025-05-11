#!/usr/bin/env python
"""
GPU-only Face Emotion Detection Application

This application performs real-time face detection and emotion recognition
using GPU acceleration (CUDA) only.
"""
import sys
import os
import cv2
import torch
import argparse
import time

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import application modules
from src.utils.gpu_check import check_gpu_availability, cleanup_gpu_memory, log_memory_usage
from src.models.face_detector import FaceDetector
from src.models.emotion_classifier import EmotionClassifier
from src.utils.frame_processor import FrameProcessor
from src.utils.video_capture import VideoCaptureThread
import src.config as config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Face Emotion Detection with GPU")
    
    parser.add_argument(
        "--source", 
        type=str, 
        default="0", 
        help="Video source (camera index, video file path, or RTSP URL)"
    )
    
    parser.add_argument(
        "--width", 
        type=int, 
        default=config.DISPLAY_WIDTH, 
        help="Video width"
    )
    
    parser.add_argument(
        "--height", 
        type=int, 
        default=config.DISPLAY_HEIGHT, 
        help="Video height"
    )
    
    parser.add_argument(
        "--show_landmarks",
        action="store_true",
        help="Show facial landmarks in the output"
    )
    
    return parser.parse_args()

def main():
    """Main application entry point"""
    args = parse_args()
    
    # Check GPU availability
    try:
        check_gpu_availability()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("This application requires CUDA GPU acceleration.")
        return 1
        
    # Parse video source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    try:
        # Initialize components
        print("Initializing face detector (MediaPipe)...")
        face_detector = FaceDetector()  # MediaPipe doesn't require GPU
        
        print("Initializing emotion classifier...")
        emotion_classifier = EmotionClassifier(device="cuda")
        
        print("Initializing frame processor...")
        frame_processor = FrameProcessor(face_detector, emotion_classifier)
        
        print(f"Opening video source: {source}")
        video_capture = VideoCaptureThread(
            source=source,
            width=args.width,
            height=args.height
        )
        
        print("Starting processing loop...")
        
        # Processing loop
        while True:
            # Get frame from video capture
            ret, frame = video_capture.read_latest()
            
            if not ret:
                print("No frame available. Waiting...")
                time.sleep(0.1)
                continue
            
            # Process frame
            try:
                processed_frame, results = frame_processor.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow("Face Emotion Detection", processed_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                    
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory! Cleaning up and reducing batch size...")
                cleanup_gpu_memory()
                time.sleep(1)  # Give some time for memory to be freed
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Cleanup resources
        print("Cleaning up resources...")
        if 'video_capture' in locals():
            video_capture.stop()
        if 'face_detector' in locals():
            face_detector.release()
        if 'emotion_classifier' in locals():
            emotion_classifier.release()
            
        cv2.destroyAllWindows()
        cleanup_gpu_memory()
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 