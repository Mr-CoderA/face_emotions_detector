#!/usr/bin/env python
"""
Test script for facial emotion recognition model performance

This script loads the emotion classification model and tests it on sample facial images
to evaluate accuracy and performance.
"""
import os
import sys
import time
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import application modules
from src.utils.gpu_check import check_gpu_availability, cleanup_gpu_memory, log_memory_usage
from src.models.emotion_classifier import EmotionClassifier
from src.models.face_detector import FaceDetector
import src.config as config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Facial Emotion Recognition Model")
    
    parser.add_argument(
        "--image", 
        type=str, 
        default=None, 
        help="Path to test image (if not provided, will look for sample images in ./test_images)"
    )
    
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./test_images",
        help="Directory containing test images"
    )
    
    parser.add_argument(
        "--detect_faces",
        action="store_true",
        help="Automatically detect faces in the images before emotion recognition"
    )
    
    return parser.parse_args()

def display_emotion_results(image, emotions, face_box=None, title="Emotion Results"):
    """Display emotion prediction results on the image"""
    h, w = image.shape[:2]
    result_img = np.zeros((h + 200, w, 3), dtype=np.uint8)
    result_img[0:h, 0:w] = image
    
    # If face box is provided, draw it
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Sort emotions by probability
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    
    # Draw emotions as a bar chart
    chart_top = h + 30
    bar_height = 20
    text_offset = 150
    max_bar_width = w - text_offset - 10
    
    for i, (emotion, prob) in enumerate(sorted_emotions):
        # Position for this emotion's bar
        y_pos = chart_top + i * (bar_height + 5)
        
        # Draw emotion name
        cv2.putText(
            result_img, 
            f"{emotion}:", 
            (10, y_pos + bar_height//2 + 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            1
        )
        
        # Draw probability bar
        bar_width = int(max_bar_width * prob)
        cv2.rectangle(
            result_img,
            (text_offset, y_pos),
            (text_offset + bar_width, y_pos + bar_height),
            (0, 255, 0),
            -1
        )
        
        # Draw probability text
        cv2.putText(
            result_img, 
            f"{prob:.2f}", 
            (text_offset + bar_width + 5, y_pos + bar_height//2 + 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            1
        )
    
    # Display top emotion on the image
    top_emotion = sorted_emotions[0][0]
    prob = sorted_emotions[0][1]
    cv2.putText(
        result_img, 
        f"{top_emotion}: {prob:.2f}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, 
        (0, 255, 0), 
        2
    )
    
    # Show the result
    cv2.imshow(title, result_img)
    key = cv2.waitKey(0)
    return key

def process_test_images(classifier, image_paths, face_detector=None):
    """Process each test image and display results"""
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # If face detector is provided, detect faces first
        if face_detector:
            print("Detecting faces...")
            detected_faces = face_detector.detect_faces(image)
            if not detected_faces:
                print("No faces detected in the image")
                continue
            
            # Process each detected face
            print(f"Found {len(detected_faces)} faces")
            for i, face in enumerate(detected_faces):
                face_crop = face.crop_from_frame(image)
                if face_crop is None:
                    continue
                
                # Get facial landmarks if available
                landmarks_text = ""
                if face.landmarks:
                    landmarks_text = " with landmarks"
                
                # Draw the detection
                annotated_img = face_detector.draw_detections(image, [face])
                
                # Predict emotions for this face
                start_time = time.time()
                emotions = classifier.predict_emotion(face_crop)
                elapsed_time = time.time() - start_time
                
                print(f"Face {i+1}{landmarks_text}: Prediction time: {elapsed_time:.3f} seconds")
                print(f"Face {i+1} emotions:")
                for emotion, prob in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {emotion}: {prob:.4f}")
                
                # Display result
                key = display_emotion_results(
                    annotated_img, 
                    emotions, 
                    face_box=face.box, 
                    title=f"Face {i+1} - {img_path.name}"
                )
                
                # Break on ESC key
                if key == 27:
                    return
        else:
            # Process the whole image without face detection
            start_time = time.time()
            emotions = classifier.predict_emotion(image)
            elapsed_time = time.time() - start_time
            
            print(f"Prediction time: {elapsed_time:.3f} seconds")
            print("Emotions:")
            for emotion, prob in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {prob:.4f}")
                
            # Display result
            key = display_emotion_results(image, emotions, title=f"Emotion: {img_path.name}")
            
            # Break on ESC key
            if key == 27:
                break
                
    cv2.destroyAllWindows()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Check GPU availability
    try:
        check_gpu_availability()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("This test requires CUDA GPU acceleration.")
        return 1
    
    # Get test images
    if args.image:
        if os.path.exists(args.image):
            test_images = [Path(args.image)]
        else:
            print(f"Error: Image not found: {args.image}")
            return 1
    else:
        # Check test directory exists or create it
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            os.makedirs(str(test_dir), exist_ok=True)
            print(f"Created test directory: {test_dir}")
            print("Please place test facial images in this directory and run again.")
            return 0
            
        # Find all images in the test directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = []
        for ext in image_extensions:
            test_images.extend(list(test_dir.glob(f"*{ext}")))
        
        if not test_images:
            print(f"No images found in {test_dir}. Please add test images and try again.")
            return 0
    
    print(f"Found {len(test_images)} test images")
    
    try:
        # Initialize face detector if needed
        face_detector = None
        if args.detect_faces:
            print("Initializing MediaPipe face detector...")
            face_detector = FaceDetector()
        
        # Initialize emotion classifier
        print(f"Loading emotion classifier model: {config.EMOTION_MODEL_NAME}")
        start_time = time.time()
        emotion_classifier = EmotionClassifier(device="cuda")
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Process test images
        process_test_images(emotion_classifier, test_images, face_detector)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up resources
        if 'emotion_classifier' in locals():
            emotion_classifier.release()
        if face_detector:
            face_detector.release()
        cleanup_gpu_memory()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 