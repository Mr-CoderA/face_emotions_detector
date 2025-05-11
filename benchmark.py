#!/usr/bin/env python
"""
GPU Benchmark for Face Emotion Detection

This script benchmarks the face emotion detection components to measure:
1. GPU memory usage
2. Processing speed (FPS)
3. Model loading times
"""
import os
import sys
import time
import cv2
import numpy as np
import torch
import argparse

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import application modules
from src.utils.gpu_check import check_gpu_availability, cleanup_gpu_memory, log_memory_usage
from src.models.face_detector import FaceDetector
from src.models.emotion_classifier import EmotionClassifier
from src.utils.frame_processor import FrameProcessor
import src.config as config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark GPU Face Emotion Detection")
    
    parser.add_argument(
        "--test_image", 
        type=str, 
        default=None, 
        help="Path to test image (if not provided, a synthetic test image will be generated)"
    )
    
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=100, 
        help="Number of iterations for the benchmark"
    )
    
    parser.add_argument(
        "--num_faces", 
        type=int, 
        default=4, 
        help="Number of synthetic faces to generate (if test_image not provided)"
    )
    
    return parser.parse_args()

def generate_test_image(num_faces=4, width=640, height=480):
    """Generate a synthetic test image with random colored rectangles for face simulation"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate random "faces" (colored rectangles)
    for _ in range(num_faces):
        # Random face size between 80x80 and 200x200
        face_width = np.random.randint(80, 200)
        face_height = np.random.randint(80, 200)
        
        # Random position ensuring face is within image bounds
        x = np.random.randint(0, width - face_width)
        y = np.random.randint(0, height - face_height)
        
        # Random color
        color = (
            np.random.randint(50, 255),
            np.random.randint(50, 255),
            np.random.randint(50, 255)
        )
        
        # Draw filled rectangle
        cv2.rectangle(image, (x, y), (x + face_width, y + face_height), color, -1)
        
        # Draw facial features (simple eyes and mouth)
        eye_size = max(face_width, face_height) // 10
        cv2.circle(image, (x + face_width//3, y + face_height//3), eye_size, (0, 0, 0), -1)
        cv2.circle(image, (x + 2*face_width//3, y + face_height//3), eye_size, (0, 0, 0), -1)
        
        # Draw mouth
        mouth_y = y + 2*face_height//3
        cv2.line(image, (x + face_width//4, mouth_y), (x + 3*face_width//4, mouth_y), (0, 0, 0), 2)
    
    return image

def benchmark_face_detector(face_detector, test_image, iterations=100):
    """Benchmark face detector performance"""
    print("\n===== Face Detector Benchmark =====")
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        face_detector.detect_faces(test_image)
    
    # Benchmark
    print(f"Running benchmark for {iterations} iterations...")
    start_time = time.time()
    
    for i in range(iterations):
        faces = face_detector.detect_faces(test_image)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Detected {len(faces)} faces")
    
    elapsed_time = time.time() - start_time
    fps = iterations / elapsed_time
    
    print(f"Detected {len(faces)} faces in last iteration")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    # Draw the last detection for visualization
    if faces:
        annotated_image = face_detector.draw_detections(test_image, faces)
        cv2.imshow("Face Detection Result", annotated_image)
        cv2.waitKey(1)
    
    log_memory_usage("After face detection benchmark")
    return fps

def benchmark_emotion_classifier(emotion_classifier, test_image, face_detector, iterations=100):
    """Benchmark emotion classifier performance"""
    print("\n===== Emotion Classifier Benchmark =====")
    
    # Get face crops from detector
    detected_faces = face_detector.detect_faces(test_image)
    face_crops = []
    
    for face in detected_faces:
        face_crop = face.crop_from_frame(test_image)
        if face_crop is not None:
            face_crops.append(face_crop)
    
    if not face_crops:
        print("No faces detected for emotion classification benchmark")
        return 0
    
    print(f"Testing with {len(face_crops)} face crops")
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        emotion_classifier.predict_batch(face_crops)
    
    # Benchmark
    print(f"Running benchmark for {iterations} iterations...")
    start_time = time.time()
    
    for i in range(iterations):
        emotions = emotion_classifier.predict_batch(face_crops)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Processed {len(emotions)} faces")
    
    elapsed_time = time.time() - start_time
    fps = iterations / elapsed_time
    
    print(f"Classified {len(emotions)} faces in last iteration")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    # Show top emotion for each face
    print("\nEmotion Classification Results:")
    for i, emotion_dict in enumerate(emotions):
        if emotion_dict:
            top_emotion = max(emotion_dict.items(), key=lambda x: x[1])
            print(f"Face {i+1}: {top_emotion[0]} ({top_emotion[1]:.2f})")
    
    log_memory_usage("After emotion classification benchmark")
    return fps

def benchmark_full_pipeline(frame_processor, test_image, iterations=100):
    """Benchmark the full processing pipeline"""
    print("\n===== Full Pipeline Benchmark =====")
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        frame_processor.process_frame(test_image)
    
    # Benchmark
    print(f"Running benchmark for {iterations} iterations...")
    start_time = time.time()
    
    for i in range(iterations):
        processed_frame, results = frame_processor.process_frame(test_image)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Processed frame with {len(results)} face results")
    
    elapsed_time = time.time() - start_time
    fps = iterations / elapsed_time
    
    print(f"Processed frame with {len(results)} face results in last iteration")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    log_memory_usage("After full pipeline benchmark")
    
    # Display the processed frame
    cv2.imshow("Benchmark Result", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return fps

def main():
    """Main benchmark entry point"""
    args = parse_args()
    
    # Check GPU availability
    try:
        check_gpu_availability()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("This benchmark requires CUDA GPU acceleration.")
        return 1
    
    # Load or generate test image
    if args.test_image and os.path.exists(args.test_image):
        print(f"Loading test image: {args.test_image}")
        test_image = cv2.imread(args.test_image)
    else:
        print(f"Generating synthetic test image with {args.num_faces} faces")
        test_image = generate_test_image(
            num_faces=args.num_faces,
            width=config.DISPLAY_WIDTH,
            height=config.DISPLAY_HEIGHT
        )
    
    # Resize image if needed
    if test_image.shape[1] != config.DISPLAY_WIDTH or test_image.shape[0] != config.DISPLAY_HEIGHT:
        test_image = cv2.resize(test_image, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
    
    # Display test image
    cv2.imshow("Test Image", test_image)
    cv2.waitKey(1)
    
    try:
        # Load face detector
        start_time = time.time()
        face_detector = FaceDetector()  # No need to specify device for MediaPipe
        face_detector_load_time = time.time() - start_time
        print(f"Face detector loaded in {face_detector_load_time:.2f} seconds")
        
        # Load emotion classifier
        start_time = time.time()
        emotion_classifier = EmotionClassifier(device="cuda")
        emotion_classifier_load_time = time.time() - start_time
        print(f"Emotion classifier loaded in {emotion_classifier_load_time:.2f} seconds")
        
        # Initialize frame processor
        frame_processor = FrameProcessor(face_detector, emotion_classifier)
        
        # Run benchmarks
        face_detector_fps = benchmark_face_detector(
            face_detector, 
            test_image, 
            iterations=args.iterations
        )
        
        emotion_classifier_fps = benchmark_emotion_classifier(
            emotion_classifier, 
            test_image, 
            face_detector, 
            iterations=args.iterations
        )
        
        full_pipeline_fps = benchmark_full_pipeline(
            frame_processor, 
            test_image, 
            iterations=args.iterations
        )
        
        # Print summary
        print("\n===== Benchmark Summary =====")
        print(f"Face Detector Load Time: {face_detector_load_time:.2f} seconds")
        print(f"Emotion Classifier Load Time: {emotion_classifier_load_time:.2f} seconds")
        print(f"Face Detector FPS: {face_detector_fps:.2f}")
        print(f"Emotion Classifier FPS: {emotion_classifier_fps:.2f}")
        print(f"Full Pipeline FPS: {full_pipeline_fps:.2f}")
        log_memory_usage("Final")
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return 1
    finally:
        # Clean up
        if 'face_detector' in locals():
            face_detector.release()
        if 'emotion_classifier' in locals():
            emotion_classifier.release()
        cleanup_gpu_memory()
        cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 