#!/usr/bin/env python
"""
Face Emotion Detection Optimization Benchmark

Evaluates performance across different optimization configurations.
"""
import os
import sys
import time
import cv2
import torch
import numpy as np
import argparse
from tabulate import tabulate
from tqdm import tqdm

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import application modules
from src.utils.gpu_check import check_gpu_availability, cleanup_gpu_memory, log_memory_usage
from src.utils.memory_manager import GPUMemoryManager
from src.models.face_detector import FaceDetector
from src.models.emotion_classifier import EmotionClassifier
import src.config as config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Face Emotion Detection Optimization Benchmark")
    
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to test video file (default: use webcam)"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index if no video file specified (default: 0)"
    )
    
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process for each configuration (default: 100)"
    )
    
    parser.add_argument(
        "--resolution",
        type=str,
        default="640x480",
        help="Processing resolution (default: 640x480)"
    )
    
    return parser.parse_args()

def benchmark_configuration(video_source, num_frames, width, height, config_name, config_dict):
    """
    Benchmark a specific configuration
    
    Args:
        video_source: Video file path or camera index
        num_frames: Number of frames to process
        width, height: Resolution to use
        config_name: Name of this configuration
        config_dict: Dictionary of configuration parameters
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\nBenchmarking configuration: {config_name}")
    
    # Override config settings temporarily
    original_values = {}
    for key, value in config_dict.items():
        if hasattr(config, key):
            original_values[key] = getattr(config, key)
            setattr(config, key, value)
    
    # Initialize memory manager
    memory_manager = GPUMemoryManager(
        optimization_level=config_dict.get("MEMORY_OPTIMIZATION", config.MEMORY_OPTIMIZATION)
    )
    memory_manager.start_monitoring()
    
    # Initialize video source
    if isinstance(video_source, str) and os.path.exists(video_source):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(int(video_source))
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source")
        return None
    
    try:
        # Initialize models
        face_detector = FaceDetector()
        emotion_classifier = EmotionClassifier(device="cuda")
        
        # Performance metrics
        frame_times = []
        detection_times = []
        emotion_times = []
        max_memory = 0
        max_memory_pressure = 0
        face_counts = []
        
        # Process frames
        for _ in tqdm(range(num_frames), desc=f"Testing {config_name}"):
            # Read frame
            ret, frame = cap.read()
            if not ret:
                # If video ended, seek back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Start timing
            frame_start = time.time()
            
            # Detect faces
            detection_start = time.time()
            faces = face_detector.detect_faces(frame)
            detection_end = time.time()
            
            # Extract face crops
            face_crops = []
            valid_faces = []
            
            for face in faces:
                face_crop = face.crop_from_frame(frame)
                if face_crop is not None:
                    face_crops.append(face_crop)
                    valid_faces.append(face)
            
            # Record face count
            face_counts.append(len(valid_faces))
            
            # Classify emotions
            if face_crops:
                emotion_start = time.time()
                emotion_classifier.predict_batch(face_crops)
                emotion_end = time.time()
                emotion_times.append(emotion_end - emotion_start)
            
            # End timing
            frame_end = time.time()
            frame_times.append(frame_end - frame_start)
            detection_times.append(detection_end - detection_start)
            
            # Track memory
            allocated = torch.cuda.memory_allocated()
            max_memory = max(max_memory, allocated)
            max_memory_pressure = max(max_memory_pressure, memory_manager.get_memory_pressure())
            
            # Delay to allow memory pressure to build
            if _ % 10 == 0:
                time.sleep(0.01)
        
        # Calculate metrics
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
        avg_emotion_time = sum(emotion_times) / len(emotion_times) if emotion_times else 0
        avg_face_count = sum(face_counts) / len(face_counts) if face_counts else 0
        
        # Compile results
        results = {
            "config_name": config_name,
            "avg_fps": avg_fps,
            "avg_frame_time": avg_frame_time * 1000,  # Convert to ms
            "avg_detection_time": avg_detection_time * 1000,  # Convert to ms
            "avg_emotion_time": avg_emotion_time * 1000,  # Convert to ms
            "max_memory_gb": max_memory / (1024**3),  # Convert to GB
            "max_memory_pressure": max_memory_pressure,
            "avg_face_count": avg_face_count
        }
        
        print(f"Results for {config_name}:")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Average frame time: {avg_frame_time * 1000:.2f} ms")
        print(f"  Maximum GPU memory: {max_memory / (1024**3):.2f} GB")
        print(f"  Maximum memory pressure: {max_memory_pressure * 100:.1f}%")
        
        return results
    
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None
    finally:
        # Clean up resources
        if 'cap' in locals():
            cap.release()
        if 'face_detector' in locals():
            face_detector.release()
        if 'emotion_classifier' in locals():
            emotion_classifier.release()
        if 'memory_manager' in locals():
            memory_manager.release()
        
        # Restore original config values
        for key, value in original_values.items():
            setattr(config, key, value)
        
        # Force GPU memory cleanup
        cleanup_gpu_memory()
        
        # Wait a bit to let GPU recover
        time.sleep(1.0)

def main():
    """Main function"""
    args = parse_args()
    
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
        print("This benchmark requires CUDA GPU acceleration.")
        return 1
    
    # Set up benchmark configurations
    configurations = [
        {
            "name": "Baseline",
            "config": {
                "DETECTION_INTERVAL": 1,
                "FACE_TRACKING_ENABLED": False,
                "USE_FP16": False,
                "MEMORY_OPTIMIZATION": "normal",
                "FACE_DETECTION_RESIZE_FACTOR": 1.0
            }
        },
        {
            "name": "Face Tracking",
            "config": {
                "DETECTION_INTERVAL": 3,
                "FACE_TRACKING_ENABLED": True,
                "USE_FP16": False,
                "MEMORY_OPTIMIZATION": "normal",
                "FACE_DETECTION_RESIZE_FACTOR": 1.0
            }
        },
        {
            "name": "FP16 Precision",
            "config": {
                "DETECTION_INTERVAL": 1,
                "FACE_TRACKING_ENABLED": False,
                "USE_FP16": True,
                "MEMORY_OPTIMIZATION": "normal",
                "FACE_DETECTION_RESIZE_FACTOR": 1.0
            }
        },
        {
            "name": "Memory Optimization",
            "config": {
                "DETECTION_INTERVAL": 1,
                "FACE_TRACKING_ENABLED": False,
                "USE_FP16": False,
                "MEMORY_OPTIMIZATION": "aggressive",
                "FACE_DETECTION_RESIZE_FACTOR": 1.0
            }
        },
        {
            "name": "Detection Resize",
            "config": {
                "DETECTION_INTERVAL": 1,
                "FACE_TRACKING_ENABLED": False,
                "USE_FP16": False,
                "MEMORY_OPTIMIZATION": "normal",
                "FACE_DETECTION_RESIZE_FACTOR": 0.5
            }
        },
        {
            "name": "Fully Optimized",
            "config": {
                "DETECTION_INTERVAL": 3,
                "FACE_TRACKING_ENABLED": True,
                "USE_FP16": True,
                "MEMORY_OPTIMIZATION": "aggressive",
                "FACE_DETECTION_RESIZE_FACTOR": 0.5
            }
        }
    ]
    
    # Determine video source
    if args.video and os.path.exists(args.video):
        video_source = args.video
        print(f"Using video file: {video_source}")
    else:
        video_source = args.camera
        print(f"Using camera index: {video_source}")
    
    # Run benchmarks
    results = []
    
    for config in configurations:
        result = benchmark_configuration(
            video_source=video_source,
            num_frames=args.frames,
            width=width,
            height=height,
            config_name=config["name"],
            config_dict=config["config"]
        )
        
        if result:
            results.append(result)
        
        # Short pause between configurations
        time.sleep(2.0)
    
    # Print comparative results
    if results:
        print("\n=== Benchmark Results ===\n")
        
        # Prepare table data
        table_data = []
        headers = ["Configuration", "FPS", "Frame Time (ms)", "Detection (ms)", 
                  "Emotion (ms)", "Max Mem (GB)", "Mem Pressure", "Avg Faces"]
        
        for r in results:
            table_data.append([
                r["config_name"],
                f"{r['avg_fps']:.2f}",
                f"{r['avg_frame_time']:.2f}",
                f"{r['avg_detection_time']:.2f}",
                f"{r['avg_emotion_time']:.2f}",
                f"{r['max_memory_gb']:.2f}",
                f"{r['max_memory_pressure']*100:.1f}%",
                f"{r['avg_face_count']:.1f}"
            ])
        
        # Print table
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Calculate improvement percentage
        if len(results) > 1:
            baseline = results[0]
            optimized = results[-1]
            
            fps_improvement = (optimized["avg_fps"] / baseline["avg_fps"] - 1) * 100
            memory_reduction = (1 - optimized["max_memory_gb"] / baseline["max_memory_gb"]) * 100
            
            print(f"\nImprovement Summary (Baseline vs Fully Optimized):")
            print(f"  FPS improvement: {fps_improvement:.1f}%")
            print(f"  Memory reduction: {memory_reduction:.1f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 