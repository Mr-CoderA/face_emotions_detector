#!/usr/bin/env python
"""
Real-time Webcam/RTSP Facial Emotion Detection

Streamlined application that detects faces using MediaPipe and performs
emotion recognition using a Vision Transformer model from Hugging Face.
Supports both local webcam and RTSP IP camera streams.
"""
import os
import sys
import cv2
import time
import argparse
import numpy as np
import torch
from typing import List, Dict, Any

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import application modules
from src.utils.gpu_check import check_gpu_availability, cleanup_gpu_memory, log_memory_usage
from src.utils.memory_manager import GPUMemoryManager
from src.models.face_detector import FaceDetector
from src.models.emotion_classifier import EmotionClassifier
import src.config as config

# Emotion colors (BGR format)
EMOTION_COLORS = {
    'neutral': (255, 255, 255),    # White
    'happiness': (0, 255, 255),    # Yellow
    'surprise': (0, 165, 255),     # Orange
    'sadness': (255, 0, 0),        # Blue
    'anger': (0, 0, 255),          # Red
    'disgust': (0, 128, 0),        # Green
    'fear': (255, 0, 255),         # Magenta
    'contempt': (128, 0, 128)      # Purple
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Webcam/RTSP Facial Emotion Detection")
    
    # Use a mutually exclusive group for input source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--camera", 
        type=int, 
        default=0, 
        help="Local camera index (default: 0)"
    )
    source_group.add_argument(
        "--rtsp", 
        type=str, 
        help="RTSP camera URL (e.g., rtsp://username:password@192.168.1.100:554/stream)"
    )
    
    parser.add_argument(
        "--resolution", 
        type=str, 
        default="640x480", 
        help="Camera resolution (default: 640x480)"
    )
    
    parser.add_argument(
        "--show_landmarks", 
        action="store_true", 
        help="Show facial landmarks"
    )
    
    parser.add_argument(
        "--flip", 
        action="store_true", 
        help="Flip the camera horizontally (selfie mode)"
    )
    
    parser.add_argument(
        "--display_style", 
        type=str, 
        choices=["simple", "detailed", "minimal"], 
        default="detailed",
        help="Display style: simple, detailed, or minimal"
    )
    
    parser.add_argument(
        "--fullscreen", 
        action="store_true", 
        help="Run in fullscreen mode"
    )
    
    parser.add_argument(
        "--throttle_fps",
        action="store_true",
        help="Throttle FPS to prevent GPU overload"
    )
    
    parser.add_argument(
        "--max_fps",
        type=int,
        default=config.MAX_FPS,
        help=f"Maximum FPS limit (default: {config.MAX_FPS})"
    )
    
    parser.add_argument(
        "--optimization",
        type=str,
        choices=["normal", "aggressive", "conservative"],
        default=config.MEMORY_OPTIMIZATION,
        help="Memory optimization level"
    )
    
    # RTSP-specific settings
    parser.add_argument(
        "--rtsp_buffer",
        type=int,
        default=1,
        help="RTSP buffer size (default: 1, use 0 for no buffering)"
    )
    
    parser.add_argument(
        "--rtsp_reconnect",
        action="store_true",
        help="Enable automatic reconnection to RTSP stream"
    )
    
    parser.add_argument(
        "--rtsp_reconnect_interval",
        type=int,
        default=5,
        help="Seconds to wait before reconnection attempts (default: 5)"
    )
    
    # Video quality improvement settings
    parser.add_argument(
        "--brightness",
        type=float,
        default=0.0,
        help="Adjust brightness: -1.0 to 1.0 (default: 0.0)"
    )
    
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Adjust contrast: 0.0 to 3.0 (default: 1.0)"
    )
    
    parser.add_argument(
        "--blur",
        type=int,
        default=0,
        help="Apply gaussian blur to reduce noise (0 = disabled, odd values like 3, 5, etc.)"
    )
    
    parser.add_argument(
        "--performance_mode",
        action="store_true",
        help="Enable performance mode (lower resolution processing for better FPS)"
    )
    
    parser.add_argument(
        "--rtsp_hw_acceleration",
        action="store_true",
        help="Use hardware acceleration for RTSP decoding when available"
    )
    
    return parser.parse_args()

def draw_emotions(frame, faces, emotions_list, display_style="detailed", show_landmarks=True):
    """Draw emotion labels and face boxes on the frame with advanced styling

    Args:
        frame: Input video frame
        faces: List of detected faces
        emotions_list: List of emotion predictions for each face
        display_style: Visualization style ("simple", "detailed", or "minimal")
        show_landmarks: Whether to show facial landmarks

    Returns:
        Frame with annotations
    """
    result_frame = frame.copy()
    
    # Background overlay for information
    if display_style == "detailed":
        # Add semi-transparent overlay at the top
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (0, 0), (result_frame.shape[1], 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result_frame, 0.4, 0, result_frame)
    
    for i, (face, emotions) in enumerate(zip(faces, emotions_list)):
        if emotions is None:
            continue
            
        # Get bounding box
        x1, y1, x2, y2 = face.box
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Get top emotion
        top_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion_name, confidence = top_emotion
        
        # Get color for this emotion
        color = EMOTION_COLORS.get(emotion_name, config.BOX_COLOR)
        
        if display_style == "detailed":
            # Draw a nicer bounding box with gradient
            alpha = 0.6
            overlay = result_frame.copy()
            
            # Filled rectangle with transparency
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
            
            # Draw border
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw top-left corner lines
            corner_length = min(30, box_width // 3, box_height // 3)
            cv2.line(result_frame, (x1, y1), (x1 + corner_length, y1), color, 3)
            cv2.line(result_frame, (x1, y1), (x1, y1 + corner_length), color, 3)
            
            # Draw bottom-right corner lines
            cv2.line(result_frame, (x2, y2), (x2 - corner_length, y2), color, 3)
            cv2.line(result_frame, (x2, y2), (x2, y2 - corner_length), color, 3)
            
            # Draw emotion label with background
            label_bg_color = (30, 30, 30)  # Dark background
            label_text_color = color
            label = f"{emotion_name.upper()}: {confidence:.2f}"
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )[0]
            
            # Background rectangle for text
            cv2.rectangle(
                result_frame,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0] + 10, y1),
                label_bg_color,
                -1
            )
            
            # Text
            cv2.putText(
                result_frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                label_text_color,
                2
            )
            
            # Show all emotions as bar chart on the right
            if box_width > 100:  # Only for sufficiently large faces
                bar_height = 5
                bar_gap = 3
                bar_width = 40
                all_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                
                for j, (emo, prob) in enumerate(all_emotions):
                    # Position
                    bar_x = x2 + 5
                    bar_y = y1 + j * (bar_height + bar_gap)
                    
                    # Background bar (gray)
                    cv2.rectangle(
                        result_frame,
                        (bar_x, bar_y),
                        (bar_x + bar_width, bar_y + bar_height),
                        (80, 80, 80),
                        -1
                    )
                    
                    # Foreground bar (emotion color)
                    emo_color = EMOTION_COLORS.get(emo, config.BOX_COLOR)
                    cv2.rectangle(
                        result_frame,
                        (bar_x, bar_y),
                        (bar_x + int(bar_width * prob), bar_y + bar_height),
                        emo_color,
                        -1
                    )
        
        elif display_style == "simple":
            # Simple rectangle and text
            cv2.rectangle(
                result_frame,
                (x1, y1),
                (x2, y2),
                color,
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
                color,
                config.FONT_THICKNESS
            )
            
        elif display_style == "minimal":
            # Just colored corner brackets
            bracket_length = min(20, box_width // 4, box_height // 4)
            thickness = 2
            
            # Draw corners only
            # Top-left
            cv2.line(result_frame, (x1, y1), (x1 + bracket_length, y1), color, thickness)
            cv2.line(result_frame, (x1, y1), (x1, y1 + bracket_length), color, thickness)
            
            # Top-right
            cv2.line(result_frame, (x2, y1), (x2 - bracket_length, y1), color, thickness)
            cv2.line(result_frame, (x2, y1), (x2, y1 + bracket_length), color, thickness)
            
            # Bottom-left
            cv2.line(result_frame, (x1, y2), (x1 + bracket_length, y2), color, thickness)
            cv2.line(result_frame, (x1, y2), (x1, y2 - bracket_length), color, thickness)
            
            # Bottom-right
            cv2.line(result_frame, (x2, y2), (x2 - bracket_length, y2), color, thickness)
            cv2.line(result_frame, (x2, y2), (x2, y2 - bracket_length), color, thickness)
            
            # Small emotion indicator
            text_size = cv2.getTextSize(
                emotion_name[0:3].upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )[0]
            
            cv2.putText(
                result_frame,
                emotion_name[0:3].upper(),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
        
        # Draw facial landmarks if available and enabled
        if show_landmarks and face.landmarks:
            for landmark_name, (x, y) in face.landmarks.items():
                cv2.circle(
                    result_frame,
                    (x, y),
                    3,  # Slightly larger radius
                    (0, 0, 255) if display_style != "minimal" else (255, 255, 255),
                    -1  # Filled circle
                )
    
    # Add FPS information and other stats
    if display_style != "minimal":
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(
            result_frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Larger font
            (0, 255, 255),  # Yellow color
            2
        )
        
        # Add face count
        face_count = len([f for f in faces if f is not None])
        face_text = f"Faces: {face_count}"
        
        cv2.putText(
            result_frame,
            face_text,
            (result_frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        # Add memory pressure indicator if available
        if hasattr(memory_manager, 'get_memory_pressure'):
            memory_pressure = memory_manager.get_memory_pressure()
            memory_text = f"GPU: {memory_pressure*100:.0f}%"
            
            # Choose color based on pressure
            if memory_pressure > 0.9:
                memory_color = (0, 0, 255)  # Red
            elif memory_pressure > 0.7:
                memory_color = (0, 165, 255)  # Orange
            else:
                memory_color = (0, 255, 0)  # Green
                
            cv2.putText(
                result_frame,
                memory_text,
                (result_frame.shape[1] // 2 - 40, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                memory_color,
                2
            )
    
    return result_frame

def adjust_frame_quality(frame, args):
    """Apply quality adjustments to the frame
    
    Args:
        frame: Input video frame
        args: Command line arguments
        
    Returns:
        Adjusted frame
    """
    if frame is None:
        return None
    
    # Apply brightness and contrast adjustments
    if args.brightness != 0.0 or args.contrast != 1.0:
        # Convert to float for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply brightness adjustment
        if args.brightness != 0.0:
            frame_float = frame_float + args.brightness
            # Clip values to valid range [0, 1]
            frame_float = np.clip(frame_float, 0, 1)
        
        # Apply contrast adjustment
        if args.contrast != 1.0:
            # Apply contrast around midpoint 0.5
            frame_float = (frame_float - 0.5) * args.contrast + 0.5
            # Clip values to valid range [0, 1]
            frame_float = np.clip(frame_float, 0, 1)
        
        # Convert back to uint8
        frame = (frame_float * 255).astype(np.uint8)
    
    # Apply gaussian blur to reduce noise
    if args.blur > 0 and args.blur % 2 == 1:  # Must be odd number
        frame = cv2.GaussianBlur(frame, (args.blur, args.blur), 0)
    
    return frame

def open_camera_source(args):
    """
    Initialize and open the camera source (local webcam or RTSP).
    
    Args:
        args: Command line arguments
        
    Returns:
        OpenCV VideoCapture object
    """
    if args.rtsp:
        print(f"Opening RTSP stream: {args.rtsp}")
        
        # Configure RTSP-specific options
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        
        # Create capture object with RTSP URL
        if args.rtsp_hw_acceleration:
            # Try to use hardware acceleration when available
            print("Attempting to use hardware acceleration for RTSP decoding")
            cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
            
            # Set preferred decoder
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        else:
            cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
        
        # Set buffer size (smaller is more real-time but may drop frames)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, args.rtsp_buffer)
        
        # Don't try to set resolution for RTSP - use the stream's native resolution
    else:
        print(f"Opening local camera (index: {args.camera})")
        cap = cv2.VideoCapture(args.camera)
        
        # Parse resolution (only for local cameras)
        try:
            width, height = map(int, args.resolution.split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except ValueError:
            print(f"Invalid resolution format: {args.resolution}. Using default camera resolution.")
    
    # Check if camera opened successfully
    if not cap.isOpened():
        if args.rtsp:
            raise RuntimeError(f"Could not open RTSP stream: {args.rtsp}")
        else:
            raise RuntimeError(f"Could not open local camera (index: {args.camera})")
    
    # For RTSP, read a test frame to confirm connection
    if args.rtsp:
        ret, _ = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read from RTSP stream: {args.rtsp}")
        else:
            print(f"Successfully connected to RTSP stream")
            
            # Get actual stream resolution
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Stream resolution: {width}x{height}, {fps} FPS")
    
    return cap

def main():
    """Main function"""
    global current_fps, memory_manager
    current_fps = 0
    
    args = parse_args()
    
    # Initialize global memory manager
    memory_manager = GPUMemoryManager(optimization_level=args.optimization)
    
    # Check GPU availability
    try:
        check_gpu_availability()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("This application requires CUDA GPU acceleration.")
        return 1
    
    try:
        # Start memory monitoring
        memory_manager.start_monitoring()
        
        # Initialize models
        print(f"Initializing MediaPipe face detector...")
        face_detector = FaceDetector()
        
        print(f"Loading emotion recognition model: {config.EMOTION_MODEL_NAME}")
        emotion_classifier = EmotionClassifier(device="cuda")
        
        # Initialize camera source (webcam or RTSP)
        cap = open_camera_source(args)
        
        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {width}x{height}")
        
        # For RTSP streams, set additional optimization properties
        if args.rtsp:
            # Increase the buffer size to make decoding more robust
            cap.set(cv2.CAP_PROP_BUFFERSIZE, args.rtsp_buffer)
            
            # Try to enable thread optimization for decoding
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # If resolution is too high, force a lower resolution for better performance
            if width > 1280:
                target_width = 640
                target_height = int(height * (target_width / width))
                print(f"Scaling high-resolution RTSP stream to {target_width}x{target_height} for better performance")
        
        # FPS calculation variables
        frame_count = 0
        fps_start_time = time.time()
        processing_times = []
        
        # FPS throttling variables
        throttle_fps = args.throttle_fps or config.THROTTLE_FPS
        target_frame_time = 1.0 / args.max_fps if throttle_fps else 0
        last_frame_time = time.time()
        
        # Skip frames for better performance
        skip_frame_count = 0
        max_skip_frames = 1  # Skip every other frame by default for RTSP
        if args.rtsp:
            # Skip more frames for high-resolution streams
            if width > 1280:
                max_skip_frames = 2  # Skip 2 out of 3 frames
        
        # RTSP reconnection variables
        last_frame_time_rtsp = time.time()
        reconnect_attempted = False
        
        # Performance mode variables
        performance_frame = None
        performance_scale = 0.5 if args.performance_mode else 1.0
        
        # For tracking faces across frames to reduce jitter
        last_face_boxes = []
        
        print("Press 'q' or 'ESC' to exit")
        print("Press 's' to cycle through display styles")
        print("Press 'f' to toggle fullscreen mode")
        print("Press 't' to toggle FPS throttling")
        print("Press 'r' to reconnect to RTSP stream")
        print("Press '+'/'-' to adjust brightness")
        print("Press '['/']' to adjust contrast")
        
        # Create window
        window_name = "Facial Emotion Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        if args.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Set initial display style
        display_style = args.display_style
        
        # Dynamic quality settings
        brightness_adjust = args.brightness
        contrast_adjust = args.contrast
        blur_value = args.blur
        
        # Main loop
        while True:
            # FPS throttling - wait if needed
            if throttle_fps:
                current_time = time.time()
                elapsed_since_last = current_time - last_frame_time
                sleep_time = max(0, target_frame_time - elapsed_since_last)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            
            # Skip frames for better performance if needed
            if args.rtsp and skip_frame_count < max_skip_frames:
                skip_frame_count += 1
                continue
            else:
                skip_frame_count = 0
            
            # Check for RTSP reconnection if needed
            if not ret and args.rtsp and args.rtsp_reconnect:
                current_time = time.time()
                # Only attempt reconnection if we haven't had a successful frame for a while
                if current_time - last_frame_time_rtsp > args.rtsp_reconnect_interval and not reconnect_attempted:
                    print(f"RTSP connection lost. Attempting to reconnect...")
                    cap.release()
                    time.sleep(1)  # Short delay before reconnection
                    
                    try:
                        cap = open_camera_source(args)
                        reconnect_attempted = True
                        print("Reconnection attempt completed")
                        # Try reading again
                        ret, frame = cap.read()
                    except Exception as e:
                        print(f"Reconnection failed: {e}")
                        
                    # If still can't read, wait for next attempt
                    if not ret:
                        # Show reconnection message
                        msg_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(
                            msg_frame,
                            "RTSP Connection Lost - Reconnecting...",
                            (width // 4, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2
                        )
                        cv2.imshow(window_name, msg_frame)
                        
                        # Check for keys even during reconnection attempts
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # ESC key
                            break
                        elif key == ord('r'):  # Manual reconnect
                            reconnect_attempted = False  # Reset to allow immediate retry
                        
                        continue
                    
                # If we're still failing and already attempted reconnect, show message and wait
                if not ret:
                    msg_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(
                        msg_frame,
                        "RTSP Connection Lost - Press 'r' to retry",
                        (width // 4, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow(window_name, msg_frame)
                    
                    # Check for keys even during failure
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # ESC key
                        break
                    elif key == ord('r'):  # Manual reconnect
                        reconnect_attempted = False  # Reset to allow immediate retry
                    
                    continue
            
            # Handle frame reading failure (non-RTSP or RTSP without reconnection)
            if not ret:
                print("Error: Could not read frame from camera source")
                break
            
            # If we get here, we have a valid frame
            if args.rtsp:
                last_frame_time_rtsp = time.time()
                reconnect_attempted = False
            
            # Flip horizontally for selfie view if enabled (local webcams only)
            if args.flip and not args.rtsp:
                frame = cv2.flip(frame, 1)
            
            # Apply quality adjustments
            frame = adjust_frame_quality(frame, args)
            
            # Apply dynamic quality settings if changed
            if brightness_adjust != args.brightness or contrast_adjust != args.contrast or blur_value != args.blur:
                # Create temporary args object with current values
                temp_args = argparse.Namespace(
                    brightness=brightness_adjust,
                    contrast=contrast_adjust,
                    blur=blur_value
                )
                frame = adjust_frame_quality(frame, temp_args)
            
            # Resize larger frames for display and processing
            if args.rtsp and width > 1280:
                target_width = 640
                target_height = int(height * (target_width / width))
                display_frame = cv2.resize(frame, (target_width, target_height))
            else:
                display_frame = frame.copy()
            
            # Start measuring processing time
            start_time = time.time()
            
            # Check memory pressure before processing
            memory_pressure = memory_manager.get_memory_pressure()
            
            # Resize frame for processing if in performance mode
            if args.performance_mode:
                h, w = display_frame.shape[:2]
                performance_frame = cv2.resize(display_frame, (int(w * performance_scale), int(h * performance_scale)))
                process_frame = performance_frame
            else:
                process_frame = display_frame
            
            # Detect faces only every N frames to improve performance
            detect_this_frame = (frame_count % 3 == 0)
            
            if detect_this_frame or not last_face_boxes:
                # Detect faces
                faces = face_detector.detect_faces(process_frame)
                
                # Save face boxes for tracking
                last_face_boxes = [(face.box, face.landmarks) for face in faces]
            else:
                # Use previous face detections for better performance
                faces = []
                for box, landmarks in last_face_boxes:
                    face = face_detector.create_face_from_box(box, landmarks)
                    faces.append(face)
            
            # Extract face crops
            face_crops = []
            valid_faces = []
            
            for face in faces:
                face_crop = face.crop_from_frame(process_frame)
                if face_crop is not None:
                    face_crops.append(face_crop)
                    valid_faces.append(face)
            
            # Classify emotions
            emotions_list = []
            if face_crops:
                emotions_list = emotion_classifier.predict_batch(face_crops)
            
            # If we processed a smaller frame, scale up the face coordinates for display
            if args.performance_mode and valid_faces:
                for face in valid_faces:
                    # Scale bounding box
                    x1, y1, x2, y2 = face.box
                    face.box = (
                        int(x1 / performance_scale),
                        int(y1 / performance_scale),
                        int(x2 / performance_scale),
                        int(y2 / performance_scale)
                    )
                    
                    # Scale landmarks if present
                    if face.landmarks:
                        for key, (x, y) in face.landmarks.items():
                            face.landmarks[key] = (
                                int(x / performance_scale),
                                int(y / performance_scale)
                            )
            
            # Draw results on frame
            result_frame = draw_emotions(
                display_frame, 
                valid_faces, 
                emotions_list, 
                display_style=display_style,
                show_landmarks=args.show_landmarks
            )
            
            # Calculate processing time for this frame
            elapsed_time = time.time() - start_time
            processing_times.append(elapsed_time)
            if len(processing_times) > 30:
                processing_times.pop(0)  # Keep only the last 30 frames
            
            # Calculate FPS
            frame_count += 1
            fps_elapsed_time = time.time() - fps_start_time
            if fps_elapsed_time >= 1.0:
                current_fps = frame_count / fps_elapsed_time
                frame_count = 0
                fps_start_time = time.time()
                
                # Log memory usage every 5 seconds
                if int(time.time()) % 5 == 0:
                    log_memory_usage("During processing")
                    
                    # Print performance stats
                    if processing_times:
                        avg_time = sum(processing_times) / len(processing_times)
                        print(f"Average processing time: {avg_time:.3f}s, FPS: {current_fps:.1f}")
            
            # Display result
            cv2.imshow(window_name, result_frame)
            
            # Perform memory cleanup if needed
            memory_manager.cleanup()
            
            # Check for keys
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on 'q' or ESC
            if key == ord('q') or key == 27:
                break
                
            # Cycle through display styles on 's'
            elif key == ord('s'):
                styles = ["simple", "detailed", "minimal"]
                current_idx = styles.index(display_style)
                next_idx = (current_idx + 1) % len(styles)
                display_style = styles[next_idx]
                print(f"Switched to {display_style} display style")
                
            # Toggle fullscreen on 'f'
            elif key == ord('f'):
                is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if is_fullscreen else cv2.WINDOW_FULLSCREEN
                )
                
            # Toggle FPS throttling on 't'
            elif key == ord('t'):
                throttle_fps = not throttle_fps
                print(f"FPS throttling {'enabled' if throttle_fps else 'disabled'}")
                
            # Force RTSP reconnection on 'r'
            elif key == ord('r') and args.rtsp:
                print("Manual RTSP reconnection requested")
                cap.release()
                time.sleep(1)  # Short delay before reconnection
                try:
                    cap = open_camera_source(args)
                    print("RTSP reconnection successful")
                except Exception as e:
                    print(f"RTSP reconnection failed: {e}")
            
            # Adjust brightness
            elif key == ord('+') or key == ord('='):
                brightness_adjust = min(1.0, brightness_adjust + 0.05)
                print(f"Brightness: {brightness_adjust:.2f}")
            elif key == ord('-') or key == ord('_'):
                brightness_adjust = max(-1.0, brightness_adjust - 0.05)
                print(f"Brightness: {brightness_adjust:.2f}")
                
            # Adjust contrast
            elif key == ord(']') or key == ord('}'):
                contrast_adjust = min(3.0, contrast_adjust + 0.1)
                print(f"Contrast: {contrast_adjust:.1f}")
            elif key == ord('[') or key == ord('{'):
                contrast_adjust = max(0.1, contrast_adjust - 0.1)
                print(f"Contrast: {contrast_adjust:.1f}")
                
            # Toggle blur
            elif key == ord('b'):
                if blur_value == 0:
                    blur_value = 3
                elif blur_value == 3:
                    blur_value = 5
                else:
                    blur_value = 0
                print(f"Blur: {blur_value}")
                
            # Toggle performance mode
            elif key == ord('p'):
                args.performance_mode = not args.performance_mode
                performance_scale = 0.5 if args.performance_mode else 1.0
                print(f"Performance mode: {'enabled' if args.performance_mode else 'disabled'}")
                
            # Toggle skip frame mode
            elif key == ord('k'):
                max_skip_frames = (max_skip_frames + 1) % 4  # Cycle through 0, 1, 2, 3
                print(f"Skipping {max_skip_frames} frames")
    
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory! Please reduce the resolution or close other applications.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if 'cap' in locals():
            cap.release()
        if 'face_detector' in locals():
            face_detector.release()
        if 'emotion_classifier' in locals():
            emotion_classifier.release()
        if 'memory_manager' in locals():
            memory_manager.release()
        cv2.destroyAllWindows()
        cleanup_gpu_memory()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 