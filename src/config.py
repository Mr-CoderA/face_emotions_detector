"""
Configuration file for emotion detection app
"""

# GPU configuration
DEVICE = "cuda"  # Force GPU only, no CPU fallback
GPU_MEMORY_LIMIT = 5.5 * 1024 * 1024 * 1024  # ~5.5GB VRAM limit (leaving some buffer)

# Model configuration
FACE_DETECTION_CONFIDENCE = 0.9
EMOTION_MODEL_NAME = "nateraw/vit-base-ferplus"  # High-accuracy facial emotion recognition model
BATCH_SIZE = 1  # Small batch size to conserve memory

# Face detection settings
DETECTION_INTERVAL = 3  # Only run face detection every N frames
MAX_FACES = 4  # Maximum number of faces to process simultaneously

# Performance optimization settings
USE_FP16 = True  # Use half-precision (FP16) for model inference
FACE_TRACKING_ENABLED = True  # Enable face tracking to avoid redundant detections
TRACKER_PERSIST_TIME = 1.0  # How long to track a face without re-detection (seconds)
MEMORY_OPTIMIZATION = "aggressive"  # Options: "normal", "aggressive", "conservative"
FACE_DETECTION_RESIZE_FACTOR = 0.5  # Resize input for face detection to save memory
THROTTLE_FPS = False  # Whether to limit FPS to avoid GPU overload
MAX_FPS = 30  # Maximum FPS to limit processing to (if throttling enabled)

# Emotions to detect - aligned with FER+ labels used by the model
EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

# Display settings
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FONT_SCALE = 0.5
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)  # Green
BOX_COLOR = (0, 255, 0)  # Green
BOX_THICKNESS = 2

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