"""
Test script to check if circular import issues are resolved
"""
print("Testing imports...")

# Import the shared types
from src.models.face_types import DetectedFace
print("✓ Successfully imported DetectedFace from face_types")

# Import the tracker
from src.models.face_tracker import FaceTracker
print("✓ Successfully imported FaceTracker from face_tracker")

# Import the detector
from src.models.face_detector import FaceDetector
print("✓ Successfully imported FaceDetector from face_detector")

print("All imports successful! Circular import issue is resolved.") 