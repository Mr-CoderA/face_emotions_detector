@echo off
echo Testing Facial Emotion Recognition Model...
mkdir test_images 2>nul
echo Place test facial images in the test_images folder.
echo.
echo Running with automatic face detection (MediaPipe)...
python test_emotion_model.py --detect_faces %*
pause 