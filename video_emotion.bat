@echo off
echo GPU-Optimized Face Emotion Detection (Video Processing Mode)
echo =========================================================
echo.

if "%~1"=="" (
    echo ERROR: No video file specified.
    echo Usage: video_emotion.bat path\to\video.mp4 [options]
    echo.
    echo Available options:
    echo --preview          : Show preview during processing
    echo --show_landmarks   : Display facial landmarks
    echo --speedup 2.0      : Process at 2x speed
    echo --resolution 640x480 : Set output resolution
    echo --optimization [normal^|aggressive^|conservative] : Set memory optimization level
    echo.
    goto :eof
)

REM Activate the virtual environment if it exists
if exist face_emotion_env\Scripts\activate.bat (
    call face_emotion_env\Scripts\activate.bat
)

REM Run with optimized settings
python video_emotion_app.py %* --optimization aggressive

REM Deactivate the virtual environment
if exist face_emotion_env\Scripts\deactivate.bat (
    call face_emotion_env\Scripts\deactivate.bat
) 