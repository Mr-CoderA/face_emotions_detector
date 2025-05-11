@echo off
echo GPU-Optimized Face Emotion Detection - Optimization Benchmark
echo ===========================================================
echo This benchmark will compare different optimization configurations
echo to measure their impact on performance and memory usage.
echo.

REM Activate the virtual environment if it exists
if exist face_emotion_env\Scripts\activate.bat (
    call face_emotion_env\Scripts\activate.bat
)

REM Check for video file argument
if "%~1"=="" (
    echo Using webcam for benchmark (use optimization_benchmark.bat path\to\video.mp4 for video test)
    python optimization_benchmark.py --frames 50
) else (
    echo Using video file: %1
    python optimization_benchmark.py --video %1 --frames 100
)

REM Deactivate the virtual environment
if exist face_emotion_env\Scripts\deactivate.bat (
    call face_emotion_env\Scripts\deactivate.bat
)

echo.
echo Benchmark complete. 