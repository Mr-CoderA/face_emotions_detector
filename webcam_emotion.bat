@echo off
echo GPU-Optimized Face Emotion Detection (Camera Mode)
echo =================================================
echo Use local webcam or RTSP IP camera for real-time emotion detection
echo.

echo Activating conda environment 'face_emotion_env'...
call conda activate face_emotion_env || call activate face_emotion_env
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate conda environment. Please make sure face_emotion_env exists.
    exit /b 1
)

echo Controls:
echo - Press 'q' or 'ESC' to exit
echo - Press 's' to change display style
echo - Press 'f' for fullscreen
echo - Press 't' to toggle FPS limiting
echo - Press 'r' to reconnect RTSP stream (if using RTSP)
echo - Press '+'/'-' to adjust brightness
echo - Press '['/']' to adjust contrast
echo - Press 'b' to toggle blur (noise reduction)
echo - Press 'p' to toggle performance mode
echo.

REM Use the first parameter as RTSP URL if provided
IF [%1] == [] (
    echo Using local webcam (default)
    python webcam_emotion_app.py --camera 0 --flip --display_style simple --throttle_fps --optimization aggressive --rtsp_reconnect --contrast 1.2 --brightness 0.2 --blur 0 --performance_mode --max_fps 15
) ELSE (
    echo Using RTSP camera: %*
    python webcam_emotion_app.py --rtsp %* --display_style simple --throttle_fps --optimization aggressive --rtsp_reconnect --contrast 1.3 --brightness 0.25 --rtsp_buffer 0 --max_fps 15 --rtsp_hw_acceleration
)

echo.
echo Deactivating conda environment...
call conda deactivate || call deactivate
