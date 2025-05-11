@echo off
echo Testing face emotion imports...
echo ===========================
echo.

call conda activate face_emotion_env || call activate face_emotion_env
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate conda environment. Please make sure face_emotion_env exists.
    exit /b 1
)

python test_imports.py

echo.
call conda deactivate || call deactivate 