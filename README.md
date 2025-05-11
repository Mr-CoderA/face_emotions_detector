# GPU-Optimized Face Emotion Detection

Real-time facial emotion recognition application optimized specifically for NVIDIA GPUs with limited VRAM (GTX 1060 6GB).

## Features

- High-accuracy emotion detection with 8 emotions (neutral, happiness, surprise, sadness, anger, disgust, fear, contempt)
- Fast face detection using MediaPipe (CPU) with tracked faces between frames
- GPU-accelerated emotion classification using Vision Transformer model
- Efficient memory management with adaptive optimizations for limited VRAM
- Multiple visualization styles: detailed, simple, and minimal
- Support for webcam input and video file processing
- FPS throttling to prevent GPU overload
- Memory pressure monitoring with automatic recovery
- Face tracking for optimized performance

## Included Applications

This repository includes several applications:

1. **Webcam Emotion Detection** (`webcam_emotion_app.py`) - Real-time emotion detection from webcam
2. **Video File Processing** (`video_emotion_app.py`) - Process video files and save results
3. **Main Application** (`app.py`) - Original application with advanced configuration
4. **Emotion Model Testing** (`test_emotion_model.py`) - Test the emotion model on images
5. **Benchmark Tool** (`benchmark.py`) - Benchmark performance on your hardware

## Face Detection

The application uses MediaPipe Face Detection, which provides:
- Fast detection speed even on CPU
- Accurate face bounding boxes with confidence scores
- 6 facial landmarks (eyes, nose, mouth, ears)
- Support for multiple faces with consistent tracking

## Emotion Recognition Model

The application uses the `nateraw/vit-base-ferplus` Vision Transformer model from Hugging Face, which is trained on the FER+ dataset and provides high-accuracy facial emotion recognition. The model can detect 8 emotion classes:

- Neutral
- Happiness
- Surprise
- Sadness
- Anger
- Disgust
- Fear
- Contempt

## Requirements

- Python 3.7+
- NVIDIA GeForce GTX 1060 or better GPU (6GB+ VRAM)
- CUDA and cuDNN installed

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/face-emotion-detection.git
cd face-emotion-detection
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv face_emotion_env
face_emotion_env\Scripts\activate  # Windows
source face_emotion_env/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Usage

### Webcam Mode

Run the webcam application with default settings:

```bash
python webcam_emotion_app.py
```

Or use the Windows batch file:

```
webcam_emotion.bat
```

#### Webcam Options

| Option | Description |
|--------|-------------|
| `--camera INDEX` | Camera index (default: 0) |
| `--resolution WIDTHxHEIGHT` | Camera resolution (default: 640x480) |
| `--show_landmarks` | Show facial landmarks |
| `--flip` | Flip the camera horizontally (selfie mode) |
| `--display_style [simple\|detailed\|minimal]` | Choose visualization style (default: detailed) |
| `--fullscreen` | Run in fullscreen mode |
| `--throttle_fps` | Enable FPS throttling to prevent GPU overload |
| `--max_fps FPS` | Maximum FPS limit when throttling is enabled (default: 30) |
| `--optimization [normal\|aggressive\|conservative]` | Memory optimization level (default: aggressive) |

#### Interactive Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Exit the application |
| `s` | Cycle through display styles |
| `f` | Toggle fullscreen mode |
| `t` | Toggle FPS throttling |

### Video Processing Mode

Process a video file:

```bash
python video_emotion_app.py path/to/video.mp4
```

Or use the Windows batch file:

```
video_emotion.bat path/to/video.mp4
```

#### Video Processing Options

| Option | Description |
|--------|-------------|
| `--output PATH` | Output video path (default: input_filename_emotions.mp4) |
| `--resolution WIDTHxHEIGHT` | Output resolution (default: 640x480) |
| `--show_landmarks` | Show facial landmarks |
| `--preview` | Show preview window during processing |
| `--speedup FACTOR` | Speed up factor (e.g., 2.0 for double speed) |

## Performance Optimization

This application is specifically optimized for GPUs with limited VRAM, like the NVIDIA GTX 1060 (6GB). Key optimizations include:

- **Dynamic Memory Management**: Continuously monitors GPU memory usage and applies adaptive optimizations
- **FP16 Precision**: Uses half-precision (FP16) for model inference to reduce memory footprint
- **Face Tracking**: Tracks faces between frames to minimize expensive detection operations
- **Throttled Detection**: Only runs face detection every few frames to save resources
- **Memory Caching**: Caches results to avoid redundant processing
- **Adaptive Batch Sizing**: Dynamically adjusts batch size based on memory pressure
- **GPU Memory Monitoring**: Real-time memory pressure visualization with automatic cleanup

### Memory Optimization Levels

- **Conservative**: Minimizes memory cleanup operations, best for higher-end GPUs
- **Normal**: Balanced approach suitable for most systems
- **Aggressive**: Frequent memory cleanup and management, best for limited VRAM (default)

## Model Information

- **Face Detection**: MediaPipe Face Detection (runs on CPU)
- **Emotion Recognition**: Vision Transformer model (`nateraw/vit-base-ferplus`) from Hugging Face

## Troubleshooting

If you encounter issues with CUDA out-of-memory errors:

1. Try running with more aggressive memory optimization:
```bash
python webcam_emotion_app.py --optimization aggressive
```

2. Enable FPS throttling to prevent GPU overload:
```bash
python webcam_emotion_app.py --throttle_fps --max_fps 20
```

3. Reduce resolution:
```bash
python webcam_emotion_app.py --resolution 320x240
```

4. Close other GPU-intensive applications

## Testing and Benchmarking

Use the provided test and benchmark tools:

```bash
python test_emotion_model.py  # Test emotion model on sample images
python benchmark.py  # Run performance benchmark
```

Or use the Windows batch files:
```
test_emotion.bat
benchmark.bat
```

## License

[MIT License](LICENSE) "# face_emotions_detector" 
