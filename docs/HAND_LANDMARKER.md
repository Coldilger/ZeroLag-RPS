# 📦 About hand_landmarker.task

## What is it?

`hand_landmarker.task` is a **pre-trained hand detection model** from Google's MediaPipe library. It detects hand landmarks (21 key points per hand) in real-time video.

## Why is it needed?

The project uses **MediaPipe 0.10.35** which changed its API:

| Version | API | Model | Storage |
|---------|-----|-------|---------|
| **0.9.x** (old) | `mp.solutions.hands` | Embedded in library | Automatic |
| **0.10.35+** (new) | `mediapipe.tasks.python.vision` | Separate `.task` file | Must download |

## Where does it come from?

The model is automatically downloaded from:
```
https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task
```

**File size:** ~9.2 MB

## First-Run Behavior

When you run the game for the first time:

1. **`run_game.py`** checks for dependencies ✓
2. **`live_inference.py`** looks for `model/hand_landmarker.task`
3. If not found, automatically downloads it ⬇️
4. Caches it locally for future runs (no re-download needed)

```
Downloading hand_landmarker model to model/hand_landmarker.task...
  Trying: https://storage.googleapis.com/mediapipe-models/...
  ✗ Failed: HTTP Error 404
  Trying: https://storage.googleapis.com/mediapipe-assets/...
  ✓ Model downloaded successfully
```

## Troubleshooting

### Q: Why is my first run slow?
A: It's downloading the ~9.2 MB model. Subsequent runs use the cached file (instant).

### Q: What if download fails?
A: Manual download:
```bash
# Download the model file
wget https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task -O model/hand_landmarker.task

# Or on Windows (PowerShell):
Invoke-WebRequest https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task -OutFile model/hand_landmarker.task
```

### Q: Can I share the .task file?
A: **Not in this repo** - it's auto-downloaded on first run. This keeps the repo lightweight and ensures you always have the latest model version.

However, you *can* pre-download it when setting up a deployment:
```bash
python -c "
import urllib.request
from pathlib import Path
Path('model').mkdir(exist_ok=True)
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task',
    'model/hand_landmarker.task'
)
print('✓ Model downloaded')
"
```

## How It Works in the Code

```python
# 1. Check if model exists
model_path = "model/hand_landmarker.task"

# 2. Download if missing
if not Path(model_path).exists():
    urllib.request.urlretrieve(url, model_path)

# 3. Load for hand detection
base_opts = base_options.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_opts)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# 4. Use in game loop
detection_result = hand_landmarker.detect(mp_image)
```

## 🎯 Key Points

- ✓ **Automatic**: Downloads on first run, cached locally
- ✓ **Fast**: Uses TensorFlow Lite for real-time inference
- ✓ **Accurate**: Google's production-grade hand detection
- ✓ **Cross-platform**: Works on Windows, macOS, Linux
- ✓ **Offline**: Once downloaded, works without internet

---

**Last updated:** MediaPipe 0.10.35+ (Tasks API)
