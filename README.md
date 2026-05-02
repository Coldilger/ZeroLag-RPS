# ZeroLag RPS: Real-Time Anticipation of Hand Gestures

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Solutions-blueviolet)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

> **Final Project for Deep Learning for Computer Vision (20600)**
> *Bocconi University, MSc in Data Science*

## 📜 Abstract
Real-time gesture recognition systems often suffer from latency due to the classification of completed motions. **ZeroLag RPS** is a model designed to defeat a human opponent in Rock-Paper-Scissors by predicting moves *before* gesture completion.

We developed a hybrid architecture combining a **ResNet-18 spatial encoder** with a **Temporal Convolutional Network (TCN) head**, trained on a custom dataset of gesture sequences. The TCN's dilated causal convolutions efficiently model temporal dependencies within a 64-frame window, enabling the detection of subtle "wind-up" micromovements. Deployed in a live environment, the system delivers zero-latency counter-moves for a seamless user experience.
## 🎥 Demo
![System Demo](docs/rps_s.gif)

*The system predicts "Scissors" (and plays "Rock") while the hand is still in the wind-up phase.*

## 🧠 Model Architecture

Our approach solves the **"Rock Paradox"** (where all moves start looking like Rock) using a specialized pipeline:

1.  **Spatial Focus:** **MediaPipe Hands** extracts skeletal crops to ensure translation invariance and remove background noise.
2.  **Feature Extraction:** A **ResNet-18** encoder converts frames into feature vectors.
3.  **Temporal Modeling:** A **TCN (Temporal Convolutional Network)** processes a rolling buffer of 64 frames. We chose TCNs over LSTMs to enable parallelization and precise receptive field engineering.
4.  **Optimization:** Trained using a **Sigmoid Time-Weighted Loss** to penalize early-game ambiguity less than late-game precision.

## 📂 Project Structure

```text
ZeroLag-RPS/
├── data/                  # Dataset placeholder (download link below)
├── docs/                  # Documentation, Report, and Slides
│   ├── Project_CV_report.pdf
│   └── Presentation.pptx
├── models/                # Trained model weights (.pth)
├── notebooks/             # Experiments and drafts
│   ├── TCN_training.ipynb
│   └── data_analysis.ipynb
├── scripts/               # Source code
│   ├── live_inference.py  # Main script for the live battle
│   ├── preprocessing.py   # Hand crop and normalization logic
│   └── utils.py           # Helper functions
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Coldilger/ZeroLag-RPS.git
cd ZeroLag-RPS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Game

**Option A (Recommended):**
```bash
python run_game.py
```
This launcher automatically checks dependencies and downloads the MediaPipe hand detection model if needed.

**Option B (Direct):**
```bash
python scripts/live_inference.py
```

### ⚙️ First Run Setup
On first launch, the game will automatically:
1. ✓ Download `hand_landmarker.task` model (~9.2 MB) from Google's MediaPipe storage
2. ✓ Load the trained RPS classifier (`model/rps_tcn_model.pth`)
3. ✓ Initialize your webcam

The `hand_landmarker.task` file is stored in `model/` and is required for MediaPipe 0.10.35+.

📖 **Want to know more?** See [HAND_LANDMARKER.md](docs/HAND_LANDMARKER.md) for technical details about the hand detection model.

### 🎮 In-Game Controls
- **SPACE** - Start a new round
- **Q** - Quit the game

## 📊 Dataset
The model was trained on a custom dataset comprising ~5,300 frames of Rock, Paper, and Scissors gestures recorded at 20fps.
- Download the dataset [here](https://drive.google.com/drive/folders/1yZMYdbUkMhnvQazs3OVpYUC7R1Ky9l4b?usp=sharing)

## 👥 Contributors
- Ilia Koldyshev
- Gaia Iori
- Parsa Bakhtiari
