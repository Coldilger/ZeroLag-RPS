# 🚀 Quick Start

Just want to play? Here's the 30-second setup.

## Installation

```bash
# 1. Clone
git clone https://github.com/Coldilger/ZeroLag-RPS.git
cd ZeroLag-RPS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the game
python run_game.py
```

That's it! 🎮

## What happens next?

1. **First run:** Downloads the hand detection model (~9.2 MB) — wait ~10 seconds
2. **Game window opens:** Shows your webcam with gesture recognition
3. **Press SPACE** to start a round
4. **Press Q** to quit

## Controls

| Key | Action |
|-----|--------|
| SPACE | Start new round |
| Q, ESC | Quit game |

**💡 Tip:** Make sure the game window is **in focus** (click on it) before using keyboard shortcuts.

## Troubleshooting

**"Module not found"**
→ Run: `pip install -r requirements.txt`

**"Camera not found"**
→ Make sure your webcam is plugged in and no other app is using it

**"AttributeError: module 'mediapipe'..."**
→ Make sure you're using MediaPipe 0.10.35: `pip install mediapipe==0.10.35`

**Slow first run?**
→ It's downloading the hand detection model. Next runs are fast! ⚡

## Want details?

- 📖 Full docs: [README.md](README.md)
- 🤖 Model info: [HAND_LANDMARKER.md](docs/HAND_LANDMARKER.md)
- 📊 Academic paper: [docs/](docs/)

---

**Questions?** Open an issue on GitHub!
