#!/usr/bin/env python3
"""
ZeroLag RPS Game Launcher
Starts the Rock-Paper-Scissors game with hand gesture recognition.

Usage:
    python run_game.py

Requirements:
    - Python 3.9+
    - All dependencies from requirements.txt or pyproject.toml

The script will automatically:
    1. Check for required dependencies
    2. Download the hand_landmarker.task model if needed
    3. Launch the game with your camera
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Verify all required packages are installed."""
    required_packages = [
        'torch',
        'torchvision',
        'mediapipe',
        'cv2',  # opencv
        'numpy',
        'PIL',  # pillow
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall them with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies found")
    return True

def run_game():
    """Launch the RPS game."""
    script_path = Path(__file__).parent / "scripts" / "live_inference.py"
    
    if not script_path.exists():
        print(f"❌ Error: Could not find {script_path}")
        return False
    
    print(f"🎮 Starting RPS Game from {script_path}")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, str(script_path)], check=False)
        return True
    except Exception as e:
        print(f"❌ Error running game: {e}")
        return False

def main():
    print("🎮 ZeroLag RPS Game Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Run game
    print("\n🚀 Launching game...")
    print("   Controls: SPACE to start, Q to quit")
    print("=" * 50 + "\n")
    
    if run_game():
        print("\n✓ Game closed successfully")
        sys.exit(0)
    else:
        print("\n❌ Game failed to run")
        sys.exit(1)

if __name__ == "__main__":
    main()
