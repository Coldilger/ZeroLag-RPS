"""
🎮 ZeroLag Rock-Paper-Scissors with Hand Gesture Recognition
Real-time inference game using MediaPipe hand detection and PyTorch TCN model.

HOW TO RUN:
-----------
Option 1 (Recommended - from project root):
    python run_game.py

Option 2 (From anywhere if dependencies installed):
    python -m scripts.live_inference

Option 3 (Direct):
    python scripts/live_inference.py

REQUIREMENTS:
-----------
- Python 3.9+
- MediaPipe 0.10.35 (with hand_landmarker.task model)
- PyTorch 2.1+
- OpenCV, NumPy, Pillow

AUTO-SETUP:
----------
The script automatically:
1. Detects your webcam
2. Downloads hand_landmarker.task model (9.2 MB) if missing
3. Loads the trained TCN model

GAME CONTROLS:
-------------
SPACE - Start new round
Q - Quit game

MODEL FILES:
-----------
- model/rps_tcn_model.pth : Trained gesture classifier
- model/hand_landmarker.task : MediaPipe hand detection model (auto-downloaded)
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import time
import os
import math
import threading
import mediapipe as mp
from mediapipe.tasks.python.vision import hand_landmarker as mp_hand
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from collections import deque
from torchvision import transforms
from PIL import Image
from pathlib import Path
import urllib.request

# ========================================
# Version of the file: live_battle_10f.py
# ========================================


# --- CONFIGURATION ---
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
TCN_MODEL_PATH = repo_root / "model" / "rps_tcn_model.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN = 64
FRAME_SIZE = 224
FEATURE_DIM = 128
CHANNELS = [64, 64, 64, 64]
KERNEL_SIZE = 6
PAD = 50 

CLASS_MAP = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
WINNING_MOVE = {'Rock': 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}

# --- SOUND SYSTEM ---
def play_sound_worker(type):
    try:
        import winsound
        if type == "countdown": winsound.Beep(1000, 100)
        if type == "win": winsound.Beep(1500, 300)
        if type == "cheat": winsound.Beep(300, 400)
    except ImportError:
        pass

def play_sound_async(type):
    t = threading.Thread(target=play_sound_worker, args=(type,), daemon=True)
    t.start()

# --- MODEL DEFINITIONS ---
class ResNetFrameEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None) 
        modules = list(resnet.children())[:-1] 
        self.backbone = nn.Sequential(*modules)
        self.proj = nn.Linear(512, feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(1)
        return self.proj(features)

class GestureTCNHead(nn.Module):
    def __init__(self, feature_dim=128, num_classes=3):
        super().__init__()
        from pytorch_tcn import TCN
        self.tcn = TCN(
            num_inputs=feature_dim, 
            num_channels=CHANNELS, 
            kernel_size=KERNEL_SIZE,
            dropout=0.2, 
            causal=True, 
            input_shape='NCL' 
        )
        self.classifier = nn.Linear(CHANNELS[-1], num_classes)

    def forward(self, features):
        features = features.permute(0, 2, 1)
        tcn_out = self.tcn(features)
        last_out = tcn_out[:, :, -1]
        return self.classifier(last_out)

class UnifiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetFrameEncoder(feature_dim=FEATURE_DIM)
        self.decoder = GestureTCNHead(feature_dim=FEATURE_DIM, num_classes=3)

# --- UTILS ---
transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
])

# --- UI DRAWING HELPERS ---
def draw_text_with_outline(img, text, pos, font_scale, color, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+3)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_centered_text(img, text, font_scale=1.0, color=(255,255,255), thickness=2):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    draw_text_with_outline(img, text, (text_x, text_y), font_scale, color, thickness)

def draw_result_ui(img, bot_move, outcome_text):
    h, w, _ = img.shape
    line1_prefix = "BOT: "
    line1_move = bot_move.upper()
    line2 = outcome_text
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 3
    
    w_prefix = cv2.getTextSize(line1_prefix, font, scale, thick)[0][0]
    
    center_x = w // 2
    center_y = h // 2
    
    y_line1 = center_y - 20
    y_line2 = center_y + 40
    
    # Calculate Line 1 Total Width to center it
    w_move = cv2.getTextSize(line1_move, font, scale, thick)[0][0]
    w_total_line1 = w_prefix + w_move
    x_line1 = center_x - (w_total_line1 // 2)
    
    # Calculate Line 2 Width
    w_line2 = cv2.getTextSize(line2, font, scale, thick)[0][0]
    x_line2 = center_x - (w_line2 // 2)
    
    # Draw Line 1
    draw_text_with_outline(img, line1_prefix, (x_line1, y_line1), scale, (200, 200, 200), thick)
    draw_text_with_outline(img, line1_move, (x_line1 + w_prefix, y_line1), scale, (0, 255, 255), thick)
    
    # Draw Line 2
    draw_text_with_outline(img, line2, (x_line2, y_line2), scale + 0.2, (0, 0, 255), thick)

# --- GEOMETRIC HEURISTICS ---
def get_dist(landmarks, idx1, idx2):
    p1 = landmarks.landmark[idx1]
    p2 = landmarks.landmark[idx2]
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def is_fist(landmarks):
    wrist = 0
    middle_closed = get_dist(landmarks, 12, wrist) < get_dist(landmarks, 9, wrist)
    ring_closed   = get_dist(landmarks, 16, wrist) < get_dist(landmarks, 13, wrist)
    pinky_closed  = get_dist(landmarks, 20, wrist) < get_dist(landmarks, 17, wrist)
    return (middle_closed and ring_closed and pinky_closed)

def is_paper(landmarks):
    wrist = 0
    index_open  = get_dist(landmarks, 8, wrist) > get_dist(landmarks, 5, wrist)
    middle_open = get_dist(landmarks, 12, wrist) > get_dist(landmarks, 9, wrist)
    ring_open   = get_dist(landmarks, 16, wrist) > get_dist(landmarks, 13, wrist)
    pinky_open  = get_dist(landmarks, 20, wrist) > get_dist(landmarks, 17, wrist)
    return (index_open and middle_open and ring_open and pinky_open)

def get_current_gesture(landmarks):
    if is_fist(landmarks): return 'Rock'
    if is_paper(landmarks): return 'Paper'
    wrist = 0
    idx_open = get_dist(landmarks, 8, wrist) > get_dist(landmarks, 5, wrist)
    mid_open = get_dist(landmarks, 12, wrist) > get_dist(landmarks, 9, wrist)
    ring_closed = get_dist(landmarks, 16, wrist) < get_dist(landmarks, 13, wrist)
    if idx_open and mid_open and ring_closed: return 'Scissors'
    return None

def run_game():
    print(f"Loading AI Brain...")
    if not TCN_MODEL_PATH.exists():
        print(f"Error loading model: file not found at '{TCN_MODEL_PATH}'")
        return

    full_model = UnifiedModel().to(DEVICE)
    try:
        checkpoint = torch.load(TCN_MODEL_PATH, map_location=DEVICE)
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("tcn.") or key.startswith("classifier."):
                new_key = f"decoder.{key}"
            else:
                new_key = key
            new_state_dict[new_key] = value
        full_model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    full_model.eval()
    encoder = full_model.encoder.to(DEVICE)
    decoder = full_model.decoder.to(DEVICE)
    
    print("Initializing MediaPipe (Tasks API)...")

    # Prepare path for the MediaPipe hand landmarker task model
    model_path = repo_root / "model" / "hand_landmarker.task"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try multiple URLs for downloading the model
    download_urls = [
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
    ]
    
    if not model_path.exists():
        print(f"Downloading hand_landmarker model to {model_path}...")
        downloaded = False
        for url in download_urls:
            try:
                print(f"  Trying: {url}")
                urllib.request.urlretrieve(url, str(model_path))
                print("  ✓ Model downloaded successfully")
                downloaded = True
                break
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
        
        if not downloaded:
            print("ERROR: Could not download hand_landmarker model from any source.")
            print("Please download manually from:")
            print("  https://developers.google.com/mediapipe/solutions/vision/hand_landmarker")
            print(f"And place it at: {model_path}")
            return

    # Create HandLandmarker using Tasks API
    try:
        print(f"Loading hand_landmarker from {model_path}...")
        base_opts = base_options.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(base_options=base_opts)
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
        print("✓ HandLandmarker initialized successfully")
    except Exception as e:
        print(f"ERROR initializing HandLandmarker: {e}")
        print(f"Traceback: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return
    
    cap = cv2.VideoCapture(0)
    
    feature_buffer = deque(maxlen=SEQ_LEN)
    last_features = None 
    last_landmarks = None 
    
    state = "IDLE" 
    start_time = 0
    
    predicted_user_move = ""
    computer_move = ""
    outcome_text = ""
    result_confidence = 0.0
    
    sound_played = {"1": False, "2": False, "win": False}
    
    print("Ready. Press 'SPACE' to start. 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h_img, w_img, _ = frame.shape
        
        # --- 1. PROCESSING ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert frame to MediaPipe Image and run detection with Tasks API
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            detection_result = hand_landmarker.detect(mp_img)
        except Exception as e:
            print(f"Detection error: {e}")
            detection_result = None
        
        # Wrap detection_result to mimic old `results.multi_hand_landmarks` structure for compatibility
        class ResultsWrapper:
            def __init__(self, hand_landmarks):
                # detection_result.hand_landmarks is already a list of NormalizedLandmarkList objects
                self.multi_hand_landmarks = hand_landmarks if hand_landmarks else []

        results = ResultsWrapper(detection_result.hand_landmarks if detection_result else None)
        
        current_features = None
        current_gesture_live = None 
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            last_landmarks = hand_landmarks
            current_gesture_live = get_current_gesture(hand_landmarks)
            
            x_list = [lm.x * w_img for lm in hand_landmarks.landmark]
            y_list = [lm.y * h_img for lm in hand_landmarks.landmark]
            
            cx = (min(x_list) + max(x_list)) / 2
            cy = (min(y_list) + max(y_list)) / 2
            box_size = max(max(x_list)-min(x_list), max(y_list)-min(y_list)) + PAD
            
            x1 = max(0, int(cx - box_size/2))
            y1 = max(0, int(cy - box_size/2))
            x2 = min(w_img, int(cx + box_size/2))
            y2 = min(h_img, int(cy + box_size/2))
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    tensor_img = transform(pil_img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        current_features = encoder(tensor_img).cpu().numpy().flatten()
                        last_features = current_features
        elif last_features is not None:
            current_features = last_features

        if current_features is not None:
            feature_buffer.append(current_features)
        
        # --- 2. GAME LOGIC ---
        if state == "IDLE":
            draw_centered_text(frame, "PRESS SPACE TO START", font_scale=1.5, thickness=3)
            
        elif state == "COUNTDOWN":
            elapsed = time.time() - start_time
            
            # --- UPDATED TIMING (1.6s intervals) ---
            if elapsed < 1.6: 
                draw_centered_text(frame, "1...", font_scale=3.0)
                if not sound_played["1"]: 
                    play_sound_async("countdown")
                    sound_played["1"] = True
            elif elapsed < 3.2: 
                draw_centered_text(frame, "2...", font_scale=3.0)
                if not sound_played["2"]: 
                    play_sound_async("countdown")
                    sound_played["2"] = True
            
            # TRIGGER PREDICTION (at 3.0s, 200ms before 'Go' would happen)
            if elapsed > 3.0: 
                state = "PREDICT" 
                
        elif state == "PREDICT":
            while len(feature_buffer) < SEQ_LEN and len(feature_buffer) > 0:
                feature_buffer.append(feature_buffer[-1])

            if len(feature_buffer) == SEQ_LEN:
                seq_features = torch.tensor(np.array(feature_buffer), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = decoder(seq_features)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    
                    pred_idx = pred_idx.item()
                    conf_val = conf.item()
                    
                    # --- UPDATED OVERRIDES ---
                    if last_landmarks:
                        # Only enforcing Rock. Removed Paper override as requested.
                        if is_fist(last_landmarks): 
                            pred_idx = 0 # Rock
                            conf_val = 1.0
                    
                    predicted_user_move = CLASS_MAP[pred_idx]
                    computer_move = WINNING_MOVE[predicted_user_move]
                    outcome_text = "YOU LOSE! :)"
                    result_confidence = conf_val
                    
                    state = "RESULT"
                    start_time = time.time()
                    if not sound_played["win"]:
                        play_sound_async("win")
                        sound_played["win"] = True
            else:
                draw_centered_text(frame, "Wait...", font_scale=1.0)

        elif state == "RESULT":
            is_cheating = False
            if current_gesture_live and predicted_user_move:
                if current_gesture_live != predicted_user_move:
                    if (time.time() - start_time > 1.0): 
                        draw_centered_text(frame, "CHEAT DETECTED!", font_scale=1.5, color=(0,0,255))
                        play_sound_async("cheat")
                        is_cheating = True
            
            if not is_cheating:
                draw_result_ui(frame, computer_move, outcome_text)
                # Show Confidence in Corner
                draw_text_with_outline(frame, f"Conf: {result_confidence:.0%}", (20, h_img - 20), 0.7, (200, 200, 200), thickness=1)
            
            # --- FASTER RESET (2.0s) ---
            if time.time() - start_time > 2.0: 
                state = "IDLE"
                feature_buffer.clear()
                sound_played = {"1": False, "2": False, "win": False}

        cv2.imshow("Magic RPS", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord(' ') and state == "IDLE":
            state = "COUNTDOWN"
            feature_buffer.clear()
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()