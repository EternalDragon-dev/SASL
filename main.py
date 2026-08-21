#!/usr/bin/env python3
"""
SASL Fingerspelling Recognizer — manual build.
Recognises South African Sign Language (SASL) handshape letters in real time
using MediaPipe hand and pose landmark models and a rule-based classifier.

Run:
    MPLBACKEND=Agg python main.py

Keys:
    Q  quit
"""

import time
from collections import Counter, deque  # deque = fixed-length ring buffer; Counter = frequency map

import cv2                              # OpenCV: webcam capture + all drawing
import mediapipe as mp                  # top-level MediaPipe package (needed for mp.Image)
import numpy as np                      # used for palm-center averaging
from mediapipe.tasks import python as mp_tasks          # Tasks API base layer
from mediapipe.tasks.python import vision as mp_vision  # Tasks API vision models
from types import SimpleNamespace       # lightweight object with .x .y .z attributes
from typing import Optional

# SASL classification modules (joint-angle feature vector + rule-based letter classifier)
from joint_angles import compute_joint_angles
from sign_classifier import (
    build_feature_vector,
    classify_static,
    load_ml_classifier,
    classify_static_ml,
)

# ── Model paths ───────────────────────────────────────────────────────────────
# These .task files are pre-trained MediaPipe neural network models.
# The hand model provides 21 landmarks; the pose model provides 33 body landmarks.
MODEL_PATH     = "models/hand_landmarker.task"

# Number of frames kept in the voting buffer for SASL letter smoothing.
# A letter must appear in at least half of these frames before being displayed.
LETTER_HISTORY = 14

# ── MediaPipe hand landmark indices ────────────────────────────────────────────
# MediaPipe numbers the 21 hand landmarks 0–20 in a fixed pattern:
#   0 = wrist, then each finger has 4 points (MCP → PIP → DIP → tip).
#   Fingertips are always at indices 4, 8, 12, 16, 20.
FINGERTIPS = {
    4:  "THUMB",
    8:  "INDEX",
    12: "MIDDLE",
    16: "RING",
    20: "PINKY",
}

# Landmarks used to compute the palm center: wrist (0), thumb base (1),
# and the base knuckle (MCP) of each finger (5, 9, 13, 17).
PALM_INDICES = [0, 1, 5, 9, 13, 17]

# Each tuple (a, b) means: draw a line between landmark a and landmark b.
# Together these pairs trace the full skeleton of the hand.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                   # palm edge (wrist to pinky base)
]

# ── Colors (BGR format) ────────────────────────────────────────────────────────
# OpenCV uses Blue-Green-Red channel order, NOT the RGB you may be used to.
# e.g. pure red in RGB is (255, 0, 0) but in BGR it is (0, 0, 255).
COLOR_LANDMARK   = (0, 255, 255)   # yellow
COLOR_FINGERTIP  = (0, 255, 0)     # green
COLOR_CENTER     = (255, 0, 255)   # magenta
COLOR_CONNECTION = (255, 200, 0)   # cyan  (default skeleton color)
COLOR_LABEL      = (255, 255, 255) # white

# Skeleton turns this color when a SASL letter is confidently detected
COLOR_ACTIVE = (0, 200, 255)  # gold

def compute_palm_center(lms, w: int, h: int) -> tuple[int, int]:
    """
    Return the pixel (x, y) of the palm center.

    MediaPipe gives normalized coordinates in the range 0.0–1.0.
    Multiplying by the frame dimension converts them to pixel positions.
    Averaging the 6 palm landmarks gives a stable center point that
    doesn't shift when individual fingers move.
    """
    xs = [lms[i].x * w for i in PALM_INDICES]  # normalized x → pixel column
    ys = [lms[i].y * h for i in PALM_INDICES]  # normalized y → pixel row
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

def draw_hand(frame, lms, w: int, h: int, conn_color: tuple = COLOR_CONNECTION) -> None:
    """
    Draw the full hand skeleton, landmark dots, fingertip labels, and palm center.

    conn_color defaults to cyan (no letter detected) or gold (letter detected).
    OpenCV draws in-place — this function returns None and mutates the frame.
    """
    # 1. Draw bone connections
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame,
                 (int(lms[s].x * w), int(lms[s].y * h)),
                 (int(lms[e].x * w), int(lms[e].y * h)),
                 conn_color, 2, cv2.LINE_AA)  # thickness=2, anti-aliased

    # 2. Small dot at every one of the 21 landmarks
    for lm in lms:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, COLOR_LANDMARK, cv2.FILLED)

    # 3. Larger highlighted dot + text label at each fingertip
    for idx, name in FINGERTIPS.items():
        cx, cy = int(lms[idx].x * w), int(lms[idx].y * h)
        cv2.circle(frame, (cx, cy), 8, COLOR_FINGERTIP, cv2.FILLED)
        cv2.putText(frame, name, (cx + 10, cy - 10),  # offset so text doesn't cover the dot
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL, 1, cv2.LINE_AA)

    # 4. Palm center dot + label
    pcx, pcy = compute_palm_center(lms, w, h)
    cv2.circle(frame, (pcx, pcy), 10, COLOR_CENTER, cv2.FILLED)
    cv2.putText(frame, "CENTER", (pcx + 12, pcy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CENTER, 1, cv2.LINE_AA)

def main() -> None:
    """
    Entry point. Sets up the webcam and both MediaPipe models, then runs the
    frame loop until the user presses Q.

    The function is structured in three layers:
      1. Setup   — open webcam, configure models, initialise state
      2. Loop    — read frame → run models → draw → handle keys
      3. Cleanup — release webcam and close windows
    """
    # Prefer the stable Mac camera (index 1) and keep index 0 as a fallback.
    # Some Continuity Camera devices initially stream, then stop providing frames.
    def open_and_warm(indices=(1, 0), warm_reads=10, delay=0.1):
        for idx in indices:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            for _ in range(warm_reads):
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                time.sleep(delay)
            cap.release()
        return None

    cap = open_and_warm(indices=(1, 0), warm_reads=10, delay=0.1)
    if cap is None:
        print("Error: Could not open webcam (tried indices 1 and 0).")
        print("- Ensure Terminal/VS Code has Camera permission in macOS System Settings.")
        print("- Close other apps that may be using the camera (Zoom/FaceTime/Photo Booth).")
        return

    print("Hand Gesture Recognizer — press 'q' to quit.")

    # BaseOptions tells MediaPipe where to find the model file on disk.
    # HandLandmarkerOptions configures the hand model:
    #   - num_hands: track up to 2 hands simultaneously
    #   - detection_confidence: threshold to START tracking a new hand (strict)
    #   - tracking_confidence: threshold to CONTINUE tracking (more lenient)
    #   - VIDEO mode: model uses frame timestamps to improve temporal smoothing
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    start_t = None  # set on the first frame to anchor timestamps

    # Per-hand voting buffer: keeps the last LETTER_HISTORY raw classification
    # results for the left and right hand separately.
    # The live display only shows a letter once it stabilises across frames.
    letter_history: dict[str, deque] = {
        "Left":  deque(maxlen=LETTER_HISTORY),
        "Right": deque(maxlen=LETTER_HISTORY),
    }

    # `with` ensures the MediaPipe model is cleanly released when the block exits.
    # The hand landmarker holds model resources and must be closed to avoid leaks.
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:

        # Try to load an exported ML classifier (ONNX). If it exists and
        # onnxruntime is available, we'll use it for higher-coverage
        # classification. Otherwise we fall back to the rule-based classifier.
        ml_session = None
        try:
            ml_session, ml_label_names = load_ml_classifier()
            print("Loaded ML classifier: models/sign_classifier.onnx")
        except FileNotFoundError:
            ml_session, ml_label_names = None, None
            print("No trained ML model found (models/sign_classifier.onnx). Using rule-based classifier.")
        except ImportError:
            ml_session, ml_label_names = None, None
            print("onnxruntime not installed — install it to use ML classifier. Using rule-based classifier.")
        except Exception as exc:
            ml_session, ml_label_names = None, None
            print(f"Warning: failed to load ML classifier: {exc}. Using rule-based classifier.")

        failed_reads = 0
        while True:
            # ── Read frame ───────────────────────────────────────────────────
            ret, frame = cap.read()  # ret=False if webcam disconnects
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads >= 30:
                    print("Error: webcam stopped providing frames.")
                    break
                time.sleep(0.05)
                continue
            failed_reads = 0

            # Mirror the image so hand movements feel natural (like a mirror)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape  # shape is (height, width, channels) — note order

            # ── Timestamp ────────────────────────────────────────────────────
            # VIDEO mode requires a monotonically increasing millisecond timestamp
            # per frame. time.monotonic() is used because it cannot go backwards
            # (unlike time.time() which can be adjusted by the OS clock).
            t = time.monotonic()
            if start_t is None:
                start_t = t
            timestamp_ms = int((t - start_t) * 1000)

            # ── Run both models ───────────────────────────────────────────────
            # Convert BGR (OpenCV format) → RGB (MediaPipe format), then wrap
            # in mp.Image. Both models share the same image object — no duplication.
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # World landmarks are metric-scale and palm-centred — better for
            # the SASL feature vector than image landmarks. Fall back if unavailable.
            world_lms_all = getattr(results, "hand_world_landmarks", None) or []

            # ── Process each detected hand ────────────────────────────────────
            if results.hand_landmarks:
                for i, (hand_lm_raw, hand_info_list) in enumerate(
                    zip(results.hand_landmarks, results.handedness)
                ):
                    # Wrap raw MediaPipe landmarks in SimpleNamespace objects
                    # so all helper functions receive a consistent .x .y .z interface
                    lms = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z) for lm in hand_lm_raw]

                    # World landmarks for the SASL feature vector (more scale-invariant)
                    if i < len(world_lms_all):
                        wlms = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z) for lm in world_lms_all[i]]
                    else:
                        wlms = lms  # fallback to image landmarks

                    label    = hand_info_list[0].category_name  # "Left" or "Right"
                    pcx, pcy = compute_palm_center(lms, w, h)

                    # ── SASL classification ───────────────────────────────────
                    # Step 1: compute 15 joint flexion angles from landmarks
                    angles = compute_joint_angles(wlms)
                    # Step 2: build the 22-dimensional feature vector
                    fv     = build_feature_vector(wlms, angles)
                    # Step 3: classify using ML if available, otherwise use rules.
                    # `raw` is either None or a tuple (label, confidence).
                    if ml_session is not None:
                        raw = classify_static_ml((ml_session, ml_label_names), fv)
                    else:
                        raw = classify_static(fv, angles)

                    # Step 4: voting buffer — append this frame's raw result.
                    # This buffer smooths the displayed letter across consecutive frames.
                    hist = letter_history[label]
                    hist.append(raw[0] if raw else None)  # None if no match

                    # Step 5: only display a letter if it holds the majority vote
                    # This prevents single noisy frames from flashing wrong letters
                    sasl_letter: Optional[str] = None
                    sasl_conf: float = 0.0
                    votes = Counter(x for x in hist if x is not None)
                    if votes:
                        top_letter, top_n = votes.most_common(1)[0]
                        if top_n >= LETTER_HISTORY // 2:  # majority = more than half
                            sasl_letter = top_letter
                            sasl_conf   = raw[1] if raw else 0.0

                    # Skeleton turns gold when a letter is detected, cyan otherwise
                    skel_color = COLOR_ACTIVE if sasl_letter else COLOR_CONNECTION
                    draw_hand(frame, lms, w, h, skel_color)

                    if sasl_letter:
                        # Large letter above the palm
                        cv2.putText(frame, sasl_letter,
                                    (pcx - 30, pcy - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.8, COLOR_ACTIVE, 5, cv2.LINE_AA)
                        # Confidence percentage below the letter
                        cv2.putText(frame, f"{sasl_conf * 100:.0f}%",
                                    (pcx - 22, pcy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACTIVE, 2, cv2.LINE_AA)

                    cv2.putText(frame, f"{label} hand",
                                (pcx - 35, pcy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_LABEL, 2, cv2.LINE_AA)

            # ── Display and key handling ──────────────────────────────────────
            cv2.imshow("SASL Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    # Always release the webcam and destroy windows — even after a break or error
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()