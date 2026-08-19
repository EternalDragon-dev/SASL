#!/usr/bin/env python3
"""SASL calibration script.

Records joint angle ranges and feature vectors for a single letter.
Saves samples to data/<LETTER>.csv and prints observed angle ranges
as a ready-to-paste rule block for sign_classifier.py.

Usage:
    MPLBACKEND=Agg python calibrate.py --letter A
    MPLBACKEND=Agg python calibrate.py --letter B --auto

Keys:
    SPACE    Capture one sample
    A        Toggle auto-capture (one sample every 0.5 s)
    Q        Quit and print calibration summary
"""

# Standard library imports: argument parsing, file writing, math helpers,
# timing, file paths, and a small generic object wrapper.
import argparse
import csv
import math
import time
from pathlib import Path
from types import SimpleNamespace

# Third-party imports: OpenCV for camera and drawing, MediaPipe for hand
# tracking, and NumPy for numeric arrays.
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# Project imports: compute joint angles and build the feature vector.
from joint_angles import compute_joint_angles, JOINT_ANGLE_DEFS
from sign_classifier import build_feature_vector

# Path to the local MediaPipe hand landmark model.
MODEL_PATH = "models/hand_landmarker.task"

# Directory where the collected samples will be stored.
DATA_DIR = Path("data")

# Interval between automatic sample captures when auto mode is enabled.
AUTO_INTERVAL = 0.5

# Hand skeleton edges for drawing the hand overlay.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Keys for the 15 joint angles we compute from joint_angles.py.
ANGLE_KEYS = [name for name, _, _, _ in JOINT_ANGLE_DEFS]

# Special feature vector entries beyond the 15 joint angles.
SPECIAL_FEATURES = [
    ("thumb_abduction", 15),
    ("ring_pinky_spread", 21),
]

# Drawing colors in BGR format for OpenCV.
COLOR_TEXT = (240, 240, 240)
COLOR_HIGHLIGHT = (220, 220, 100)
COLOR_SKELETON = (0, 200, 255)
COLOR_LANDMARK = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the calibrator."""
    parser = argparse.ArgumentParser(
        description="Calibrate SASL letter features and record samples."
    )
    parser.add_argument(
        "--letter",
        required=True,
        help="SASL letter to calibrate (A-Y, excluding motion letters J/Z).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically capture one sample every 0.5 seconds.",
    )
    return parser.parse_args()


def ensure_data_dir() -> None:
    """Create the data directory if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def landmark_list_to_namespace(landmarks):
    """Convert MediaPipe landmark objects to a simple .x/.y/.z namespace list."""
    return [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks]


def draw_hand(frame, lms, color=COLOR_SKELETON) -> None:
    """Draw the detected hand skeleton and landmarks onto the frame."""
    # Convert normalized landmark coordinates (0.0–1.0) into pixel coordinates.
    h, w = frame.shape[:2]
    for start, end in CONNECTIONS:
        cv2.line(
            frame,
            (int(lms[start].x * w), int(lms[start].y * h)),
            (int(lms[end].x * w), int(lms[end].y * h)),
            color,
            2,
            cv2.LINE_AA,
        )

    # Draw a small circle at each hand landmark.
    for lm in lms:
        cv2.circle(
            frame,
            (int(lm.x * w), int(lm.y * h)),
            4,
            COLOR_LANDMARK,
            cv2.FILLED,
        )


def draw_angles_overlay(frame, angles: dict, fv: np.ndarray) -> None:
    """Draw computed angles and special features as text on the frame."""
    groups = [
        ("THUMB", ["thumb_cmc", "thumb_mp", "thumb_ip"]),
        ("INDEX", ["index_mcp", "index_pip", "index_dip"]),
        ("MIDDLE", ["middle_mcp", "middle_pip", "middle_dip"]),
        ("RING", ["ring_mcp", "ring_pip", "ring_dip"]),
        ("PINKY", ["little_mcp", "little_pip", "little_dip"]),
    ]

    # Start drawing the text in the upper-left corner.
    x, y = 8, 30
    for finger, keys in groups:
        cv2.putText(frame, finger, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    COLOR_TEXT, 1, cv2.LINE_AA)
        y += 16
        for key in keys:
            val = angles.get(key)
            if val is not None:
                cv2.putText(frame, f"  {key}: {val:.1f}",
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                            COLOR_HIGHLIGHT, 1, cv2.LINE_AA)
            y += 14
        y += 4

    # Display the two special features from the feature vector.
    cv2.putText(frame, "SPECIAL", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_TEXT, 1, cv2.LINE_AA)
    y += 16
    cv2.putText(frame, f"  thumb_abd:    {fv[15]:.1f}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                COLOR_HIGHLIGHT, 1, cv2.LINE_AA)
    y += 14
    cv2.putText(frame, f"  ring_pinky:   {fv[21]:.1f}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                COLOR_HIGHLIGHT, 1, cv2.LINE_AA)


def init_ranges() -> dict[str, list[float]]:
    """Create a dictionary to track the observed minimum and maximum values."""
    ranges = {key: [math.inf, -math.inf] for key in ANGLE_KEYS}
    for key, _ in SPECIAL_FEATURES:
        ranges[key] = [math.inf, -math.inf]
    return ranges


def update_ranges(ranges: dict[str, list[float]], fv: np.ndarray) -> None:
    """Update the min/max ranges using the latest feature vector values."""
    for idx, key in enumerate(ANGLE_KEYS):
        value = float(fv[idx])
        lo, hi = ranges[key]
        ranges[key][0] = min(lo, value)
        ranges[key][1] = max(hi, value)

    for key, idx in SPECIAL_FEATURES:
        value = float(fv[idx])
        lo, hi = ranges[key]
        ranges[key][0] = min(lo, value)
        ranges[key][1] = max(hi, value)


def format_range(rng: list[float]) -> str:
    """Format a min/max range for human-readable printing."""
    lo, hi = rng
    if lo == math.inf or hi == -math.inf:
        return "N/A"
    return f"{lo:.1f}–{hi:.1f}"


def write_sample(writer, fv: np.ndarray) -> None:
    """Write one feature vector row into the CSV file."""
    writer.writerow([float(x) for x in fv.tolist()])


def print_summary(letter: str, csv_path: Path, sample_count: int,
                  ranges: dict[str, list[float]]) -> None:
    """Print a summary of the samples collected in this run."""
    print(f"\nCalibration complete for letter '{letter}'")
    print(f"Samples captured this session: {sample_count}")
    print(f"CSV file: {csv_path}")

    if sample_count == 0:
        print("No samples were captured.")
        return

    # Print the observed low/high values for each feature.
    print("\nObserved feature ranges:")
    for key in ANGLE_KEYS:
        print(f"  {key}: {format_range(ranges[key])}")
    for key, _ in SPECIAL_FEATURES:
        print(f"  {key}: {format_range(ranges[key])}")

    # Print a block that can be copied into sign_classifier.py to bootstrap
    # the rule ranges for this letter.
    print("\nPaste this block into sign_classifier.py _RULES:")
    print(f'    "{letter}": {{')
    for key in ANGLE_KEYS:
        lo, hi = ranges[key]
        print(f'        "{key}": ({lo:.1f}, {hi:.1f}),')
    for key, _ in SPECIAL_FEATURES:
        lo, hi = ranges[key]
        print(f'        "{key}": ({lo:.1f}, {hi:.1f}),')
    print("    },")


def main() -> None:
    """Main program flow: parse args, open webcam, run capture loop, clean up."""
    args = parse_args()
    letter = args.letter.strip().upper()
    auto_capture = args.auto

    ensure_data_dir()
    output_path = DATA_DIR / f"{letter}.csv"

    # If auto mode is enabled, schedule the first automatic capture.
    next_auto_time = time.monotonic() + AUTO_INTERVAL if auto_capture else float("inf")

    # Open the default camera (index 0). Allow a short warm-up period as some
    # cameras may return an empty frame immediately after opening.
    def open_and_warm(indices=(0, 1), warm_reads=10, delay=0.1):
        for idx in indices:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            # Try a few reads to let the camera initialize.
            for _ in range(warm_reads):
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                time.sleep(delay)
            # No good frame; release and try next index.
            cap.release()
        return None

    cap = open_and_warm(indices=(0, 1), warm_reads=10, delay=0.1)
    if cap is None:
        print("Error: could not open webcam (tried indices 0 and 1).")
        print("- Ensure the Camera permission is granted to Terminal/VS Code in macOS System Settings.")
        print("- Close other apps that may be using the camera (Zoom, FaceTime, Photo Booth).")
        return

    # Configure MediaPipe's hand detector.
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    sample_count = 0
    ranges = init_ranges()
    status_message = "Press SPACE to capture, A to toggle auto, Q to quit."

    # Use a single with-block so both the model and CSV file are closed cleanly.
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker, \
         open(output_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        start_time = None

        while True:
            # Read the next frame from the webcam.
            ret, frame = cap.read()
            if not ret:
                print("Error: webcam frame not read.")
                break

            # Mirror the frame so it feels like looking in a mirror.
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # MediaPipe video mode needs a monotonically increasing timestamp.
            timestamp = time.monotonic()
            if start_time is None:
                start_time = timestamp
            elapsed_ms = int((timestamp - start_time) * 1000)

            # Convert captured frame from OpenCV BGR to MediaPipe RGB format.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect the hand and return landmarks for this frame.
            results = landmarker.detect_for_video(mp_image, elapsed_ms)

            hand_detected = False
            capture_on_frame = False

            # Use the first detected hand only. If world landmarks are available,
            # prefer them because they are metric-scale and more stable for
            # feature extraction.
            if results.hand_landmarks:
                hand_detected = True
                raw_landmarks = results.hand_landmarks[0]
                world_landmarks = getattr(results, "hand_world_landmarks", None)
                lms = landmark_list_to_namespace(raw_landmarks)

                if world_landmarks and len(world_landmarks) > 0:
                    wlms = landmark_list_to_namespace(world_landmarks[0])
                else:
                    wlms = lms

                # Compute the 15 joint angles and the full 22D feature vector.
                angles = compute_joint_angles(wlms)
                fv = build_feature_vector(wlms, angles)

                # Draw the hand landmarks and the angle overlay for debugging.
                draw_hand(frame, lms)
                draw_angles_overlay(frame, angles, fv)

                # If auto mode is active and the interval has passed, capture.
                if auto_capture and timestamp >= next_auto_time:
                    write_sample(writer, fv)
                    sample_count += 1
                    update_ranges(ranges, fv)
                    next_auto_time = timestamp + AUTO_INTERVAL
                    status_message = f"Captured sample #{sample_count} (auto)."
                    capture_on_frame = True

            if not hand_detected:
                status_message = "No hand detected. Please place your hand in view."

            # Draw status text across the top and bottom of the frame.
            cv2.rectangle(frame, (0, 0), (w, 28), (10, 10, 10), cv2.FILLED)
            cv2.putText(frame, f"Letter: {letter}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Samples: {sample_count}", (220, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(frame, status_message, (8, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(frame, "AUTO ON" if auto_capture else "AUTO OFF",
                        (w - 120, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COLOR_HIGHLIGHT, 1, cv2.LINE_AA)

            cv2.imshow("SASL Calibrator", frame)

            # Read keyboard input and respond to keys.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            if key == ord(" ") and hand_detected:
                # Manual capture: append one sample when the user presses SPACE.
                write_sample(writer, fv)
                sample_count += 1
                update_ranges(ranges, fv)
                status_message = f"Captured sample #{sample_count}."
            if key == ord("a") or key == ord("A"):
                # Toggle auto-capture mode on/off.
                auto_capture = not auto_capture
                if auto_capture:
                    next_auto_time = time.monotonic() + AUTO_INTERVAL
                status_message = "Auto-capture ON." if auto_capture else "Auto-capture OFF."

            if capture_on_frame:
                status_message = f"Captured sample #{sample_count} (auto)."

    cap.release()
    cv2.destroyAllWindows()
    print_summary(letter, output_path, sample_count, ranges)


if __name__ == "__main__":
    main()
