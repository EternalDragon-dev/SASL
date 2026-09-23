#!/usr/bin/env python3
"""Record ordered hand trajectories for SASL motion-letter calibration.

Usage:
    MPLBACKEND=Agg python motion_calibrate.py --letter J

Keys:
    SPACE  start or stop the current trajectory
    Q      quit
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "models/hand_landmarker.task"
MOTION_DIR = Path("motion_data")
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture ordered motion-letter trajectories.")
    parser.add_argument("--letter", required=True, help="Motion letter to record, such as H, J, P, Q, or Z.")
    return parser.parse_args()


def palm_center(lms: list[SimpleNamespace]) -> tuple[float, float]:
    indices = (0, 1, 5, 9, 13, 17)
    return (
        sum(lms[index].x for index in indices) / len(indices),
        sum(lms[index].y for index in indices) / len(indices),
    )


def hand_scale(lms: list[SimpleNamespace]) -> float:
    dx = lms[0].x - lms[9].x
    dy = lms[0].y - lms[9].y
    dz = lms[0].z - lms[9].z
    return max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-6)


def draw_hand(frame, lms: list[SimpleNamespace]) -> None:
    height, width = frame.shape[:2]
    for start, end in CONNECTIONS:
        cv2.line(
            frame,
            (int(lms[start].x * width), int(lms[start].y * height)),
            (int(lms[end].x * width), int(lms[end].y * height)),
            (0, 200, 255), 2, cv2.LINE_AA,
        )
    for lm in lms:
        cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 4, (0, 255, 255), cv2.FILLED)


def main() -> None:
    args = parse_args()
    letter = args.letter.strip().upper()
    if len(letter) != 1 or not letter.isalpha():
        raise SystemExit("--letter must be one alphabetic character")

    output_dir = MOTION_DIR / letter
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_number = len(list(output_dir.glob("sample_*.csv"))) + 1

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    recording = False
    trajectory: list[tuple[float, float, float, float, float, float, float, float]] = []
    status = "Press SPACE to start a motion sample. Q quits."
    start_time = time.monotonic()
    failed_reads = 0

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads >= 30:
                    break
                continue
            failed_reads = 0
            frame = cv2.flip(frame, 1)
            timestamp = time.monotonic()
            elapsed_ms = int((timestamp - start_time) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), elapsed_ms
            )

            if results.hand_landmarks:
                lms = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z) for lm in results.hand_landmarks[0]]
                draw_hand(frame, lms)
                if recording:
                    cx, cy = palm_center(lms)
                    scale = hand_scale(lms)
                    index = lms[8]
                    trajectory.append((timestamp, cx, cy, index.x, index.y, index.z, scale, len(trajectory)))

            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (width, 35), (10, 10, 10), cv2.FILLED)
            cv2.putText(frame, f"Motion letter: {letter}  Samples: {sample_number - 1}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, status, (8, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.imshow("SASL Motion Calibrator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" "):
                if not recording:
                    recording = True
                    trajectory = []
                    status = "Recording... perform the motion, then press SPACE again."
                else:
                    recording = False
                    if len(trajectory) < 3:
                        status = "Too few frames; press SPACE to retry."
                        trajectory = []
                        continue
                    output_path = output_dir / f"sample_{sample_number:03d}.csv"
                    with output_path.open("w", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(["timestamp", "palm_x", "palm_y", "index_x", "index_y", "index_z", "hand_scale", "frame_index"])
                        writer.writerows(trajectory)
                    sample_number += 1
                    trajectory = []
                    status = f"Saved {output_path}. Press SPACE for the next sample."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
