#!/usr/bin/env python3
"""Live demo for exported motion-letter templates.

Usage:
    python motion_demo.py

Keys:
    SPACE  start or stop one gesture recording
    Q      quit
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from motion_classifier import classify_trajectory, load_motion_templates

MODEL_PATH = "models/hand_landmarker.task"


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def palm_center(lms: list[SimpleNamespace]) -> tuple[float, float]:
    indices = (0, 1, 5, 9, 13, 17)
    return (
        sum(lms[index].x for index in indices) / len(indices),
        sum(lms[index].y for index in indices) / len(indices),
    )


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
    try:
        templates = load_motion_templates()
    except FileNotFoundError:
        print("No motion template artifact found.")
        print("Capture samples, then run: python train_motion_templates.py")
        return
    except (KeyError, ValueError) as exc:
        print(f"Could not load motion templates: {exc}")
        return

    labels = sorted({template.label for template in templates})
    print(f"Loaded {len(templates)} templates for: {', '.join(labels)}")

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    recording = False
    trajectory: list[tuple[float, float]] = []
    result_text = "No result yet"
    status = "Press SPACE to record a gesture. Q quits."
    start_time = time.monotonic()
    failed_reads = 0

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads >= 30:
                    print("Webcam stopped providing frames.")
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
                    trajectory.append(palm_center(lms))
                    for previous, current in zip(trajectory, trajectory[1:]):
                        height, width = frame.shape[:2]
                        cv2.line(
                            frame,
                            (int(previous[0] * width), int(previous[1] * height)),
                            (int(current[0] * width), int(current[1] * height)),
                            (0, 255, 0), 3, cv2.LINE_AA,
                        )

            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (width, 42), (10, 10, 10), cv2.FILLED)
            cv2.putText(frame, f"Motion demo | Templates: {', '.join(labels)}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, result_text, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, status, (8, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.imshow("SASL Motion Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" "):
                if not recording:
                    recording = True
                    trajectory = []
                    result_text = "Recording..."
                    status = "Perform the gesture, then press SPACE to classify."
                else:
                    recording = False
                    if len(trajectory) < 3:
                        result_text = "Rejected: too few trajectory points"
                        status = "Press SPACE to try again."
                        continue
                    result = classify_trajectory(trajectory, templates)
                    if result is None:
                        result_text = "Rejected: no template matched"
                    else:
                        label, confidence = result
                        result_text = f"Detected: {label} ({confidence:.0%})"
                    status = "Press SPACE for another gesture, or Q to quit."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
