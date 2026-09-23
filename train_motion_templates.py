#!/usr/bin/env python3
"""Build a normalized motion-template artifact from motion_data/.

Usage:
    python train_motion_templates.py

Input:
    motion_data/<LETTER>/sample_*.csv

Output:
    models/motion_templates.npz
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from motion_classifier import normalize_trajectory

MOTION_DIR = Path("motion_data")
MODEL_PATH = Path("models/motion_templates.npz")


def load_points(path: Path) -> np.ndarray:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"{path} is empty")
    return np.array([[float(row["palm_x"]), float(row["palm_y"])] for row in rows], dtype=np.float32)


def build_templates() -> tuple[np.ndarray, np.ndarray]:
    if not MOTION_DIR.exists():
        raise SystemExit("No motion_data directory found. Capture samples with motion_calibrate.py first.")

    trajectories: list[np.ndarray] = []
    labels: list[str] = []
    for letter_dir in sorted(path for path in MOTION_DIR.iterdir() if path.is_dir()):
        for sample_path in sorted(letter_dir.glob("sample_*.csv")):
            try:
                trajectories.append(normalize_trajectory(load_points(sample_path)))
                labels.append(letter_dir.name.upper())
            except (ValueError, KeyError) as exc:
                print(f"Skipping {sample_path}: {exc}")

    if not trajectories:
        raise SystemExit("No valid motion samples found.")
    return np.stack(trajectories), np.array(labels)


def main() -> None:
    trajectories, labels = build_templates()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_PATH, trajectories=trajectories, labels=labels)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Saved {len(labels)} motion templates to {MODEL_PATH}")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} samples")


if __name__ == "__main__":
    main()
