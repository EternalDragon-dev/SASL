"""Trajectory normalization and template matching for motion letters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionTemplate:
    label: str
    points: np.ndarray


def load_motion_templates(path: str | Path = "models/motion_templates.npz") -> list[MotionTemplate]:
    """Load normalized trajectory templates exported by train_motion_templates.py."""
    with np.load(path, allow_pickle=False) as artifact:
        trajectories = artifact["trajectories"]
        labels = artifact["labels"]
    if trajectories.ndim != 3 or trajectories.shape[2] != 2:
        raise ValueError("motion template artifact must contain trajectories shaped (N, points, 2)")
    if len(trajectories) != len(labels):
        raise ValueError("motion template labels and trajectories have different lengths")
    return [
        MotionTemplate(str(label), np.asarray(points, dtype=np.float32))
        for label, points in zip(labels.tolist(), trajectories)
    ]


def normalize_trajectory(points, sample_count: int = 32) -> np.ndarray:
    """Translate, scale, and resample an ordered Nx2 trajectory."""
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("trajectory must contain at least two x/y points")

    values = values - values[0]
    scale = float(np.max(np.linalg.norm(values, axis=1)))
    if scale <= 1e-6:
        raise ValueError("trajectory has no measurable movement")
    values = values / scale

    distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= 1e-6:
        raise ValueError("trajectory has no measurable path length")

    targets = np.linspace(0.0, cumulative[-1], sample_count)
    x = np.interp(targets, cumulative, values[:, 0])
    y = np.interp(targets, cumulative, values[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


def trajectory_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return mean pointwise Euclidean distance between normalized paths."""
    if left.shape != right.shape:
        raise ValueError("trajectories must have the same shape")
    return float(np.mean(np.linalg.norm(left - right, axis=1)))


def classify_trajectory(
    points,
    templates: list[MotionTemplate],
    rejection_threshold: float = 0.35,
) -> tuple[str, float] | None:
    """Return the closest template and confidence, or reject the trajectory."""
    if not templates:
        return None
    normalized = normalize_trajectory(points, templates[0].points.shape[0])
    distances = [trajectory_distance(normalized, template.points) for template in templates]
    best_index = int(np.argmin(distances))
    distance = distances[best_index]
    if distance > rejection_threshold:
        return None
    return templates[best_index].label, max(0.0, 1.0 - distance / rejection_threshold)
