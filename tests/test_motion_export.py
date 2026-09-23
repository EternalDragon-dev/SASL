import csv

import numpy as np

from motion_classifier import load_motion_templates
from train_motion_templates import build_templates


def test_motion_template_export_round_trip(tmp_path, monkeypatch):
    motion_dir = tmp_path / "motion_data" / "J"
    motion_dir.mkdir(parents=True)
    sample_path = motion_dir / "sample_001.csv"
    with sample_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "palm_x", "palm_y", "index_x", "index_y", "index_z", "hand_scale", "frame_index"])
        writer.writerows([
            [0.0, 0.0, 0.0, 0, 0, 0, 1, 0],
            [0.1, 1.0, 0.0, 1, 0, 0, 1, 1],
            [0.2, 1.0, 1.0, 1, 1, 0, 1, 2],
        ])

    monkeypatch.setattr("train_motion_templates.MOTION_DIR", tmp_path / "motion_data")
    trajectories, labels = build_templates()
    assert trajectories.shape == (1, 32, 2)
    assert labels.tolist() == ["J"]

    model_path = tmp_path / "motion_templates.npz"
    np.savez_compressed(model_path, trajectories=trajectories, labels=labels)
    loaded = load_motion_templates(model_path)
    assert loaded[0].label == "J"
    assert loaded[0].points.shape == (32, 2)
