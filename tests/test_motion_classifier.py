import numpy as np

from motion_classifier import MotionTemplate, classify_trajectory, normalize_trajectory


def test_normalize_trajectory_is_translation_and_scale_invariant():
    original = np.array([[2, 3], [3, 3], [4, 4], [5, 4]], dtype=np.float32)
    transformed = original * 4 + 20

    assert np.allclose(normalize_trajectory(original), normalize_trajectory(transformed))


def test_classifier_matches_close_template_and_rejects_far_path():
    template_points = normalize_trajectory([[0, 0], [1, 0], [1, 1], [2, 1]])
    templates = [MotionTemplate("J", template_points)]

    assert classify_trajectory([[10, 10], [11, 10], [11, 11], [12, 11]], templates)[0] == "J"
    assert classify_trajectory([[0, 0], [0, 1], [1, 1], [1, 0]], templates, rejection_threshold=0.05) is None
