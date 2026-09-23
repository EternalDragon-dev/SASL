import math

import pytest

from sequence_classifier import BoundaryDetector, SequenceClassifier, WordBuffer


class _Obs:
    def __init__(self, timestamp, x, y, speed=0.0, label="A", conf=0.9):
        self.timestamp = timestamp
        self.palm_x = x
        self.palm_y = y
        self.palm_z = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        self.speed = speed
        self.static_prediction = label
        self.static_confidence = conf


def test_boundary_detector_emits_segment_after_pause():
    detector = BoundaryDetector(
        min_segment_frames=2,
        low_speed_threshold=0.08,
        pause_frames_for_boundary=2,
        max_segment_frames=20,
    )

    event = None
    for i, speed in enumerate([0.3, 0.25, 0.05, 0.04, 0.03, 0.03]):
        event = detector.update(i, speed)
        if i >= 3 and event is not None:
            break

    assert event is not None


def test_sequence_classifier_collects_letter_events():
    classifier = SequenceClassifier(min_segment_frames=2, min_confidence=0.55)

    for i in range(5):
        classifier.observe(_Obs(0.1 * i, 0.1, 0.2, speed=0.02, label="A", conf=0.9))

    events = classifier.flush_completed_segments()
    assert len(events) >= 1
    assert events[0].label == "A"


def test_sequence_classifier_does_not_repeat_same_letter_while_held():
    classifier = SequenceClassifier(min_segment_frames=2, min_confidence=0.55, low_speed_threshold=0.2, pause_frames_for_boundary=2)

    first = classifier.observe(_Obs(0.1, 0.1, 0.2, speed=0.02, label="C", conf=0.9))
    second = classifier.observe(_Obs(0.2, 0.1, 0.2, speed=0.02, label="C", conf=0.9))
    third = classifier.observe(_Obs(0.3, 0.1, 0.2, speed=0.02, label="C", conf=0.9))

    assert first is None
    assert second is not None
    assert third is None


def test_word_buffer_commits_and_clears_words():
    buffer = WordBuffer()
    buffer.add_letter("A")
    buffer.add_letter("P")
    buffer.add_letter("P")
    buffer.add_letter("L")
    buffer.add_letter("E")
    buffer.commit_current_word()

    assert buffer.committed_words == ["APPLE"]
    assert buffer.current_word == ""
