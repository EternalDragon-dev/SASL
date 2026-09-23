from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from boundary_detector import BoundaryDetector
from word_buffer import WordBuffer


@dataclass
class FrameObservation:
    timestamp: float
    hand_label: str
    palm_x: float
    palm_y: float
    palm_z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    speed: float
    feature_vector: Optional[np.ndarray] = None
    static_prediction: Optional[str] = None
    static_confidence: float = 0.0


@dataclass
class RecognizedLetter:
    label: str
    confidence: float
    start_time: float
    end_time: float
    segment_type: str = "static"


class SequenceClassifier:
    """Simple Recommendation 6 sequence layer.

    It keeps a short segment buffer, uses a lightweight pause detector to decide
    when a letter candidate is complete, and emits a single RecognizedLetter once.
    """

    def __init__(
        self,
        min_segment_frames: int = 3,
        min_confidence: float = 0.55,
        low_speed_threshold: float = 0.08,
        pause_frames_for_boundary: int = 3,
        max_segment_frames: int = 90,
        duplicate_cooldown: float = 2.0,
    ) -> None:
        self.min_segment_frames = min_segment_frames
        self.min_confidence = min_confidence
        self._segment: deque[FrameObservation] = deque(maxlen=max_segment_frames)
        self._completed: deque[RecognizedLetter] = deque()
        self._detector = BoundaryDetector(
            min_segment_frames=min_segment_frames,
            low_speed_threshold=low_speed_threshold,
            pause_frames_for_boundary=pause_frames_for_boundary,
            max_segment_frames=max_segment_frames,
        )
        self._last_emitted_label: Optional[str] = None
        self._last_emitted_time: float = -float("inf")
        self._duplicate_cooldown = duplicate_cooldown
        self._last_observed_label: Optional[str] = None

    def observe(self, observation: FrameObservation) -> Optional[RecognizedLetter]:
        self._segment.append(observation)

        boundary = self._detector.update(observation.timestamp, observation.speed)
        if boundary is None:
            return None

        if len(self._segment) < self.min_segment_frames:
            self._segment.clear()
            return None

        votes = Counter(
            obs.static_prediction
            for obs in self._segment
            if obs.static_prediction is not None
        )
        if not votes:
            self._segment.clear()
            return None

        label, _count = votes.most_common(1)[0]
        confidences = [
            obs.static_confidence
            for obs in self._segment
            if obs.static_prediction == label
        ]
        confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
        if confidence < self.min_confidence:
            self._segment.clear()
            return None

        if self._last_emitted_label is not None and label == self._last_emitted_label:
            self._segment.clear()
            return None

        event = RecognizedLetter(
            label=label,
            confidence=confidence,
            start_time=self._segment[0].timestamp,
            end_time=observation.timestamp,
            segment_type="static",
        )
        self._last_emitted_label = label
        self._last_emitted_time = observation.timestamp
        self._last_observed_label = label
        self._completed.append(event)
        self._segment.clear()
        return event

    def flush_completed_segments(self) -> list[RecognizedLetter]:
        events = list(self._completed)
        self._completed.clear()
        return events


__all__ = [
    "BoundaryDetector",
    "FrameObservation",
    "RecognizedLetter",
    "SequenceClassifier",
    "WordBuffer",
]
