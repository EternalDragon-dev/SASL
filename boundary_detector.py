from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundaryEvent:
    timestamp: float
    segment_frames: int
    low_speed_frames: int


class BoundaryDetector:
    """Very small, explainable boundary detector for the Recommendation 6 MVP.

    It answers a narrow question: "has this hand been still long enough to
    treat the current segment as a completed letter?"
    """

    def __init__(
        self,
        min_segment_frames: int = 5,
        low_speed_threshold: float = 0.08,
        pause_frames_for_boundary: int = 3,
        max_segment_frames: int = 90,
    ) -> None:
        self.min_segment_frames = min_segment_frames
        self.low_speed_threshold = low_speed_threshold
        self.pause_frames_for_boundary = pause_frames_for_boundary
        self.max_segment_frames = max_segment_frames

        self._segment_frames = 0
        self._low_speed_frames = 0

    def reset(self) -> None:
        self._segment_frames = 0
        self._low_speed_frames = 0

    def update(self, timestamp: float, speed: float) -> BoundaryEvent | None:
        self._segment_frames += 1

        if speed <= self.low_speed_threshold:
            self._low_speed_frames += 1
        else:
            self._low_speed_frames = 0

        if self._segment_frames < self.min_segment_frames:
            return None

        if self._segment_frames > self.max_segment_frames:
            self.reset()
            return BoundaryEvent(timestamp=timestamp, segment_frames=self._segment_frames, low_speed_frames=self._low_speed_frames)

        if self._low_speed_frames >= self.pause_frames_for_boundary:
            event = BoundaryEvent(
                timestamp=timestamp,
                segment_frames=self._segment_frames,
                low_speed_frames=self._low_speed_frames,
            )
            self.reset()
            return event

        return None
