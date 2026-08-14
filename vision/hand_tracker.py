"""MediaPipe, wrapped so the rest of the project never sees it.

    with HandTracker() as tracker:
        hand = tracker.track(frame)

``hand`` is None when no hand is visible, otherwise it carries the 21
landmarks and which hand MediaPipe thinks it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


class TrackerError(RuntimeError):
    """The hand model is missing or unusable."""


@dataclass(frozen=True)
class Hand:
    landmarks: list
    handedness: str


class HandTracker:
    """Finds one hand in a frame."""

    def __init__(self, model_path=MODEL_PATH, confidence=0.7):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self._landmarker = None
        self._mp = None

    def open(self):
        if not self.model_path.exists():
            raise TrackerError(f"hand model missing at {self.model_path}")

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp

        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path)),
                num_hands=1,
                min_hand_detection_confidence=self.confidence,
                min_hand_presence_confidence=self.confidence,
                min_tracking_confidence=self.confidence,
            )
        )

        return self

    def track(self, frame):
        """Find a hand in a BGR frame, or return None."""

        import cv2

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        found = self._landmarker.detect(image)

        if not found.hand_landmarks:
            return None

        return Hand(
            landmarks=found.hand_landmarks[0],
            handedness=found.handedness[0][0].category_name
        )

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.close()
