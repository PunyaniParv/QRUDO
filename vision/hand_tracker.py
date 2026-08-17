"""MediaPipe, wrapped so the rest of the project never sees it.

    with HandTracker() as tracker:
        hand = tracker.track(frame)

``hand`` is None when no hand is visible, otherwise it carries the 21
landmarks and which hand MediaPipe thinks it is.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


class TrackerError(RuntimeError):
    """The hand model is missing or unusable."""


@dataclass(frozen=True)
class Hand:
    """One hand, in both of the forms MediaPipe reports it.

    ``landmarks`` are normalised to the frame: use them for where the hand
    is on screen.  ``world`` is in metres with real depth: use it for what
    shape the hand is making.  Measuring shape from the normalised set is
    what made a hand pointing at the camera look like a fist.
    """

    landmarks: list
    world: list | None
    handedness: str


class _Point:
    """A landmark restated in full-frame coordinates, after a crop."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def unwrap(landmarks, box):
    """Crop-relative landmarks, restated for the whole frame.

    ``box`` is the crop as (x0, y0, w, h) fractions of the frame.  The
    depth is scaled by the crop's width for the same reason x is: the
    model reports z in units of the image it saw, and the image it saw
    was ``w`` of the real one.  The world landmarks need none of this
    -- they are metres of hand, whatever window they came through.
    """

    x0, y0, w, h = box

    return [_Point(x0 + point.x * w, y0 + point.y * h, point.z * w)
            for point in landmarks]


class Scanner:
    """Where to show the detector a frame, so far hands exist at all.

    The palm detector sees whatever it is given at about two hundred
    pixels across, however large the frame really is.  A hand at two
    and a half metres is forty pixels of a 1600-wide frame -- five
    pixels of the detector's view, and five pixels is not a hand to
    it.  Nothing about the landmark model is short-sighted; once a
    hand is found it is measured through a crop around where it was.
    The range limit is entirely in the finding.

    So the finding gets the same courtesy.  While there is no hand,
    the full frame alternates with a sweep of enlarged views, one look
    per frame -- never two looks at one frame, so nothing about a near
    hand slows down, and a near hand is still found on the very next
    full view.  A hand found anywhere is then followed through a
    window a few hand-spans wide, which is the one place it is known
    to be, and where it is large enough to keep finding.  Lost for
    long enough, the sweep starts over.
    """

    #: The views tried while no hand is tracked, as (x0, y0, w, h)
    #: fractions of the frame.  The full frame between every zoomed
    #: look, so a hand at arm's length waits one frame at most; the
    #: corner views overlap generously, so a hand on a seam is whole
    #: in at least one of them.
    FULL = (0.0, 0.0, 1.0, 1.0)
    SWEEP = [
        FULL, (0.25, 0.25, 0.50, 0.50),
        FULL, (0.00, 0.00, 0.60, 0.60),
        FULL, (0.40, 0.00, 0.60, 0.60),
        FULL, (0.00, 0.40, 0.60, 0.60),
        FULL, (0.40, 0.40, 0.60, 0.60),
    ]

    #: The window a followed hand is watched through, in spans of the
    #: hand itself: the whole hand, plus a gesture's worth of travel on
    #: every side.
    WINDOW_SPANS = 5.0

    #: And never smaller than this much of the frame, so the window
    #: cannot shrink to the point of starving the detector.
    WINDOW_FLOOR = 0.25

    #: A miss widens the window before anything is given up: the blur
    #: of a fast gesture is exactly when the hand is hardest to find,
    #: and hardest to find is not gone.
    WIDEN = 1.5
    MISSES_TO_SWEEP = 5

    def __init__(self):
        self._step = 0
        self._window = None
        self._misses = 0

    def view(self):
        """The (x0, y0, w, h) fraction of the frame to look at now."""

        if self._window is not None:
            return self._window

        view = self.SWEEP[self._step % len(self.SWEEP)]
        self._step += 1

        return view

    def found(self, cx, cy, span):
        """A hand at (cx, cy), ``span`` hand-scale, all frame fractions."""

        side = min(1.0, max(self.WINDOW_FLOOR, span * self.WINDOW_SPANS))

        self._window = self._around(cx, cy, side)
        self._misses = 0

    def missed(self):
        """No hand this frame: widen the window, then give it up."""

        if self._window is None:
            return

        self._misses += 1

        if self._misses >= self.MISSES_TO_SWEEP:
            self._window = None
            self._misses = 0
            self._step = 0
            return

        x0, y0, w, h = self._window
        side = min(1.0, w * self.WIDEN)

        self._window = self._around(x0 + w / 2, y0 + h / 2, side)

    @staticmethod
    def _around(cx, cy, side):
        """A ``side``-sized box centred there, held inside the frame."""

        x0 = min(max(cx - side / 2, 0.0), 1.0 - side)
        y0 = min(max(cy - side / 2, 0.0), 1.0 - side)

        return (x0, y0, side, side)


class HandTracker:
    """Finds one hand in a frame."""

    #: How sure MediaPipe has to be before reporting a hand.
    #:
    #: 0.7 is its own suggestion and suits a hand filling the frame.  A
    #: hand three metres away is twenty pixels across, and the model is
    #: rightly less certain about it -- at 0.7 it says nothing at all,
    #: which reads as QRUDO not working from across the room.  Lower, it
    #: offers a guess, and the gesture tests behind it are strict enough to
    #: throw away the bad ones.
    DEFAULT_CONFIDENCE = 0.5

    def __init__(self, model_path=MODEL_PATH, confidence=DEFAULT_CONFIDENCE):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self._landmarker = None
        self._mp = None
        self._warned_no_world = False
        self._scanner = Scanner()

    def open(self):
        if not self.model_path.exists():
            raise TrackerError(f"hand model missing at {self.model_path}")

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp

        # VIDEO mode, not the default IMAGE mode.  Image mode treats
        # every frame as an unrelated photograph and runs the full-frame
        # palm detector on each one, so the whole background gets a
        # fresh chance to distract the model thirty times a second.  In
        # video mode that search only runs to *acquire* a hand; between
        # acquisitions the model follows the hand it has through a crop
        # around where it just was, and the background simply is not in
        # the picture it looks at.  It is also the mode in which
        # min_tracking_confidence means anything at all.
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=self.confidence,
                min_hand_presence_confidence=self.confidence,
                min_tracking_confidence=self.confidence,
            )
        )

        self._last_ms = 0

        return self

    def track(self, frame):
        """Find a hand in a BGR frame, or return None."""

        import cv2

        from . import hand_state

        if self._landmarker is None or self._mp is None:
            raise TrackerError("tracker used before it was opened")

        # One look per frame, chosen by the scanner: the full frame, an
        # enlarged view while searching, or the window around a hand
        # being followed.  See Scanner for why -- this is what carries
        # detection out to hands the full frame is too small to find.
        height, width = frame.shape[:2]
        x0, y0, w, h = self._scanner.view()
        left, top = int(x0 * width), int(y0 * height)
        view = frame[top:top + max(1, int(h * height)),
                     left:left + max(1, int(w * width))]

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        )

        # Video mode wants each frame stamped later than the one before.
        # Wall-clock milliseconds, nudged forward if two frames land in
        # the same one.
        stamp = int(time.monotonic() * 1000)
        self._last_ms = max(self._last_ms + 1, stamp)

        found = self._landmarker.detect_for_video(image, self._last_ms)

        if not found.hand_landmarks:
            self._scanner.missed()
            return None

        # What the crop saw, restated for the frame everyone else sees.
        box = (left / width, top / height,
               view.shape[1] / width, view.shape[0] / height)
        landmarks = unwrap(found.hand_landmarks[0], box)

        centre = landmarks[hand_state.MIDDLE_MCP]
        self._scanner.found(centre.x, centre.y, hand_state.hand_scale(landmarks))

        world = found.hand_world_landmarks

        # The shape tests prefer the world landmarks; losing them is the
        # kind of silent degradation that reads as "poses stopped
        # working" with nothing on screen to say why.  Once is enough.
        if not world and not self._warned_no_world:
            self._warned_no_world = True
            print("  ! the tracker returned no world landmarks; pose "
                  "reading degrades", file=sys.stderr)

        return Hand(
            landmarks=landmarks,
            world=world[0] if world else None,
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
