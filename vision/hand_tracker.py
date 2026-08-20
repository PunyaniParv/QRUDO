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
    #: The other hand in frame, when there is one -- a second Hand, or
    #: None.  The pairing layer in gestures.py reads it to tell both
    #: hands making the same pose (a two-hand gesture) from one hand
    #: making it alone.
    partner: "Hand | None" = None


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
        self._span = 0.0

    def view(self):
        """The (x0, y0, w, h) fraction of the frame to look at now."""

        if self._window is not None:
            return self._window

        view = self.SWEEP[self._step % len(self.SWEEP)]
        self._step += 1

        return view

    def following_near(self):
        """Whether a hand near enough for full-frame work is tracked.

        The spotter (see HandTracker.track) asks this before spending
        a look on the whole frame: near hands are found there easily,
        far hands are not, and far is the window's territory."""

        return self._window is not None and self._span >= 0.07

    #: How fast the follow window chases its target, per frame.  It
    #: GLIDES rather than jumps: video-mode tracking carries the last
    #: frame's hand rectangle into this frame's view, and a view that
    #: recentres and rescales sharply every frame -- worst with two
    #: hands, whose union window breathes with the distance between
    #: them -- occasionally hands the converter a degenerate rectangle,
    #: which is the seed of the rare native crash the supervisor mops
    #: up.  A gliding window keeps consecutive views nearly identical,
    #: which starves that seed.
    CHASE = 0.35

    def found(self, cx, cy, span):
        """A hand at (cx, cy), ``span`` hand-scale, all frame fractions."""

        side = min(1.0, max(self.WINDOW_FLOOR, span * self.WINDOW_SPANS))

        if self._window is not None:
            x0, y0, w, h = self._window
            ocx, ocy, oside = x0 + w / 2, y0 + h / 2, w

            cx = ocx + self.CHASE * (cx - ocx)
            cy = ocy + self.CHASE * (cy - ocy)
            side = oside + self.CHASE * (side - oside)

        self._window = self._around(cx, cy, side)
        self._misses = 0
        self._span = span

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
        self._spotter = None
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
                # Two, because a pair of hands is its own vocabulary: a
                # both-hands-open pose must read as 2_OPEN_PALM, not as
                # whichever single palm the model happened to find
                # first.  One-hand behaviour is untouched -- the second
                # slot simply comes back empty.
                num_hands=2,
                min_hand_detection_confidence=self.confidence,
                min_hand_presence_confidence=self.confidence,
                min_tracking_confidence=self.confidence,
            )
        )

        # A second, STATELESS landmarker with one job: noticing that a
        # second hand has entered the frame.  The follow window hugs
        # the tracked hand -- which is exactly why a second hand could
        # never join a pose: it was not in the picture the detector
        # saw.  And the video-mode tracker cannot simply glance at the
        # full frame now and then, because video mode leans on frame-
        # to-frame continuity and a jumping view breaks it.  So the
        # main tracker keeps its steady window, and this image-mode
        # spotter -- no memory, no continuity to break -- checks the
        # whole frame every few frames while a near hand is followed.
        # When it sees two hands, the window is widened to hold both,
        # and from then on the main tracker follows the pair.
        self._spotter = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=self.confidence,
            )
        )
        self._spot_beat = 0

        self._last_ms = 0
        self._last_centre = None   # where the primary hand last was

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
        crop_h, crop_w = view.shape[:2]   # the region's true size, for remap

        # Every view is resized to the one fixed shape before MediaPipe
        # sees it.  Video mode keeps internal state sized to the frames
        # it has been fed; alternating crop sizes frame-to-frame walked
        # that state off the end of a smaller buffer and SEGFAULTED deep
        # in warpPerspective -- the whole app gone, two seconds after
        # launch, worst exactly when no hand is in view (the sweep
        # churns shapes fastest then).  A constant shape removes the
        # churn, and scaling a crop up is the digital zoom the scanner
        # wanted anyway: the far hand lands on the detector even larger.
        if view.shape[:2] != (height, width):
            view = cv2.resize(view, (width, height))

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
            self._last_centre = None
            return None

        # What the crop saw, restated for the frame everyone else sees.
        # The box is the crop's true region -- landmarks are normalised
        # within the view, and that normalisation is the same whether or
        # not the view was resized on its way to the model.
        box = (left / width, top / height,
               crop_w / width, crop_h / height)

        world = found.hand_world_landmarks

        # The shape tests prefer the world landmarks; losing them is the
        # kind of silent degradation that reads as "poses stopped
        # working" with nothing on screen to say why.  Once is enough.
        if not world and not self._warned_no_world:
            self._warned_no_world = True
            print("  ! the tracker returned no world landmarks; pose "
                  "reading degrades", file=sys.stderr)

        hands = [
            Hand(
                landmarks=unwrap(found.hand_landmarks[i], box),
                world=world[i] if world and i < len(world) else None,
                handedness=found.handedness[i][0].category_name,
            )
            for i in range(len(found.hand_landmarks))
        ]

        # With two hands in frame, "the" hand must stay the same one
        # from frame to frame -- the model reports them in whatever
        # order it likes, and letting the primary flip between hands
        # would churn every stabiliser downstream.  Continuity first
        # (nearest to where the primary just was), size on first sight.
        def centre(h):
            return h.landmarks[hand_state.MIDDLE_MCP]

        if len(hands) > 1:
            if self._last_centre is not None:
                lx, ly = self._last_centre
                hands.sort(key=lambda h: (centre(h).x - lx) ** 2
                           + (centre(h).y - ly) ** 2)
            else:
                hands.sort(
                    key=lambda h: -hand_state.hand_scale(h.landmarks))

        primary = hands[0]
        c0 = centre(primary)
        s0 = hand_state.hand_scale(primary.landmarks)
        self._last_centre = (c0.x, c0.y)

        if len(hands) > 1:
            partner = hands[1]
            c1 = centre(partner)
            s1 = hand_state.hand_scale(partner.landmarks)

            # The model sometimes finds the SAME hand twice, slightly
            # offset -- a phantom that matches every pose the real
            # hand makes, which is how two-hand gestures fired from
            # one hand.  Two real hands cannot overlap: a "partner"
            # sitting within a hand's own span of the primary is the
            # primary, seen double, and is dropped.
            dist = ((c0.x - c1.x) ** 2 + (c0.y - c1.y) ** 2) ** 0.5

            if dist < 0.9 * max(s0, s1):
                self._scanner.found(c0.x, c0.y, s0)
                return primary

            # The follow window must hold BOTH hands, or the one it
            # drops is lost until the next sweep: centre on their
            # midpoint, sized so the span covers the distance between
            # them with a hand's worth of margin either side.
            span = max(s0, s1,
                       (dist + s0 + s1) / self._scanner.WINDOW_SPANS)
            self._scanner.found((c0.x + c1.x) / 2, (c0.y + c1.y) / 2,
                                span)

            return Hand(landmarks=primary.landmarks, world=primary.world,
                        handedness=primary.handedness, partner=partner)

        self._scanner.found(c0.x, c0.y, s0)

        # One hand tracked: every sixth frame, while it is near enough
        # for full-frame work, the spotter checks whether a second hand
        # is out there.  Seeing one, it widens the window to hold both;
        # the pair itself is then tracked by the main landmarker from
        # the very next frame.
        self._spot_beat += 1

        if (self._spot_beat % 6 == 0 and self._scanner.following_near()
                and self._spotter is not None):
            whole = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            spotted = self._spotter.detect(whole)

            if len(spotted.hand_landmarks) > 1:
                def mid(lm):
                    return lm[hand_state.MIDDLE_MCP]

                a, b = spotted.hand_landmarks[0], spotted.hand_landmarks[1]
                sa, sb = hand_state.hand_scale(a), hand_state.hand_scale(b)
                dist = ((mid(a).x - mid(b).x) ** 2
                        + (mid(a).y - mid(b).y) ** 2) ** 0.5
                span = max(sa, sb,
                           (dist + sa + sb) / self._scanner.WINDOW_SPANS)
                self._scanner.found((mid(a).x + mid(b).x) / 2,
                                    (mid(a).y + mid(b).y) / 2, span)

        return primary

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

        if self._spotter is not None:
            self._spotter.close()
            self._spotter = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.close()
