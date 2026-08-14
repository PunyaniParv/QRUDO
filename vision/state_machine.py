"""The parts of gesture recognition that need memory.

A single frame is not enough to decide anything.  A gesture has to survive
a few frames before it is believed, a swipe is a shape traced over half a
second, and both need to be forgotten when the hand leaves.

This is where all of that lives, so gestures.py and motion.py can be
written as questions about the present.
"""

from __future__ import annotations

import time
from collections import Counter, deque

#: How many frames a held gesture is judged over, and how many of them
#: must agree.  At about 30 frames a second this is a sixth of a second --
#: long enough to ignore a flicker, short enough to feel immediate.
STABLE_FRAMES = 5
STABLE_AGREEMENT = 4


class GestureStabiliser:
    """Believes a gesture only once it has stopped changing."""

    def __init__(self, frames=STABLE_FRAMES, agreement=STABLE_AGREEMENT):
        self.frames = frames
        self.agreement = agreement
        self.history = deque(maxlen=frames)

    def update(self, raw_gesture):
        """Feed one frame's guess, get the settled answer."""

        self.history.append(raw_gesture)

        if len(self.history) < self.frames:
            return "UNKNOWN"

        gesture, count = Counter(self.history).most_common(1)[0]

        if count >= self.agreement:
            return gesture

        return "UNKNOWN"

    def clear(self):
        self.history.clear()


class MotionHistory:
    """A short rolling record of where the hand has been.

    Samples are taken on every frame a hand is visible, whatever pose it is
    in.  That separation is deliberate: the pose decides whether a swipe is
    *allowed*, this decides whether one *happened*.  Recording only during
    a recognised pose means one badly classified frame erases the movement
    so far, which is what made swipes feel broken.
    """

    def __init__(self, window, capacity=64):
        self.window = window
        self.samples = deque(maxlen=capacity)

    def add(self, now, **values):
        self.samples.append((now, values))

    def recent(self, now):
        """Samples inside the window, oldest first."""

        return [
            sample for sample in self.samples
            if now - sample[0] <= self.window
        ]

    def clear(self):
        self.samples.clear()


class SwipeState:
    """Arming and cooling-down for swipes.

    A swipe is only accepted while the pose has been seen recently, rather
    than on every frame of the motion.  A hand turning away from the camera
    passes through angles that are genuinely hard to classify, and losing
    the pose halfway through a turn should not lose the turn.
    """

    def __init__(self, arm_hold, cooldown, window, hold=0.0, gap=0.2):
        self.arm_hold = arm_hold
        self.cooldown = cooldown
        self.hold = hold
        self.gap = gap
        self.history = MotionHistory(window)
        self.armed_until = 0.0
        self.armed_kind = None
        self.cooldown_until = 0.0
        self.armed_since = 0.0
        self.neutral_y = None
        self.lifted = False
        self.settling = False
        self._seen_kind = None
        self._seen_since = 0.0
        self._last_pose_at = 0.0

    def note_pose(self, now, kind):
        """The pose seen this frame; arms a swipe once it has been held.

        Arming on the first frame that looks right is what let a hand doing
        something else fire a swipe: reaching across the desk passes through
        shapes that resemble two fingers for an instant, and an instant was
        all it took.  Requiring the pose to persist costs a deliberate
        gesture nothing -- you are holding it anyway -- and rules out the
        accidental ones almost entirely.
        """

        if kind is None:
            # A pose lost for a frame or two is the same pose: recognition
            # flickers most while the hand is moving, which is precisely
            # when a swipe is under way.  Only a real gap restarts it.
            if now - self._last_pose_at > self.gap:
                self._seen_kind = None
            return

        # Only appearing and disappearing restart the clock.  Which of the
        # two poses it is can flip mid-turn, as the fingers swing past the
        # angle that separates them, and that is the same hand held
        # continuously -- not a new gesture.
        if self._seen_kind is None:
            self._seen_since = now

        self._seen_kind = kind
        self._last_pose_at = now

        if now - self._seen_since >= self.hold:
            if not self.is_armed(now):
                # Note when this began, so the movement that counts is the
                # movement made since -- not whatever the hand was doing on
                # its way into shot.
                self.armed_since = now
                self.neutral_y = None

            self.armed_until = now + self.arm_hold
            self.armed_kind = kind

    def is_armed(self, now):
        return now < self.armed_until

    def is_cooling(self, now):
        return now < self.cooldown_until

    def fired(self, now):
        """A swipe just happened: start fresh so its tail cannot fire again.

        ``settling`` is the important half.  Every swipe is followed by the
        hand coming back, and coming back from a leftward turn is a
        rightward turn -- the same movement a deliberate one makes.  Time
        alone cannot separate them, so nothing counts again until the hand
        has actually stopped moving.
        """

        self.settling = True
        self.history.clear()
        self.armed_until = 0.0
        self.armed_kind = None
        self._seen_kind = None
        self.cooldown_until = now + self.cooldown

    def clear(self):
        self.history.clear()
        self.armed_until = 0.0
        self.armed_kind = None
        self.neutral_y = None
        self.lifted = False
        self.settling = False
        self._seen_kind = None


class Presence:
    """Tolerates the hand vanishing for a moment.

    A fast gesture is precisely when MediaPipe is most likely to lose the
    hand for a frame or two: the blur that makes a movement fast is the
    blur that makes a hand hard to find.  Forgetting everything the instant
    it disappears throws away the swipe that was in progress -- which is
    why a slow turn worked and a quick one did not.

    So the hand has to be gone for a little while before anything is
    forgotten.  Long enough to ride out the blur, short enough that a hand
    which really left cannot combine with the next one.
    """

    def __init__(self, grace=0.35):
        self.grace = grace
        self.last_seen = None

    def seen(self, moment):
        self.last_seen = moment

    def missing(self, moment):
        """True once it has been gone long enough to forget it."""

        if self.last_seen is None:
            return False

        if moment - self.last_seen < self.grace:
            return False

        self.last_seen = None

        return True


def now():
    """Indirection so tests can replay motion on their own clock."""

    return time.time()
