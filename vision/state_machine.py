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


#: How far a finger has to fall back before it counts as down again.
#: Roughly the width of the wobble MediaPipe puts on a finger that is
#: half hidden behind the hand, which is every finger that is folded.
FINGER_BAND = 0.07

#: How many readings a finger's answer is taken from.
#:
#: Five, so that two bad readings in a row still decide nothing.  This is
#: what pays for keeping three fingers apart from two: told only that a
#: finger reads as out, there is no distinguishing a third finger held up
#: from a folded one misread, and the only thing that separates them is
#: that one of them persists.  Five readings is long enough for the
#: persistence to show.
#:
#: The cost is that a genuine change takes three frames to be believed --
#: a tenth of a second at thirty a second, on a pose that is held anyway.
#: Nothing about a swipe is delayed by it: this decides which fingers are
#: out, not where the hand is going.
FINGER_FRAMES = 5


class FingerMemory:
    """Which fingers are out, refusing to change its mind over nothing.

    A finger held near the line crosses it and back several times a
    second, and that is the measurement moving rather than the hand.  It
    costs whichever pose needs that finger to stay put: pointing needs
    three of them down at once, so it was the one that fumbled.

    Once a finger is called out it stays out until it is clearly in, and
    the other way round.  A real change clears the band in one frame and
    is not delayed; only the wobble is refused.

    The band alone is not enough, because not every bad reading is a small
    one.  A finger folded behind the hand is half hidden, and what
    MediaPipe reports for it is part measurement and part guess -- the
    guess is occasionally not close, and a reading well past the line
    walks straight through a band meant for wobble.  So each finger
    answers from the middle of its last few readings rather than its
    latest: one frame can then no longer decide anything, and it takes two
    bad ones in a row to be believed.
    """

    def __init__(self, band=FINGER_BAND, frames=FINGER_FRAMES):
        self.band = band
        self.frames = frames
        self.recent = {}
        self.out = {}
        self._last = None

        #: The steadied reading behind each answer, for whoever wants to
        #: ask a different question of the same finger.
        self.steady = {}

    def update(self, spans, threshold):
        """Feed this frame's readings, get what each finger counts as.

        Called once a frame.  Three questions are put to this in the
        course of one -- what the pose is, what a movement could be made
        from, and why -- and only the first of them is a new reading; see
        ``read`` for the others.

        ``threshold`` may be one number for every finger or a mapping
        per finger: a ring finger at rest sits much straighter than a
        pinky at rest, so one line for both is one line in the wrong
        place for one of them.
        """

        bars = (threshold if isinstance(threshold, dict)
                else dict.fromkeys(spans, threshold))

        for name, span in spans.items():
            threshold = bars[name]
            recent = self.recent.setdefault(name, deque(maxlen=self.frames))
            recent.append(span)

            # The middle reading, not the mean: an average is dragged by
            # a bad value in proportion to how bad it is, which is the
            # wrong response to one that is simply wrong.
            span = sorted(recent)[len(recent) // 2]
            self.steady[name] = span

            was = self.out.get(name)

            # Nothing remembered yet: the plain question, no band.
            if was is None:
                self.out[name] = span > threshold
            else:
                self.out[name] = (span > threshold - self.band if was
                                  else span > threshold)

        return dict(self.out)

    def read(self, spans, threshold):
        """The answer already worked out, without counting the frame again.

        Asking twice about one frame does not make it two frames.  Each
        question was pushing the same reading in again, so five readings
        covered under two frames of history and the steadying barely
        steadied anything -- which is what stopped two fingers being
        recognised.

        Falls back to asking plainly when nothing has been fed yet, so
        that anything driving the detectors directly still gets an answer.
        """

        if len(self.out) < len(spans):
            bars = (threshold if isinstance(threshold, dict)
                    else dict.fromkeys(spans, threshold))

            return {name: span > bars[name] for name, span in spans.items()}

        return dict(self.out)

    def clear(self):
        self.out.clear()
        self.recent.clear()
        self.steady.clear()
        self._last = None


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
        self.settling = False

        # For the vertical gesture: the height the hand is judged against,
        # and how long it has been still enough to move it there.
        self.neutral_y: float | None = None
        self.still_since: float | None = None
        self.raised = False

        # The same three for the sideways one: the aim it is judged
        # against, how long it has pointed there, and whether it has
        # already turned away from it and not yet come back.
        self.neutral_aim: float | None = None
        self.aim_still_since: float | None = None
        self.aim_rested_at = 0.0
        self.y_rested_at = 0.0
        self.turned = False
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
        self.turned = True
        self.history.clear()
        self.armed_until = 0.0
        self.armed_kind = None
        self._seen_kind = None
        self.cooldown_until = now + self.cooldown

    def clear(self):
        self.history.clear()
        self.armed_until = 0.0
        self.armed_since = 0.0
        self.armed_kind = None
        self.settling = False
        self.neutral_y = None
        self.still_since = None
        self.raised = False
        self.neutral_aim = None
        self.aim_still_since = None
        self.aim_rested_at = 0.0
        self.y_rested_at = 0.0
        self.turned = False
        self._seen_kind = None
        # The timestamps too.  Left behind, they are only "the past" for a
        # clock that keeps going forward -- which the fake clocks in tests
        # do not, so a leftover time from one test quietly changed what
        # the next one measured, and a real bug passed the suite for as
        # long as the order of the tests happened to hide it.
        self._seen_since = 0.0
        self._last_pose_at = 0.0
        self.cooldown_until = 0.0


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
