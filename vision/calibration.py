"""Thresholds measured from your hand, rather than guessed at.

Every number the detectors compare against -- how straight a finger has to
be, how far a wrist has to turn -- was picked by reasoning about
geometry.  That is a starting point, not an answer: a hand two metres from
a laptop webcam measures differently from one at arm's length, and
differently again on someone else's camera.

Calibrating records what your hand actually does while you make each
gesture, and sets every threshold in the gap between what you do and what
you do not, with a margin either side.  The result is written to a file
and loaded at startup.

    python main.py --calibrate
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "sarv_calibration.json"


@dataclass
class Calibration:
    """One set of measured thresholds."""

    extended_ratio: float
    open_ratio: float
    fist_reach: float
    swipe_turn: float
    swipe_turn_speed: float
    min_hand_on_screen: float

    #: Notes about the loading rather than thresholds, so they are kept
    #: out of the file and out of any comparison between two calibrations.
    #:
    #: ``incomplete`` is what the file did not have and took from the
    #: defaults; ``pulled`` is what it did have and was implausible.  Both
    #: mean those thresholds were not really measured.
    incomplete: tuple = field(default=(), compare=False, repr=False)
    pulled: tuple = field(default=(), compare=False, repr=False)

    @classmethod
    def load(cls, path=None):
        """Read a saved calibration, or None if there is not one.

        A file written before a threshold existed keeps everything it does
        have, and the missing one falls back to its default.  Throwing the
        whole thing away instead meant a calibration somebody had actually
        sat down and done was silently ignored the next time a threshold
        was added -- which is the worst of both, since nothing said so.
        """

        path = Path(path) if path else DEFAULT_PATH

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None

        known = cls.thresholds()
        given = {k: v for k, v in data.items() if k in known}

        if not given:
            return None

        missing = known - set(given)

        if missing:
            defaults = current()
            given.update({name: getattr(defaults, name) for name in missing})

        calibration = cls(**given)
        calibration.incomplete = tuple(sorted(missing))

        return calibration

    @classmethod
    def thresholds(cls):
        """The names that are actually thresholds."""

        return {f.name for f in fields(cls) if f.compare}

    def save(self, path=None):
        path = Path(path) if path else DEFAULT_PATH

        kept = {name: value for name, value in asdict(self).items()
                if name in self.thresholds()}

        path.write_text(json.dumps(kept, indent=2) + "\n")

        return path

    #: What each threshold may sensibly be, whatever was measured.  A
    #: measurement can be wrong -- a hand held badly, a landmark the camera
    #: guessed at -- and a threshold far outside these does not fail
    #: gently: it reads everything, or nothing, as the gesture.
    BOUNDS = {
        "extended_ratio": (0.60, 0.95),
        "open_ratio": (0.70, 0.98),
        "fist_reach": (0.90, 1.40),
        "min_hand_on_screen": (0.02, 0.12),
    }

    def sensible(self):
        """This calibration with anything implausible pulled back.

        Returns it and whatever had to be pulled, which is worth saying:
        it means that threshold was not really measured.
        """

        pulled = []

        for name, (low, high) in self.BOUNDS.items():
            was = getattr(self, name)
            now = min(high, max(low, was))

            if now != was:
                setattr(self, name, now)
                pulled.append(f"{name} was {was:.2f}, kept to {now:.2f}")

        return self, tuple(pulled)

    def apply(self):
        """Put these numbers where the detectors read them from.

        They are module-level constants read at the moment of each
        decision, so replacing them takes effect immediately and nothing
        has to be rebuilt.
        """

        from . import hand_state, motion

        hand_state.EXTENDED_RATIO = self.extended_ratio
        hand_state.OPEN_RATIO = self.open_ratio
        hand_state.FIST_REACH = self.fist_reach
        hand_state.MIN_HAND_ON_SCREEN = self.min_hand_on_screen

        motion.SWIPE_TURN = self.swipe_turn
        motion.SWIPE_TURN_SPEED = self.swipe_turn_speed

    def describe(self):
        return [
            f"finger out above      {self.extended_ratio:.2f}",
            f"hand open above       {self.open_ratio:.2f}",
            f"fist below            {self.fist_reach:.2f}",
            f"wrist turn            {self.swipe_turn:.2f} at {self.swipe_turn_speed:.2f}/s",
            f"smallest hand read    {self.min_hand_on_screen:.3f} of the frame",
        ]


def edge(values, share, default):
    """A value near the edge of a set, ignoring the last few.

    Not the very lowest or highest.  A calibration run of a hundred frames
    holds a few where the hand was arriving, leaving, or half read, and
    one of those is enough to put the edge somewhere absurd -- a fist
    measured 0.10 for "how straight is a straight finger" this way, when
    every real frame said about 0.9.
    """

    if not values:
        return default

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))

    return ordered[index]


def between(low, high, defaults_to, fraction=0.5, least=0.08):
    """A threshold in the gap between two measurements.

    ``low`` is the highest value seen when the answer should be no,
    ``high`` the lowest when it should be yes.  If they overlap there is no
    gap to sit in -- the two cases were not distinguishable on this camera
    -- so the existing value is kept rather than inventing one.

    A gap too narrow to be meant is treated the same way.  Given two poses
    measured a tenth apart, this used to answer with the midpoint and sound
    confident -- a threshold with no margin on either side, which on a hand
    that moves at all puts both cases on the wrong side of it about half
    the time.  Barely separable is not separable.
    """

    if high - low < least:
        return defaults_to, False

    return low + (high - low) * fraction, True


def from_samples(poses, moves, current):
    """Work thresholds out from what was recorded.

    ``poses`` maps a pose name to the readings taken while it was held,
    ``moves`` maps a movement name to the peak of each repetition, and
    ``current`` is what the thresholds are now, used wherever a
    measurement did not separate cleanly.

    Returns the calibration and a list of anything that could not be
    measured, which is worth telling the user rather than hiding.
    """

    warnings = []

    # A finger is out above this.  Curled fingers come from the fist,
    # straight ones from the open hand -- so the line goes between the
    # straightest curled finger and the most bent straight one.
    curled = [score
              for reading in poses.get("fist", [])
              for score in reading["ext"].values()]

    straight = [score
                for reading in poses.get("open", [])
                for score in reading["ext"].values()]

    # Each of these has its own scale, so what counts as a usable gap
    # differs: finger extension runs from about 0.4 to 1.0, and the fist
    # reach from 1.0 to 1.6.
    extended_ratio, ok = between(
        edge(curled, 0.90, 0.0), edge(straight, 0.10, 1.0),
        current.extended_ratio, least=0.10)

    if not ok:
        warnings.append("could not tell a straight finger from a curled one")

    # A hand is open above this, which is a different question: the other
    # side is not a fist but a hand at rest, whose fingers are straighter
    # than a fist and slacker than a spread hand.  With only the line
    # above to fall on, a resting hand landed on "open" and asked for
    # something.
    slack = [score
             for reading in poses.get("rest", [])
             for score in reading["ext"].values()]

    open_ratio, ok = between(
        edge(slack, 0.90, 0.0), edge(straight, 0.10, 1.0),
        current.open_ratio, least=0.06)

    if not ok:
        warnings.append("could not tell an open hand from a resting one")

    # A fist is below this.  The other side is a hand at rest, which is
    # the case that matters: it is what the camera sees most of the time.
    fist = [score
            for reading in poses.get("fist", [])
            for score in reading["reach"].values()]

    resting = [score
               for reading in poses.get("rest", [])
               for score in reading["reach"].values()]

    fist_reach, ok = between(
        edge(fist, 0.90, 0.0), edge(resting, 0.10, 99.0),
        current.fist_reach, least=0.12)

    if not ok:
        warnings.append("could not tell a fist from a resting hand")

    # Movements: take the weakest repetition and sit comfortably under it,
    # so the gentlest gesture you actually made still counts.
    #
    # Both directions go in together, because a wrist does not turn as far
    # one way as the other.  Measuring only the easy direction sets a bar
    # the hard one never clears, which reads as "swipe right does not
    # work" -- it was working, and being asked for more than the joint had.
    turns = [peak
             for name, peaks in moves.items()
             if name.startswith("turn")
             for peak in peaks]

    swipe_turn, swipe_turn_speed, missed = _movement(
        turns, current.swipe_turn, current.swipe_turn_speed)

    if not turns:
        warnings.append("no wrist turn was recorded")
    elif missed:
        warnings.append("one of the wrist turns barely moved, and was ignored")

    # How small a hand may look and still be read.  Taken from how large
    # yours looked while calibrating, so calibrating across the room lets
    # SARV reach that far.
    #
    # It can only ever loosen the limit, never tighten it: calibrating at
    # the keyboard would otherwise set the floor at a hand's size there
    # and quietly stop the thing working from across the room, which is
    # the point of it.
    sizes = [reading["scale"]
             for readings in poses.values()
             for reading in readings]

    min_hand = (min(edge(sizes, 0.10, 0.1) * 0.6, current.min_hand_on_screen)
                if sizes else current.min_hand_on_screen)

    return Calibration(
        extended_ratio=round(extended_ratio, 3),
        open_ratio=round(open_ratio, 3),
        fist_reach=round(fist_reach, 3),
        swipe_turn=round(swipe_turn, 3),
        swipe_turn_speed=round(swipe_turn_speed, 3),
        min_hand_on_screen=round(min_hand, 4),
    ), warnings


#: An attempt smaller than this share of the best is treated as one that
#: did not happen -- a hand out of frame, or a prompt missed.
BOTCHED = 0.4

#: However gentle the gesture, a threshold below this share of the default
#: is not believed.  Left alone, one failed attempt could set the bar at
#: almost nothing and the app would fire at everything.
FLOOR = 0.4


def _movement(peaks, default_size, default_speed, margin=0.55):
    """Thresholds from the peaks of several repetitions.

    The weakest attempt sets the bar, because a gesture that only works
    when made emphatically will fail on the day.  But an attempt far below
    the others was not a gentle gesture, it was a missed one -- the prompt
    came while the hand was out of frame, or halfway through moving back --
    and letting that set the bar puts it on the floor.
    """

    if not peaks:
        return default_size, default_speed, False

    best = max(size for size, _ in peaks)

    real = [(size, speed) for size, speed in peaks if size >= best * BOTCHED]

    size = min(s for s, _ in real) * margin
    speed = min(v for _, v in real) * margin

    return (max(size, default_size * FLOOR),
            max(speed, default_speed * FLOOR),
            len(real) < len(peaks))


def load_and_apply(path=None):
    """Use a saved calibration if there is one.  Returns it, or None."""

    calibration = Calibration.load(path)

    if calibration is None:
        return None

    calibration, pulled = calibration.sensible()
    calibration.pulled = tuple(pulled)
    calibration.apply()

    return calibration


def current():
    """The thresholds as they stand, measured or not."""

    from . import hand_state, motion

    return Calibration(
        extended_ratio=hand_state.EXTENDED_RATIO,
        open_ratio=hand_state.OPEN_RATIO,
        fist_reach=hand_state.FIST_REACH,
        swipe_turn=motion.SWIPE_TURN,
        swipe_turn_speed=motion.SWIPE_TURN_SPEED,
        min_hand_on_screen=hand_state.MIN_HAND_ON_SCREEN,
    )
