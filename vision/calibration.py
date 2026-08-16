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

#: The fingers that get measured, in the order they are named everywhere
#: else.  Kept here rather than imported so this module still loads
#: without the rest of vision.
FINGERS = ("index", "middle", "ring", "pinky")


@dataclass
class Calibration:
    """One set of measured thresholds."""

    extended_ratio: float
    open_ratio: float
    fist_reach: float
    fist_curl: float
    swipe_turn: float
    swipe_turn_speed: float
    swipe_lift: float
    swipe_lift_speed: float
    crosstalk_turn: float
    crosstalk_lift: float
    min_hand_on_screen: float

    #: Defaulted, so calibrations from before it existed still construct.
    #: The default equals the shipped EXTENDED_RATIO -- one line, the old
    #: behaviour -- because it only helps when this user's held-down and
    #: resting fingers measurably separate, and guessing a gap would cost
    #: hands that hold the swipe pose's spare fingers nearly straight.
    folded_ratio: float = 0.82

    #: The folded line per finger, or None for the single line above.
    #: Fingers do not rest equally -- a ring finger at rest sits a good
    #: deal straighter than a pinky at rest -- so each gets its line
    #: drawn between its own held-down and resting readings.
    folded_ratios: dict | None = None

    #: Notes about the loading rather than thresholds, so they are kept
    #: out of the file and out of any comparison between two calibrations.
    #:
    #: ``incomplete`` is what the file did not have and took from the
    #: defaults; ``pulled`` is what it did have and was implausible.  Both
    #: mean those thresholds were not really measured.
    incomplete: tuple = field(default=(), compare=False, repr=False)
    pulled: tuple = field(default=(), compare=False, repr=False)

    #: Thresholds that were recorded but could not be told apart, in
    #: words.  Separate from ``incomplete``, which is about a file too old
    #: to hold them at all -- the two need different things said about
    #: them, and putting these in there produced a sentence saying the
    #: calibration predated a piece of advice.
    notes: tuple = field(default=(), compare=False, repr=False)

    #: Worth knowing but not a fault: nothing here is guessed because of
    #: it.  Said after a calibration run, where it can be acted on, and
    #: not at every startup afterwards.
    advice: tuple = field(default=(), compare=False, repr=False)

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

    def save(self, path=None, profile=None):
        """Write the thresholds, and the readings they came from.

        The thresholds stay at the top level, where they have always
        been, so an older build reads this file unchanged.  The readings
        go beside them, so a newer one can work them out again.
        """

        path = Path(path) if path else DEFAULT_PATH

        kept = {name: value for name, value in asdict(self).items()
                if name in self.thresholds()}

        if profile is not None:
            kept["profile"] = profile.to_dict()

        path.write_text(json.dumps(kept, indent=2) + "\n")

        return path

    #: What each threshold may sensibly be, whatever was measured.  A
    #: measurement can be wrong -- a hand held badly, a landmark the camera
    #: guessed at -- and a threshold far outside these does not fail
    #: gently: it reads everything, or nothing, as the gesture.
    #: The ceiling on how briskly a raise must be made was 3.00, guessed
    #: with nothing to go on.  The first real calibration measured 3.15
    #: and had it pulled back -- a bound that trims the first honest
    #: measurement it meets is measuring the guess, not the hand.  These
    #: are here to catch the absurd, so it has room now.
    #:
    #: The hand-size floor is capped at the shipped default rather than
    #: at anything roomier.  Calibrating is allowed to reach further than
    #: the default, never less far: measured from a hand held near the
    #: lens it comes out around 0.10, and everything past arm's length is
    #: then thrown away before any gesture code sees it -- which looks
    #: exactly like the app having stopped working.
    BOUNDS = {
        "extended_ratio": (0.60, 0.95),
        "open_ratio": (0.70, 0.98),
        "fist_reach": (0.90, 1.40),
        "fist_curl": (0.35, 0.75),
        "folded_ratio": (0.35, 0.85),
        # The movement caps are set from what ordinary, unhurried
        # gestures measure -- a flick about 0.48 of turn at 2.2/s, a
        # gentle raise or lower about 0.79 of lift at 1.7/s: a
        # calibrated bar may ask for less than that but never more, so
        # no recording -- however emphatic, however botched -- can set a
        # bar the user's everyday gesture fails to clear.
        "swipe_turn": (0.15, 0.45),
        "swipe_turn_speed": (0.40, 2.00),
        "swipe_lift": (0.25, 0.70),
        "swipe_lift_speed": (0.40, 1.50),
        "crosstalk_turn": (0.15, 0.95),
        "crosstalk_lift": (0.15, 1.20),
        "min_hand_on_screen": (0.012, 0.022),
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
        hand_state.FOLDED_RATIO = self.folded_ratio
        hand_state.FOLDED_RATIOS = (dict(self.folded_ratios)
                                    if self.folded_ratios else None)
        hand_state.OPEN_RATIO = self.open_ratio
        hand_state.FIST_REACH = self.fist_reach
        hand_state.FIST_CURL = self.fist_curl
        hand_state.MIN_HAND_ON_SCREEN = self.min_hand_on_screen

        motion.SWIPE_TURN = self.swipe_turn
        motion.SWIPE_TURN_SPEED = self.swipe_turn_speed
        motion.SWIPE_LIFT = self.swipe_lift
        motion.SWIPE_LIFT_SPEED = self.swipe_lift_speed
        motion.CROSSTALK_TURN = self.crosstalk_turn
        motion.CROSSTALK_LIFT = self.crosstalk_lift

    def describe(self):
        return [
            f"finger out above      {self.extended_ratio:.2f}",
            f"hand open above       {self.open_ratio:.2f}",
            f"fist below            {self.fist_reach:.2f} reaching,"
            f" or {self.fist_curl:.2f} curled",
            f"wrist turn            {self.swipe_turn:.2f} at {self.swipe_turn_speed:.2f}/s",
            f"hand raised           {self.swipe_lift:.2f} at {self.swipe_lift_speed:.2f}/s",
            f"a turn may carry      {self.crosstalk_turn:.2f} of a rise",
            f"a rise may carry      {self.crosstalk_lift:.2f} of a turn",
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


#: Where a line sits between two measurements, when one of the two was
#: made deliberately for the camera and the other was not.
#:
#: A pose held for a prompt is tidier than the same pose made casually: a
#: fist for a calibration is squeezed, two fingers are folded right down,
#: an open hand is spread wide.  In use that side relaxes toward the line
#: while the other side -- a hand at rest, a finger that is simply
#: straight -- stays where it was.  Splitting the difference gives half
#: the room to a side that does not need it.
#:
#: This was found the expensive way.  A fist measured from a hard clench
#: put its line below where an ordinary fist reads, and stopped
#: recognising fists altogether.
DELIBERATE = 0.65
STEADY = 1 - DELIBERATE


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


#: Where a summary is taken.  Not the extremes: a run of a hundred frames
#: holds a few where the hand was arriving, leaving or half read, and one
#: of those is enough to put an edge somewhere absurd.
EDGES = (0.10, 0.50, 0.90)


@dataclass
class Profile:
    """What your hand measured, rather than what was concluded from it.

    Thresholds are conclusions, and a conclusion is only as good as the
    arithmetic that reached it.  Twice in one day that arithmetic turned
    out to be wrong -- once taking the hand-size floor from a close-up
    session, once working out a finger threshold from poses that did not
    include the hard case -- and both times the recording was gone, so
    the only way to benefit from the fix was to stand in front of the
    camera again.

    Keeping the readings makes a fix free: the thresholds are worked out
    again at startup from what was already recorded.

    It is also what lets gestures be combined later.  A pose and a
    movement are measured separately, because that is what they are: how
    straight your fingers go is one fact about you and how far your hand
    travels is another.  Four fingers raised is a pose already measured
    and a movement already measured, so it needs no new recording -- only
    a line saying the two go together.
    """

    #: pose -> finger -> {"ext": [low, middle, high], "reach": [...]}
    poses: dict

    #: movement -> [[size, speed], ...], the peak of each repetition
    moves: dict

    #: how big the hand looked, [low, middle, high]
    scale: list

    @classmethod
    def from_samples(cls, poses, moves):
        """Summarise a calibration session."""

        summary = {}

        for name, readings in poses.items():
            fingers = {}

            for finger in FINGERS:
                measured = {
                    kind: [reading[kind][finger]
                           for reading in readings
                           if finger in reading.get(kind, {})]
                    for kind in ("ext", "reach")
                }

                if any(measured.values()):
                    fingers[finger] = {
                        kind: [round(edge(values, share, 0.0), 3)
                               for share in EDGES]
                        for kind, values in measured.items()
                    }

            summary[name] = fingers

        sizes = [reading["scale"]
                 for readings in poses.values()
                 for reading in readings
                 if "scale" in reading]

        def keep(peak):
            """One repetition, in whichever shape it was recorded.

            The recorder keeps both axes of every movement -- a dict of
            turn, speed, lift and lift_speed, which is what lets the
            crosstalk be measured from it.  This used to unpack every
            peak as a bare (size, speed) pair, which is the shape of
            recordings from before both axes were kept -- so the very
            recordings the crosstalk needed were the ones that crashed
            the calibration that made them.
            """

            if isinstance(peak, dict):
                return {name: round(value, 3)
                        for name, value in peak.items()}

            size, speed = peak

            return [round(size, 3), round(speed, 3)]

        return cls(
            poses=summary,
            moves={name: [keep(peak) for peak in peaks]
                   for name, peaks in moves.items()},
            scale=[round(edge(sizes, share, 0.1), 4) for share in EDGES],
        )

    # Which end of a finger's summary answers which question.
    LOW, HIGH = 0, 2

    def among(self, poses, end, kind="ext", fingers=FINGERS):
        """One end of the readings, across some fingers of some poses."""

        return [self.poses[pose][finger][kind][end]
                for pose in poses
                if pose in self.poses
                for finger in fingers
                if finger in self.poses[pose]]

    #: Which measurements a movement is judged on.
    AXES = {"turn": ("turn", "speed"), "lift": ("lift", "lift_speed")}

    def reps(self, *prefixes):
        """Every repetition of the movements whose names start like this.

        Both directions go in together, because a wrist does not turn as
        far one way as the other and an arm does not rise as far as it
        falls.  Measuring only the easy direction sets a bar the hard one
        never clears, which reads as "it does not detect that way".
        """

        return [(name, peak)
                for name, peaks in self.moves.items()
                if name.startswith(prefixes)
                for peak in peaks]

    def moves_like(self, *prefixes, axis=None):
        """The size and speed of each repetition, on one axis.

        Recordings made before both axes were kept hold a bare pair, and
        that pair is whatever the movement was asked for -- so it answers
        for its own axis and says nothing about the other.
        """

        axis = axis or ("lift" if prefixes[0] in ("raise", "lower") else "turn")
        size_of, speed_of = self.AXES[axis]

        found = []

        for name, peak in self.reps(*prefixes):
            if isinstance(peak, dict):
                found.append((peak.get(size_of, 0.0), peak.get(speed_of, 0.0)))
            elif axis == self.asked_of(name):
                found.append(tuple(peak))

        return [pair for pair in found if pair[0] > 0]

    @staticmethod
    def asked_of(name):
        """Which axis a movement of this name was asked for."""

        return "turn" if name.startswith("turn") else "lift"

    def rest_signature(self):
        """The middle reading of each resting finger, or None.

        The recording the vision side compares a live hand against.  The
        calibration takes the resting pose for one promise -- that it
        asks for nothing -- and the thresholds alone cannot always keep
        it: they are lines, and a resting hand drifts across them.  The
        signature is the recording keeping its own promise.
        """

        if "rest" not in self.poses:
            return None

        middles = {finger: readings["ext"][1]
                   for finger, readings in self.poses["rest"].items()
                   if len(readings.get("ext", ())) == 3}

        return middles or None

    def crosstalk(self):
        """How much of the other movement a deliberate one carries.

        Turning the wrist raises the hand a little and raising it turns
        the wrist a little, so every movement has to say which of the two
        it was.  That was a guessed constant, applied to one of them and
        not the other -- and when it was applied to both, turns stopped
        registering, because a turn really does raise the hand.

        Measured instead: whatever share of the wrong movement your own
        turns and raises carry, with room above it.  Below that share a
        movement is clearly one thing; above it, it is a diagonal nobody
        meant and firing either would be a guess.

        The shares are measured in the same fixed units the detector
        asks the question in -- not against the calibrated bars.
        Measured against the bars, the allowance changed meaning with
        every recalibration: one that tightened the turn bar re-scored
        every ordinary drop as mostly turn, and the lower had to go far
        past the raise's distance before it counted.
        """

        from . import motion

        shares = {"turn": [], "lift": []}

        for name, peak in self.reps("turn", "raise", "lower", "lift"):
            if not isinstance(peak, dict):
                continue

            sideways = abs(peak.get("turn", 0.0)) / motion.CROSSTALK_UNIT_TURN
            upright = abs(peak.get("lift", 0.0)) / motion.CROSSTALK_UNIT_LIFT

            axis = self.asked_of(name)
            asked, other = ((sideways, upright) if axis == "turn"
                            else (upright, sideways))

            if asked > 0:
                shares[axis].append(other / asked)

        if not (shares["turn"] and shares["lift"]):
            return None, None, False

        # Room above the worst one seen, so a movement no worse than the
        # ones recorded is not turned away.
        return (max(shares["turn"]) * 1.3, max(shares["lift"]) * 1.3, True)

    def derive(self, current):
        """Work the thresholds out.  Returns them and anything unmeasured.

        ``current`` is what the thresholds are now, used wherever a
        measurement did not separate cleanly -- which is worth saying
        rather than hiding, because it means that one was not measured.
        """

        warnings = []

        # A finger is out above this.  Curled fingers come from the fist,
        # straight ones from the open hand -- so the line goes between the
        # straightest curled finger and the most bent straight one.
        #
        # And from the two-finger pose, which is where the hard case is.
        # A fist is fingers shut; two fingers up is the other two merely
        # held down, which is a good deal straighter, and it is *that* the
        # line has to sit above.  Measured from a fist alone it came out
        # at 0.65, and a peace sign then read as four fingers out and
        # matched nothing at all.
        curled = (self.among(["fist"], self.HIGH)
                  + self.among(["two"], self.HIGH, fingers=("ring", "pinky")))

        straight = (self.among(["open"], self.LOW)
                    + self.among(["two"], self.LOW,
                                 fingers=("index", "middle")))

        # Each of these has its own scale, so what counts as a usable gap
        # differs: finger extension runs from about 0.4 to 1.0, and the
        # fist reach from 1.0 to 1.6.
        # The curled side is the deliberate one: fingers folded right
        # down for a prompt, and held looser in use.
        extended_ratio, ok = between(
            max(curled, default=0.0), min(straight, default=1.0),
            current.extended_ratio, fraction=DELIBERATE, least=0.10)

        if not ok:
            warnings.append("could not tell a finger held out from one held down")

        # A finger is deliberately down below this -- a second line under
        # the one above, because between them lies the resting hand.
        # "Down" used to mean merely "not out", and a slack resting
        # finger qualified: rest curls the fingers in order, index
        # straightest and pinky deepest, so wherever the out-line cut,
        # some depth of slack left exactly the index above it and a hand
        # doing nothing read as POINT -- one slacker finger from the
        # swipe pose.
        #
        # Drawn per finger, because fingers do not rest equally: this is
        # a line between one finger's held-down reading and the same
        # finger's resting one, and a ring finger at rest sits a good
        # deal straighter than a pinky at rest.  One shared line either
        # refused a casually held-down ring or let a resting pinky count
        # as folded; it could not avoid both.
        #
        # And only drawn where this finger's recordings actually
        # separate.  Some hands hold the swipe pose's spare fingers
        # nearly straight, and for them the honest answer is the old
        # single line, said out loud, rather than a gap invented between
        # two measurements that overlap.
        folded_ratios = {}
        inseparable = []

        for finger in ("middle", "ring", "pinky"):
            held_down = self.among(["fist"], self.HIGH, fingers=(finger,))

            if finger in ("ring", "pinky"):
                held_down += self.among(["two"], self.HIGH,
                                        fingers=(finger,))

            resting = self.among(["rest"], self.LOW, fingers=(finger,))

            if resting:
                line, ok = between(
                    max(held_down, default=0.0), min(resting),
                    extended_ratio, fraction=DELIBERATE, least=0.06)
            else:
                line, ok = extended_ratio, False

            if not ok:
                line = extended_ratio
                inseparable.append(finger)

            # A folded finger is certainly not out; the lines must agree.
            folded_ratios[finger] = round(min(line, extended_ratio), 3)

        # The index is never asked to fold -- every pattern wants it out
        # -- so it keeps the strictest line there is.
        folded_ratios["index"] = round(extended_ratio, 3)

        if "ring" in inseparable or "pinky" in inseparable:
            # The fingers the swipe pose folds: these are the ones whose
            # line matters, and theirs could not be drawn.
            warnings.append("could not tell a finger held down from one"
                            " at rest")

        folded_ratio = min(folded_ratios.values())

        # A hand is open above this, which is a different question: the
        # other side is not a fist but a hand at rest, whose fingers are
        # straighter than a fist and slacker than a spread hand.  With
        # only the line above to fall on, a resting hand landed on "open"
        # and asked for something.
        #
        # Both sides are the weakest finger, because "open" means every
        # finger is out: the open hand's slackest finger has to clear the
        # line, and the resting hand only needs its slackest to fall
        # below.  Asking instead that every resting finger fall below is
        # a stricter question than the test asks, and it threw away a
        # measurement that was there -- a hand resting with a nearly
        # straight index finger and a curled pinky separates perfectly
        # well on the pinky, and this reported that it could not be told
        # apart at all.
        # Here it is the upper side that was deliberate -- a hand spread
        # wide for a prompt is wider than one shown in passing -- so the
        # line goes nearer the resting hand, which is the steady one.
        open_ratio, ok = between(
            min(self.among(["rest"], self.HIGH), default=0.0),
            min(self.among(["open"], self.LOW), default=1.0),
            current.open_ratio, fraction=STEADY, least=0.06)

        if not ok:
            warnings.append("could not tell an open hand from a resting one")

        # An open hand is one whose every finger is out and then some, so
        # this line cannot sit below the one for a single finger.  The two
        # are measured against different poses and can come out the wrong
        # way round, and when they do the second test is not a stricter
        # one -- it is no test at all, and a slack hand reads as open.
        open_ratio = max(open_ratio, extended_ratio)

        # A fist is below this.  The other side is a hand at rest, which
        # is the case that matters: it is what the camera sees most.
        fist_reach, ok = between(
            max(self.among(["fist"], self.HIGH, "reach"), default=0.0),
            min(self.among(["rest"], self.LOW, "reach"), default=99.0),
            current.fist_reach, fraction=DELIBERATE, least=0.12)

        if not ok:
            warnings.append("could not tell a fist from a resting hand")

        # The same question asked of how curled the fingers are, which is
        # the measurement that survives a fist made casually: where the
        # tips end up depends on how hard the hand is squeezed, and how
        # folded a finger is barely does.
        fist_curl, ok = between(
            max(self.among(["fist"], self.HIGH), default=0.0),
            min(self.among(["rest"], self.LOW), default=1.0),
            current.fist_curl, fraction=DELIBERATE, least=0.10)

        if not ok:
            warnings.append("could not tell a closed hand from a slack one")

        turns = self.moves_like("turn")
        lifts = self.moves_like("raise", "lower", "lift")

        swipe_turn, swipe_turn_speed, missed = _movement(
            turns, current.swipe_turn, current.swipe_turn_speed)

        if not turns:
            warnings.append("no wrist turn was recorded")
        elif missed:
            warnings.append("one of the wrist turns barely moved, and was ignored")

        swipe_lift, swipe_lift_speed, missed = _movement(
            lifts, current.swipe_lift, current.swipe_lift_speed)

        if not lifts:
            warnings.append("no raise or lower was recorded")
        elif missed:
            warnings.append("one of the raises barely moved, and was ignored")

        crosstalk_turn, crosstalk_lift, ok = self.crosstalk()

        if not ok:
            crosstalk_turn = current.crosstalk_turn
            crosstalk_lift = current.crosstalk_lift
            warnings.append("turning and raising were not measured against"
                            " each other")
        elif crosstalk_turn >= 0.95:
            warnings.append("your turns and raises look much alike, so one"
                            " may be taken for the other")

        # How small a hand may look and still be read.  Taken from how
        # large yours looked while calibrating, so calibrating across the
        # room lets SARV reach that far.  It can only ever loosen the
        # limit, never tighten it -- see BOUNDS, which is what actually
        # holds that line.
        min_hand = (min(self.scale[self.LOW] * 0.6, current.min_hand_on_screen)
                    if self.scale else current.min_hand_on_screen)

        # Nothing is wrong when this happens, but it is worth saying.  A
        # session recorded near the lens says nothing about what can be
        # read from across the room, so the limit stays where it was --
        # and somebody who calibrated in order to gain range has gained
        # none, which is not visible from the numbers.
        advice = []

        if self.scale and min_hand >= current.min_hand_on_screen:
            advice.append(
                "you calibrated close to the camera, so the range is"
                " unchanged -- run it again from where you mean to stand"
                " to reach further")

        derived = Calibration(
            extended_ratio=round(extended_ratio, 3),
            folded_ratio=round(folded_ratio, 3),
            folded_ratios=folded_ratios,
            open_ratio=round(open_ratio, 3),
            fist_reach=round(fist_reach, 3),
            fist_curl=round(fist_curl, 3),
            swipe_turn=round(swipe_turn, 3),
            swipe_turn_speed=round(swipe_turn_speed, 3),
            swipe_lift=round(swipe_lift, 3),
            swipe_lift_speed=round(swipe_lift_speed, 3),
            crosstalk_turn=round(crosstalk_turn, 3),
            crosstalk_lift=round(crosstalk_lift, 3),
            min_hand_on_screen=round(min_hand, 4),
        )

        derived.advice = tuple(advice)

        return derived, warnings

    def to_dict(self):
        return {"poses": self.poses, "moves": self.moves, "scale": self.scale}

    @classmethod
    def load(cls, path=None):
        """The profile inside a saved calibration, or None if it has none.

        Files written before profiles existed hold only the thresholds.
        They keep working -- see load_and_apply -- they simply cannot be
        worked out again.
        """

        path = Path(path) if path else DEFAULT_PATH

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None

        stored = data.get("profile")

        if not isinstance(stored, dict) or "poses" not in stored:
            return None

        return cls(poses=stored.get("poses", {}),
                   moves=stored.get("moves", {}),
                   scale=stored.get("scale", []))


def from_samples(poses, moves, current):
    """Summarise a session and work the thresholds out of it."""

    return Profile.from_samples(poses, moves).derive(current)


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

    Speed is given more room than size, and how much more is taken from
    the repetitions themselves.  Asked to do the same thing four times,
    people vary how fast they do it more than how far -- measured, and
    true of both movements on the recording this came from -- so a single
    margin under the weakest attempt leaves the speed bar too high.  It
    was: a raise had to finish inside a third of a second, and one made
    at a normal pace never counted.
    """

    if not peaks:
        return default_size, default_speed, False

    best = max(size for size, _ in peaks)

    real = [(size, speed) for size, speed in peaks if size >= best * BOTCHED]

    sizes = [size for size, _ in real]
    speeds = [speed for _, speed in real]

    size = min(sizes) * margin
    speed = min(speeds) * margin * _spread(speeds) / max(_spread(sizes), 1e-6)

    return (max(size, default_size * FLOOR),
            max(speed, default_speed * FLOOR),
            len(real) < len(peaks))


def _spread(values):
    """How alike the repetitions were: 1.0 identical, lower the less so."""

    return min(values) / max(values) if values and max(values) > 0 else 1.0


def load_and_apply(path=None):
    """Use a saved calibration if there is one.  Returns it, or None.

    Worked out again from the readings whenever the file has them, so a
    correction to the arithmetic reaches a session recorded before it was
    written -- without anyone standing in front of the camera again.
    Files from before that keep working on their stored thresholds.
    """

    profile = Profile.load(path)

    if profile is not None:
        calibration, unmeasured = profile.derive(current())
        calibration.notes = tuple(unmeasured)
    else:
        calibration = Calibration.load(path)

    if calibration is None:
        return None

    calibration, pulled = calibration.sensible()
    calibration.pulled = tuple(pulled)
    calibration.apply()

    # The rest recording itself, not only what was concluded from it:
    # the veto that keeps a resting hand reading as nothing needs the
    # signature, and files without a profile simply go without the veto.
    from . import hand_state

    hand_state.REST_SIGNATURE = (
        profile.rest_signature() if profile is not None else None)

    return calibration


def current():
    """The thresholds as they stand, measured or not."""

    from . import hand_state, motion

    return Calibration(
        extended_ratio=hand_state.EXTENDED_RATIO,
        folded_ratio=hand_state.FOLDED_RATIO,
        folded_ratios=(dict(hand_state.FOLDED_RATIOS)
                       if hand_state.FOLDED_RATIOS else None),
        open_ratio=hand_state.OPEN_RATIO,
        fist_reach=hand_state.FIST_REACH,
        fist_curl=hand_state.FIST_CURL,
        swipe_turn=motion.SWIPE_TURN,
        swipe_turn_speed=motion.SWIPE_TURN_SPEED,
        swipe_lift=motion.SWIPE_LIFT,
        swipe_lift_speed=motion.SWIPE_LIFT_SPEED,
        crosstalk_turn=motion.CROSSTALK_TURN,
        crosstalk_lift=motion.CROSSTALK_LIFT,
        min_hand_on_screen=hand_state.MIN_HAND_ON_SCREEN,
    )
