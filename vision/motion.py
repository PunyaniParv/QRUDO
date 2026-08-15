"""Swipes: what the hand did over the last half second.

    detect_swipe(landmarks, handedness) -> "SWIPE_LEFT" | "SWIPE_RIGHT" | None

Two fingers facing the camera -- an ordinary peace sign -- with the wrist
turned left or right.

There was a second gesture here, a pinch raised or lowered for volume, and
it is gone: defined by where the thumb is, it was mistaken in turn for a
fist, an open hand and two fingers, because the thumb is the landmark a
camera loses first.

That pose is deliberate.  Everything stays in the plane of the image,
which is where MediaPipe is by far the most accurate, so nothing about
recognising it leans on depth.  Aiming the fingers at the camera instead
puts the whole gesture along the one axis a single camera guesses at.

The turn is read the same whichever way the two fingers face -- held up,
or aimed at the camera.
"""

from __future__ import annotations

from . import gestures, hand_state
from .state_machine import SwipeState, now

# ---------------------------------------------------------
# Tuning
# ---------------------------------------------------------

SWIPE_WINDOW = 0.60       # seconds of movement considered
SWIPE_TURN = 0.30         # how far the aim must swing
SWIPE_TURN_SPEED = 0.80   # per second
SWIPE_CONSISTENCY = 0.65  # how directly the motion got where it ended up
SWIPE_MIN_SAMPLES = 3     # a fast flick is only a few frames long
ARM_HOLD = 0.60           # how long the pose keeps a swipe allowed
SWIPE_COOLDOWN = 0.60
SWIPE_QUIET = 0.12        # how still the hand must go before the next one
POSE_HOLD = 0.15          # the pose must be held this long before it counts

# Raising and lowering the same two fingers, for volume.  Height is judged
# against where the hand is resting rather than against where it was a
# moment ago: the way back from a raised hand is a lowered hand, and
# nobody rests their hand at the bottom of its range.
SWIPE_LIFT = 0.60         # hand-sizes it must cover
SWIPE_LIFT_SPEED = 1.20   # per second
LIFT_STILL = 0.50         # below this speed the hand counts as at rest
REST_DWELL = 1.00         # and must stay so this long to become the rest

#: How much of the other movement one may carry and still be itself, as a
#: share of itself.  Seeking and volume share a pose, so every movement
#: has to say which of the two it was.
#:
#: Asked of both alike.  It was asked of the raise only, against an
#: absolute size rather than a share -- which made a movement that was
#: half of each into a seek, because seeking was never asked the question.
#: Asking both against a guessed number was worse: a wrist turned sideways
#: really does raise the hand, so turns stopped registering.
#:
#: Calibration measures it, from how much of the wrong movement your own
#: turns and raises carry.  This is the fallback.
CROSSTALK = 0.60

#: Both scripts mirror the frame before detection, so x increases to the
#: user's right and a rightward swipe is a rise in x.
FRAME_IS_MIRRORED = True

_state = SwipeState(ARM_HOLD, SWIPE_COOLDOWN, SWIPE_WINDOW, POSE_HOLD)

#: Filled in every frame for the tuning overlay.
_debug = {}

#: The tuning numbers peak during the swipe and are back to nothing by the
#: time you look at the screen, so the highest recent value is kept too.
PEAK_HOLD = 2.0
_peaks = {}


def _peak(name, value, moment):
    """Highest value seen in the last couple of seconds."""

    best, when = _peaks.get(name, (0.0, 0.0))

    if value >= best or moment - when > PEAK_HOLD:
        _peaks[name] = (value, moment)
        return value

    return best


def reset():
    _state.clear()
    _debug.clear()
    _peaks.clear()


def debug_state():
    """Live values for the tuning overlay."""

    return dict(_debug)


def measure(times, values):
    """Summarise one signal over the window: (change, speed, directness).

    Directness compares where the signal ended up against how far it
    travelled getting there.  A clean movement scores near 1; a hand
    shaking between two positions covers ground and arrives nowhere, so it
    scores low however fast it shakes.  Counting which frames moved the
    right way is not enough -- alternating samples can reach two thirds
    agreement by luck, which is exactly how a wobble used to fire a swipe.

    Speed is measured across the movement rather than across the window.
    The window is a fixed half second and a gesture is usually shorter, so
    dividing by the whole of it charges the movement for the time the hand
    spent sitting still beforehand -- enough, measured, to drag a brisk
    turn down to exactly the threshold and lose it.
    """

    change = values[-1] - values[0]

    steps = [
        values[i + 1] - values[i]
        for i in range(len(values) - 1)
    ]

    path = sum(abs(step) for step in steps)

    directness = abs(change) / path if path > 0 else 0.0

    # Trim the still parts off each end.  A hand that never moved has
    # nothing to trim, so it keeps the whole window and stays slow.
    edge = abs(change) * 0.1
    first, last = 0, len(values) - 1

    for i, value in enumerate(values):
        if abs(value - values[0]) > edge:
            first = max(0, i - 1)
            break

    for i in range(len(values) - 1, -1, -1):
        if abs(values[-1] - values[i]) > edge:
            last = min(len(values) - 1, i + 1)
            break

    elapsed = times[last] - times[first]

    speed = abs(change) / elapsed if elapsed > 0 else 0.0

    return change, speed, directness


def _height(window, screen, mean_scale, moment):
    """How far the hand is above or below where it is resting, and how fast.

    Wherever it is held still becomes the height to judge against.  A fixed
    starting point does not work: a hand comes into view from below, so the
    pose is first seen near the bottom of its range, and from there up is
    easy while down would mean leaving the picture.

    Settling somewhere takes a full second, because there is a pause at the
    top of every raise -- and a shorter dwell made the raised position the
    new resting height, so putting the hand back down counted as lowering
    it.
    """

    if _state.neutral_y is None:
        # The middle of the first few, not the first.  A badly read frame
        # is most likely at the very moment this is taken -- losing the
        # pose is what resets it, and a frame bad enough to lose the pose
        # is a frame whose position is not to be trusted either.  Taken
        # from one sample, that frame becomes the height every later
        # raise is measured against, and the volume moves on its own.
        first = [sample[1]["y"] for sample in window[:5]]
        _state.neutral_y = sorted(first)[len(first) // 2]

    latest = window[-3:]

    if len(latest) >= 2:
        span = latest[-1][0] - latest[0][0]
        heights = [sample[1]["y"] for sample in latest]
        drift = (max(heights) - min(heights)) / mean_scale

        if span > 0 and drift / span < LIFT_STILL:
            if _state.still_since is None:
                _state.still_since = moment

            if moment - _state.still_since >= REST_DWELL:
                _state.neutral_y = heights[-1]
        else:
            _state.still_since = None

    lift = (_state.neutral_y - screen[hand_state.MIDDLE_MCP].y) / mean_scale

    # How briskly it got there, and how directly.  Position alone would
    # let a hand lowered slowly to the desk turn the volume down on its
    # way; speed alone lets a single badly-read frame do it, since a hand
    # that appears to jump and come back is briefly moving very fast
    # indeed.  The turn has been asked for directness all along and this
    # was not, which is why a lost frame could raise the volume.
    _, speed, agree = measure(
        [sample[0] for sample in window],
        [-sample[1]["y"] / mean_scale for sample in window])

    return lift, speed, agree


def _neutral_aim(aim, turn, moment):
    """Keep track of the aim a turn is judged against.

    Wherever the wrist is held becomes it, the same way the resting height
    works for the vertical gesture and for the same reason: there is no
    fixed neutral to assume, because it depends on how someone is sitting.

    Settling somewhere takes a full second -- and not at all while a turn
    is outstanding.  There is always a pause at the end of a turn, and
    adopting the angle it left the wrist at makes bringing the wrist back
    a turn in its own right, which is the very thing this is here to
    prevent.  A wrist held where a gesture put it has not chosen to rest
    there; that is the one place it is certain not to have.
    """

    if _state.neutral_aim is None:
        _state.neutral_aim = aim

    if _state.turned:
        _state.aim_still_since = None

        return

    if abs(turn) < SWIPE_QUIET:
        if _state.aim_still_since is None:
            _state.aim_still_since = moment

        if moment - _state.aim_still_since >= REST_DWELL:
            _state.neutral_aim = aim
    else:
        _state.aim_still_since = None


def detect_swipe(hand, handedness=None):
    """Detect a two-finger swipe.  ``handedness`` is accepted and ignored."""

    moment = now()

    # Which way the hand points is a question about its shape, so it comes
    # from the world landmarks; where it is in the frame is a question
    # about the screen, so that comes from the normalised ones.
    shape = hand_state.shape_of(hand)
    screen = hand_state.screen_of(hand)

    _state.history.add(
        moment,
        aim=hand_state.pointing_direction(hand),
        x=screen[hand_state.MIDDLE_MCP].x,
        y=screen[hand_state.MIDDLE_MCP].y,
        scale=hand_state.hand_scale(screen)
    )

    kind = gestures.pose_kind(hand)

    _state.note_pose(moment, kind)

    armed = _state.is_armed(moment)

    _debug.update(
        pose=kind,
        armed=armed,
        aim=round(hand_state.pointing_direction(hand), 2),
        ext=hand_state.finger_scores(shape),
        reach=hand_state.reach_scores(screen),
        why=gestures.explain(hand),
        scale=round(hand_state.hand_scale(screen), 4),
        turn=0.0,
        slide=0.0,
        speed=0.0,
        agree=0.0
    )

    if not armed:
        return None

    # Only what happened after the pose settled.  Raising a hand into
    # frame swings this reading wildly as the fingers come into view, and
    # counting that was why simply showing two fingers fired a swipe.
    window = [sample for sample in _state.history.recent(moment)
              if sample[0] >= _state.armed_since]

    if len(window) < SWIPE_MIN_SAMPLES:
        return None

    if window[-1][0] - window[0][0] <= 0:
        return None

    times = [sample[0] for sample in window]

    # Turning the wrist: works with either pose.
    turn, turn_speed, turn_agree = measure(
        times, [sample[1]["aim"] for sample in window])

    mean_scale = sum(sample[1]["scale"] for sample in window) / len(window)

    lift, lift_speed, lift_agree = _height(window, screen, mean_scale, moment)

    # Bringing the hand back is the same movement as swiping the other
    # way, so after a swipe nothing counts until it has gone quiet.
    if _state.settling:
        if abs(turn) < SWIPE_QUIET:
            _state.settling = False

        _debug.update(turn=round(abs(turn), 2), settling=True)

        return None

    _debug.update(
        settling=False,
        turn=round(abs(turn), 2),
        lift=round(lift, 2),
        lift_speed=round(lift_speed, 2),
        lift_agree=round(lift_agree, 2),
        speed=round(turn_speed, 2),
        agree=round(turn_agree, 2),
        peak_turn=round(_peak("turn", abs(turn), moment), 2),
        peak_speed=round(_peak("speed", turn_speed, moment), 2),
        peak_lift_speed=round(_peak("lift_speed", lift_speed, moment), 2),
        peak_agree=round(_peak("agree", turn_agree, moment), 2),
    )

    aim = window[-1][1]["aim"]

    _neutral_aim(aim, turn, moment)

    # Back near the aim it started from: ready to count again, either way.
    #
    # The return stroke is forgotten as it lands, rather than merely
    # being allowed to end.  It is sitting in the window at full size at
    # the moment the flag clears, so clearing the flag and then asking
    # the usual question of the usual window is asking whether the return
    # was a turn -- and it was, so it fires.  Emptying the window is what
    # is wanted here and waiting for the hand to go quiet is not: quiet
    # never comes if the next gesture follows straight on, and that one
    # would be swallowed instead.  What is left of the return is by
    # definition small, since this is the moment it arrived.
    if _state.turned and abs(aim - _state.neutral_aim) < SWIPE_TURN * 0.5:
        _state.turned = False
        _state.history.clear()

        return None

    turned = (
        _state.armed_kind == gestures.POSE_TWO_FINGER
        and abs(turn) >= SWIPE_TURN
        and turn_speed >= SWIPE_TURN_SPEED
        and turn_agree >= SWIPE_CONSISTENCY
        # Turning back is the same movement as turning the other way, and
        # no measurement of the movement itself can separate them.  What
        # separates them is where it ends up: a deliberate turn leaves the
        # aim somewhere new, and coming back returns it to where it began.
        # So one turn is allowed per departure from the resting aim.
        and not _state.turned
    )

    if _state.is_cooling(moment):
        return None

    # Back near the resting height: ready to count again, either way.
    # Before the cooldown check, because the hand comes back during it.
    if abs(lift) < SWIPE_LIFT * 0.5:
        _state.raised = False

    raised = (
        abs(lift) >= SWIPE_LIFT
        and lift_speed >= SWIPE_LIFT_SPEED
        and lift_agree >= SWIPE_CONSISTENCY
        and not _state.raised
    )

    # Which of the two it was, asked of each in the same words: a
    # movement is itself if the other one is a small share of it.  Not an
    # absolute size, which is what let a raise carrying a large turn count
    # as a raise while a turn carrying a large lift was never asked at
    # all.  A movement that is half of each is a diagonal nobody meant,
    # and it now fires nothing rather than whichever was asked first.
    sideways = abs(turn) / SWIPE_TURN
    upright = abs(lift) / SWIPE_LIFT

    if turned and upright <= sideways * CROSSTALK:
        moving_right = turn > 0

        if not FRAME_IS_MIRRORED:
            moving_right = not moving_right

        _state.fired(moment)

        return "SWIPE_RIGHT" if moving_right else "SWIPE_LEFT"

    if raised and sideways <= upright * CROSSTALK:
        # Not fired(): that waits for the hand to stop, and a hand held up
        # is not going to.  Coming back down is what readies the next one.
        _state.raised = True
        _state.still_since = None
        _state.cooldown_until = moment + SWIPE_COOLDOWN

        # The same movement, and which hand made it says what it meant.
        # This is what measuring a pose and a movement apart was for: an
        # open hand raised needed no recording of its own, being a pose
        # already measured and a movement already measured.
        if _state.armed_kind == gestures.POSE_OPEN_PALM:
            return "PALM_UP" if lift > 0 else "PALM_DOWN"

        return "SWIPE_UP" if lift > 0 else "SWIPE_DOWN"

    return None
