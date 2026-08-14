"""Swipes: what the hand did over the last half second.

    detect_swipe(landmarks, handedness) -> "SWIPE_LEFT" | "SWIPE_RIGHT" | None

The gesture is two fingers facing the camera -- an ordinary peace sign --
with the wrist rotated left or right, the fingers staying toward the lens
throughout.

That pose is deliberate.  Everything stays in the plane of the image,
which is where MediaPipe is by far the most accurate, so nothing about
recognising it leans on depth.  Aiming the fingers at the camera instead
puts the whole gesture along the one axis a single camera guesses at.

Also accepted, since they cost nothing and are what some people reach for:

  * the same rotation with the fingers aimed at the camera
  * moving the peace sign bodily across, rather than rotating it
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

SWIPE_SLIDE = 0.90        # palm widths the hand must cover
SWIPE_SLIDE_SPEED = 1.60  # palm widths per second

#: Both scripts mirror the frame before detection, so x increases to the
#: user's right and a rightward swipe is a rise in x.
FRAME_IS_MIRRORED = True

_state = SwipeState(ARM_HOLD, SWIPE_COOLDOWN, SWIPE_WINDOW)

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


def measure(values, elapsed):
    """Summarise one signal over the window: (change, speed, directness).

    Directness compares where the signal ended up against how far it
    travelled getting there.  A clean movement scores near 1; a hand
    shaking between two positions covers ground and arrives nowhere, so it
    scores low however fast it shakes.  Counting which frames moved the
    right way is not enough -- alternating samples can reach two thirds
    agreement by luck, which is exactly how a wobble used to fire a swipe.
    """

    change = values[-1] - values[0]

    steps = [
        values[i + 1] - values[i]
        for i in range(len(values) - 1)
    ]

    path = sum(abs(step) for step in steps)

    directness = abs(change) / path if path > 0 else 0.0

    return change, abs(change) / elapsed, directness


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
        scale=hand_state.hand_scale(screen)
    )

    kind = gestures.two_finger_pose_kind(hand)

    if kind is not None:
        _state.arm(moment, kind)

    armed = _state.is_armed(moment)

    _debug.update(
        pose=kind,
        armed=armed,
        aim=round(hand_state.pointing_direction(hand), 2),
        ext=hand_state.finger_scores(shape),
        turn=0.0,
        slide=0.0,
        speed=0.0,
        agree=0.0
    )

    if _state.is_cooling(moment) or not armed:
        return None

    window = _state.history.recent(moment)

    if len(window) < SWIPE_MIN_SAMPLES:
        return None

    elapsed = window[-1][0] - window[0][0]

    if elapsed <= 0:
        return None

    # Turning the wrist: works with either pose.
    turn, turn_speed, turn_agree = measure(
        [sample[1]["aim"] for sample in window], elapsed)

    # Moving the hand across: only the peace sign, and measured in palm
    # widths so distance from the camera does not matter.
    mean_scale = sum(sample[1]["scale"] for sample in window) / len(window)

    slide, slide_speed, slide_agree = measure(
        [sample[1]["x"] / mean_scale for sample in window], elapsed)

    _debug.update(
        turn=round(abs(turn), 2),
        slide=round(abs(slide), 2),
        speed=round(turn_speed, 2),
        agree=round(turn_agree, 2),
        peak_turn=round(_peak("turn", abs(turn), moment), 2),
        peak_slide=round(_peak("slide", abs(slide), moment), 2),
        peak_speed=round(_peak("speed", turn_speed, moment), 2),
        peak_agree=round(_peak("agree", turn_agree, moment), 2),
    )

    turned = (
        abs(turn) >= SWIPE_TURN
        and turn_speed >= SWIPE_TURN_SPEED
        and turn_agree >= SWIPE_CONSISTENCY
    )

    slid = (
        _state.armed_kind == gestures.POSE_PEACE
        and abs(slide) >= SWIPE_SLIDE
        and slide_speed >= SWIPE_SLIDE_SPEED
        and slide_agree >= SWIPE_CONSISTENCY
    )

    if not turned and not slid:
        return None

    moving_right = (turn if turned else slide) > 0

    if not FRAME_IS_MIRRORED:
        moving_right = not moving_right

    _state.fired(moment)

    return "SWIPE_RIGHT" if moving_right else "SWIPE_LEFT"
