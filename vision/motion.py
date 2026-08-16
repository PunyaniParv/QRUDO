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

import math

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

#: A movement must also clear this many times the reading's own jitter,
#: measured as the median step between consecutive frames of the window.
#: On a good camera this floor sits far below the calibrated bar and
#: decides nothing -- the bar is 8x to 33x the wander of a still hand,
#: measured across the working range.  On a bad one -- dim light, a poor
#: webcam, a hand at the edge of readability -- the jitter grows to meet
#: the bar, and this floor rises with it, so the failure is "gestures
#: need to be bigger" rather than "the volume moves on its own".
#:
#: It can only ever raise the bar, never lower it, which is what makes it
#: safe to apply everywhere.  Four, because a clean movement spanning n
#: frames has a median step of about 1/(n-1) of itself: at four, even a
#: six-frame flick clears its own floor with a fifth to spare, while a
#: still hand's floor overtakes the bar once the camera gets about twice
#: as bad as the worst distance measured today.
NOISE_MARGIN = 4

#: How many frames a movement must be visible across before it is
#: believed.  A hand cannot cross a gesture's worth of ground between one
#: frame and the next; a misread landmark can, and did -- it arrives as a
#: single enormous step, which is fast by any measure and perfectly
#: direct, because two points cannot disagree about a direction.  Three
#: samples is a tenth of a second at thirty frames: well inside any real
#: gesture, and outside anything one bad frame can fake.
MOVING_SAMPLES = 3
ARM_HOLD = 0.60           # how long the pose keeps a swipe allowed
SWIPE_COOLDOWN = 1.00
SWIPE_QUIET = 0.12        # how still the hand must go before the next one
QUIET_SPAN = 0.20         # and for how long: stillness is judged on this
                          # much recent history, not the whole window

#: The speed a turn must also show in the sine of the aim.  The aim is
#: an asin, and an asin steepens toward its ends: near the clamp a
#: slowly turning wrist reads as a fast one, because the steepness --
#: not the wrist -- supplies the speed.  The sine is the reading before
#: that amplification, so the same movement is asked for speed in both,
#: and a slow turn near the edge races in one and crawls in the other.
#:
#: An absolute floor, not a share of the calibrated bar: the question it
#: answers -- did the wrist move, or only the reading? -- is about the
#: geometry of the measurement, which no calibration changes.  Tied to
#: the bar, a vigorous calibration raised it past what an honest flick
#: from a resting-tilted hand can geometrically produce.  Measured: an
#: honest flick from a tilted rest shows 1.1, an unhurried one 2.5, and
#: a slow wrist releasing from the clamp 0.46.  This floor only judges
#: movements that began from a recent stop; the ones that did not are
#: refused above for that, however the edge inflates their reading.
LEAN_SPEED_FLOOR = 0.60
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
#: Both are asked, and they are asked different numbers, because the two
#: movements do not bleed into each other equally.  Turning a wrist
#: through fifty degrees barely moves the hand up or down.  Dropping a
#: hand rotates the wrist a good deal -- fifteen degrees of it is enough
#: to make the turn two thirds the size of the drop -- and that rotation
#: is not a gesture, it is what an arm does when it comes down.
#:
#: One shared number cannot say that, and the one that was here refused a
#: lower made in the ordinary way.
#:
#: Calibration measures both, from how much of the wrong movement your own
#: turns and raises carry.  These are the fallbacks.
#: Measured on the synthetic hands: a lower carrying up to twenty degrees
#: of incidental wrist roll scores 0.00 to 0.67, and a diagonal anyone
#: meant scores 1.08 or more.  The line goes between them.
CROSSTALK_TURN = 0.60     # lift a turn may carry
CROSSTALK_LIFT = 0.80     # turn a raise may carry

#: The units the crosstalk shares are measured in: the shipped bars,
#: frozen.  Calibration moves the bars; it must not move the yardstick
#: the "which movement was that" question is asked with, or the
#: allowances above -- measured against these sizes -- quietly change
#: meaning with every recalibration.
CROSSTALK_UNIT_TURN = 0.30
CROSSTALK_UNIT_LIFT = 0.60

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


def set_cooldown(seconds):
    """The configured cooldown, delivered to both places that hold it.

    The turn path asks ``_state``, which keeps the value it was built
    with; the lift path reads the module global.  Assigning only the
    global is how left and right once kept the default while up and
    down obeyed the setting.
    """

    global SWIPE_COOLDOWN

    SWIPE_COOLDOWN = seconds
    _state.cooldown = seconds


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

    # How far it ended up, against how far it travelled getting there.  A
    # clean movement scores near 1; a hand that goes out and comes most
    # of the way back covers ground and arrives nowhere, so it scores low
    # however fast it went.
    #
    # Measured over a handful of stretches rather than frame by frame.
    # Frame by frame, every frame's error goes into the sum, so the
    # measure grows with the length of the window and with how noisy the
    # reading is -- and a small movement seen from across a room, where
    # the error is a large share of each step, comes out looking like a
    # hand shaking rather than a hand moving.  It was the first thing to
    # fail at a distance, before the pose itself did.
    #
    # Averaging each stretch first cancels most of that, and costs
    # nothing real: a hand cannot change direction inside a stretch this
    # short, so anything that does is the reading and not the hand.
    # Both halves come off the same smoothed series, or the ratio is
    # comparing the whole of the movement against part of its path and
    # can exceed one, which passes everything.
    means = _stretches(values)
    walked = sum(abs(means[i + 1] - means[i]) for i in range(len(means) - 1))
    arrived = abs(means[-1] - means[0]) if means else 0.0

    directness = arrived / walked if walked > 0 else 0.0

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

    # Too few frames to be a hand moving.  Reported as no speed and no
    # directness rather than as a very fast, very clean gesture, which is
    # what a single bad frame otherwise looks like to both measures.
    if last - first + 1 < MOVING_SAMPLES:
        return change, 0.0, 0.0

    elapsed = times[last] - times[first]

    speed = abs(change) / elapsed if elapsed > 0 else 0.0

    return change, speed, directness


#: How many stretches a movement is measured across.  Four: enough to
#: catch a hand that turned back on itself, few enough that each is
#: averaged over several frames.
STRETCHES = 4


def _median_step(values):
    """The typical frame-to-frame step: the reading's own jitter.

    The median, not the mean: during a real movement a few steps are the
    movement, and the median ignores them so long as most of the window
    is quiet -- and a movement's own steps are each a small fraction of
    the whole, so even a window that is all movement estimates a floor
    well under the movement itself.
    """

    if len(values) < 3:
        return 0.0

    steps = sorted(abs(values[i + 1] - values[i])
                   for i in range(len(values) - 1))

    return steps[len(steps) // 2]


def _stretches(values):
    """Where the signal was, on average, across each stretch of the window.

    A reading that is wrong for a frame moves its stretch's average by a
    fraction of its error, instead of contributing the whole of it twice
    to a frame-by-frame path.
    """

    if len(values) < STRETCHES:
        return list(values)

    size = len(values) / STRETCHES
    means = []

    for part in range(STRETCHES):
        first = int(part * size)
        chunk = values[first:max(int((part + 1) * size), first + 1)]
        means.append(sum(chunk) / len(chunk))

    return means


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

    lift = (_state.neutral_y - screen[hand_state.WRIST].y) / mean_scale

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
        # The wrist, because it is the centre the wrist turns around: a
        # knuckle swings up or down with every roll, so measured there, a
        # turn carried a rise it never made -- in one direction only,
        # which read as one diagonal firing and its mirror staying quiet.
        y=screen[hand_state.WRIST].y,
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

    # Whether the hand is still right now: no more spread than jitter
    # across the last beat of aim readings.  Stopped is a property of
    # that beat, not of the whole window -- asking the full window to
    # drain forced a wait as long as the window itself.  The beat must
    # actually be covered: two samples a step apart cannot tell a rest
    # from a crawl, and on a slow camera a wrist creeping less than the
    # jitter bar per frame read as a hand at rest the whole way round.
    # Read from the raw history, not the armed window: being still is a
    # fact about the hand, not the pose, and the stillness that counts
    # mostly happens before the pose has finished arming.
    recent = [sample for sample in _state.history.recent(moment)
              if moment - sample[0] <= QUIET_SPAN]
    spread = (max(s[1]["aim"] for s in recent)
              - min(s[1]["aim"] for s in recent)) if recent else 0.0
    resting = (len(recent) >= 2
               and recent[-1][0] - recent[0][0] >= QUIET_SPAN * 0.7
               and spread < SWIPE_QUIET)

    if resting:
        _state.aim_rested_at = moment

    # Bringing the hand back is the same movement as swiping the other
    # way, so after a swipe nothing counts until it has gone quiet.
    if _state.settling:
        if resting:
            # A correction begun before the old window-drain test cleared
            # was not delayed but swallowed -- with the swallowed swipe
            # then keeping the window loud, so retrying faster made it
            # worse.  The history is left alone: wiping it belongs to
            # the ``turned`` reset below, which needs these samples to
            # see the aim come home -- and the return stroke stays
            # refused either way, because ``turned`` holds until it has.
            _state.settling = False

        _debug.update(turn=round(abs(turn), 2), settling=_state.settling)

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

    # The reading's own jitter, from the same window the movement is
    # measured on.  See NOISE_MARGIN.
    aim_floor = NOISE_MARGIN * _median_step(
        [sample[1]["aim"] for sample in window])
    lift_floor = NOISE_MARGIN * _median_step(
        [sample[1]["y"] / mean_scale for sample in window])

    _debug.update(noise=round(max(aim_floor, lift_floor), 2))

    _, lean_speed, _ = measure(
        times, [math.sin(sample[1]["aim"]) for sample in window])

    turned = (
        _state.armed_kind == gestures.POSE_TWO_FINGER
        and abs(turn) >= max(SWIPE_TURN, aim_floor)
        and turn_speed >= SWIPE_TURN_SPEED
        # In the sine as well as the angle -- see LEAN_SPEED_FLOOR.
        and lean_speed >= LEAN_SPEED_FLOOR
        and turn_agree >= SWIPE_CONSISTENCY
        # A turn is a departure from a stop.  The window is blind past
        # half a second, so a hand that has been turning for four reads,
        # in any one window, exactly like a flick.  What tells them apart
        # is not in the window at all: the flick's window sits a beat
        # from stillness, and the crawl's sits seconds from it.
        and moment - _state.aim_rested_at <= SWIPE_WINDOW + QUIET_SPAN
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
        abs(lift) >= max(SWIPE_LIFT, lift_floor)
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
    #
    # The shares are normalised by fixed sizes, not the calibrated bars.
    # Which movement a gesture is -- how much turn a drop carries, as a
    # share of itself -- is a fact about arms and wrists that no
    # calibration changes.  Divided by the bars, a calibration that
    # tightened the turn bar and relaxed the lift bar silently re-scored
    # every drop as more turn and less drop, and a lower with the
    # ordinary roll in it had to go far past the raise's distance before
    # it counted.
    sideways = abs(turn) / CROSSTALK_UNIT_TURN
    upright = abs(lift) / CROSSTALK_UNIT_LIFT

    if turned and upright <= sideways * CROSSTALK_TURN:
        moving_right = turn > 0

        if not FRAME_IS_MIRRORED:
            moving_right = not moving_right

        _state.fired(moment)

        return "SWIPE_RIGHT" if moving_right else "SWIPE_LEFT"

    if raised and sideways <= upright * CROSSTALK_LIFT:
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
