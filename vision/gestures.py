"""Held gestures: what shape the hand is making right now.

    detect_gesture(landmarks, handedness) -> "OPEN_PALM" | "FIST" |
                                             "POINT" | "TWO_FINGER" |
                                             "UNKNOWN"

Which way the hand faces is not checked.  It used to be, and that quietly
ruled out the two most natural ways to make these: a punch shows the
camera its knuckles rather than the palm, and pointing at the lens turns
the hand edge-on.  Both were rejected before their fingers were ever
counted.
"""

from __future__ import annotations

from . import hand_state
from .state_machine import FingerMemory, GestureStabiliser

POSE_TWO_FINGER = "two_finger"  # index and middle out: seeking and volume
POSE_OPEN_PALM = "open_palm"    # every finger out: brightness

_stabiliser = GestureStabiliser()
_fingers = FingerMemory()
_folded = FingerMemory()


def reset():
    _stabiliser.clear()
    _fingers.clear()
    _folded.clear()


def observe(hand):
    """Take this frame's reading of every finger.  Once per frame.

    detect_gesture is where a frame arrives, so that is where this is
    done; everything else asks what has already been worked out.
    """

    spans = hand_state.finger_span(hand)

    _folded.update(spans,
                   hand_state.FOLDED_RATIOS or hand_state.FOLDED_RATIO)

    return _fingers.update(spans, hand_state.EXTENDED_RATIO)


def _fingers_out(hand):
    """Which fingers are out, steadied against a finger on the line.

    Every pose here is a statement about which fingers are out, so a
    single finger flickering is a pose flickering.  Read from the memory
    rather than measured afresh, so one near the threshold settles on an
    answer instead of alternating.
    """

    return _fingers.read(hand_state.finger_span(hand),
                         hand_state.EXTENDED_RATIO)


def _fingers_folded(hand):
    """Which fingers are deliberately down, on its own steadied memory.

    Not the complement of ``_fingers_out``: between out and folded lies
    the slack of a hand at rest, and a finger there is neither -- so a
    pattern may claim it as neither.
    """

    above = _folded.read(hand_state.finger_span(hand),
                         hand_state.FOLDED_RATIOS or hand_state.FOLDED_RATIO)

    return {name: not is_above for name, is_above in above.items()}


def _steady_spans(hand):
    """The finger readings a few frames have agreed on, or this frame's."""

    if len(_fingers.steady) >= len(hand_state.FINGERS):
        return _fingers.steady

    return hand_state.finger_span(hand)


def _at_rest(hand):
    """Whether the hand matches the calibrated resting signature.

    For the explanation only, no longer a gate.  It gated briefly, and
    on a hand whose rest is naturally open the ball around the signature
    swallowed the casual versions of real poses: a swipe pose with the
    spare fingers held lazily sat entirely inside it, and a palm shown
    at resting height read as nothing until the hand was first carried
    somewhere the signature did not reach.  The per-finger folded lines
    guard the patterns structurally; the open-hand line *is* the
    measured split between this user's rest and their palm; a second
    wall behind those was strangling three real poses to guard one
    corridor.
    """

    return hand_state.looks_at_rest(_steady_spans(hand))


def _open_enough(hand):
    """Whether every finger is properly out, on the steadied readings.

    The same question hand_state.is_open asks, of the same numbers, but
    of the ones a few frames have agreed on rather than of this frame
    alone.  It is the strictest test in the file -- every finger at once,
    and properly straight rather than merely not curled -- so it is the
    one a single bad reading is most likely to overturn, and it was the
    only finger test still being asked afresh every frame.

    An open hand lowered is where that showed: the hand relaxes as it
    comes down, one finger is misread, and the pose disappears in the
    middle of the movement it was arming.
    """

    steady = _fingers.steady

    if len(steady) < len(hand_state.FINGERS):
        return hand_state.is_open(hand_state.shape_of(hand))

    return all(span > hand_state.OPEN_RATIO for span in steady.values())


def classify(hand, handedness=None):
    """The shape this single frame shows, before stabilising."""

    screen = hand_state.screen_of(hand)
    handedness = getattr(hand, "handedness", handedness)

    # A fist and an open hand are not read from behind: the back of a hand
    # is what the camera sees of someone typing or resting their hand on
    # the desk, and that should ask for nothing.  Side-on is fine -- a
    # fist from the side is still a fist -- so what is ruled out is
    # looking at the back of it, rather than anything short of a square
    # palm.  The two-finger poses are exempt entirely: the swipe can be
    # made with the fingers aimed at the lens, where the palm faces
    # neither way.
    from_behind = hand_state.is_back_of_hand(screen, handedness)

    # The fist first: every finger shut is the least ambiguous thing a
    # hand can be, so nothing else has to work around it.
    if hand_state.is_clenched(hand):
        return "UNKNOWN" if from_behind else "FIST"

    fingers = _fingers_out(hand)
    folded = _fingers_folded(hand)
    extended = sum(fingers.values())

    if extended == 0:
        # Not shut, but nothing straight either: a hand at rest.
        return "UNKNOWN"

    if (
        fingers["index"]
        and folded["middle"]
        and folded["ring"]
        and folded["pinky"]
    ):
        return "POINT"

    if _two_up(fingers, _steady_spans(hand)):
        return "TWO_FINGER"

    if extended == 4:
        # Properly open, not merely not-closed: a hand at rest has fingers
        # straighter than a fist and slacker than a spread hand, and it was
        # landing here.  And facing, not merely not-from-behind: an open
        # hand side-on is a hand reaching past the camera, not a palm
        # shown to it.
        if not hand_state.is_facing_palm(screen, handedness):
            return "UNKNOWN"

        if not _open_enough(hand):
            return "UNKNOWN"

        return "OPEN_PALM"

    return "UNKNOWN"


def explain(hand, handedness=None):
    """Why this frame reads as it does, in a few words.

    Written because "it does not recognise my fist" is not something a
    threshold can be adjusted from, and three attempts at guessing which
    one was wrong all missed.  This says which test refused, and what it
    measured, so the answer is read rather than deduced.
    """

    screen = hand_state.screen_of(hand)
    shape = hand_state.shape_of(hand)
    handedness = getattr(hand, "handedness", handedness)

    fingers = _fingers_out(hand)
    out = [name for name, is_out in fingers.items() if is_out]

    reaches = hand_state.reach_scores(screen)

    if hand_state.is_back_of_hand(screen, handedness):
        return "back of hand -- fist and open hand are not read from behind"

    if hand_state.is_clenched(hand):
        return "shut -> FIST"

    closed = [name for name, reach in reaches.items()
              if reach < hand_state.FIST_REACH]

    if not out:
        return (f"half closed: {len(closed)} of 4 fingers in, need the index "
                f"and two others for a fist")

    if len(out) == 4 and not hand_state.is_facing_palm(screen, handedness):
        strength = hand_state.palm_facing_strength(screen, handedness)
        return (f"not enough palm in view: facing {strength:+.2f}, the "
                f"pose is believed above {hand_state.FACING_CERTAINTY}")

    if fingers["index"] and fingers["middle"] and len(out) == 2:
        # The two-finger pose is near; say which down-finger refused.
        spans = _steady_spans(hand)
        drop = (min(spans.get("index", 0), spans.get("middle", 0))
                - spans.get("pinky", 1))

        if drop < hand_state.TWO_CLIFF:
            return (f"almost the swipe pose: the pinky sits "
                    f"{max(drop, 0):.2f} below the up fingers, and the "
                    f"pose begins at {hand_state.TWO_CLIFF:.2f} -- drop "
                    f"it a little further")

    if fingers["index"] and fingers["middle"] and fingers["ring"] \
            and not fingers["pinky"]:
        return ("almost the swipe pose: the ring finger is still out --"
                " let it drop")

    if len(out) == 4 and not _open_enough(hand):
        weakest = min(hand_state.finger_scores(shape).values())
        return (f"open but slack: weakest finger {weakest:.2f}, "
                f"needs {hand_state.OPEN_RATIO} to count as an open hand")

    if _at_rest(hand):
        return (f"fingers out: {', '.join(out)} -- but this matches your "
                f"hand at rest")

    return f"fingers out: {', '.join(out) or 'none'}"


def detect_gesture(hand, handedness=None):
    """The settled gesture, once it has held for a few frames.

    ``handedness`` is needed to tell a palm from the back of a hand: the
    two are mirror images, so which hand it is decides which is which.
    It is read from the hand itself when there is one.

    A user's taught gestures are consulted here, and only here, and only
    once the built-in classifier has already returned UNKNOWN.  That
    ordering is the whole of their isolation: if a built-in recognises
    the shape, the custom matcher never runs this frame, so no taught
    gesture can change what a shipped one decides -- and the settled
    answer, custom or built-in, still passes through the one stabiliser.
    """

    observe(hand)

    raw = classify(hand, handedness)

    if raw == "UNKNOWN":
        from . import custom

        matched = custom.match(hand_state.finger_span(hand))

        if matched is not None:
            raw = matched

    return _stabiliser.update(raw)


def _two_up(fingers, spans):
    """Index and middle out, the ring held short of out, the pinky
    folded on purpose.

    Exactly two, not "at most one of the others is wrong".  It was the
    looser rule briefly, because the folded fingers are the ones a camera
    reads worst and the pose rested on both of them being right at once.
    But the looser rule cannot tell three fingers from two with a misread
    ring finger -- they read identically -- and lumping them together
    spends a pose that is worth keeping to buy steadiness that can be got
    another way.

    The steadiness comes from time instead: a finger answers from several
    readings rather than one, so a misread has to persist to be believed,
    while three fingers held up persists by definition.

    The two down-fingers are asked different questions, because hands
    hold them differently.  Plenty of people make the pose with the
    ring only half-held, so the ring answers the plain question: not
    out.  The pinky is the deliberate one, and deliberate is a shape,
    not a height: dropped off the cliff, sitting well below the hand's
    own up-fingers in the same frame.  Rest is a gradient -- each
    finger a small step below the last -- and the pose is a cliff, and
    that holds at every depth of rest at once, because a resting hand
    slides as a unit and its little steps slide with it.  Measured on
    the hand that forced this: pinky held down and pinky at rest read a
    hundredth apart in *height*, where no recorded line can sit, while
    the rest-step below the up-fingers was 0.16 against a held-down
    drop of 0.30.

    Not "cliff or tucked": the absolute fold line stays for POINT, but
    here it was the remaining hole -- a deep enough rest walks the
    pinky under any fixed line, gradient intact, and only the cliff
    knows the difference.
    """

    if not (fingers["index"] and fingers["middle"] and not fingers["ring"]):
        return False

    up_floor = min(spans.get("index", 0.0), spans.get("middle", 0.0))

    return (not fingers["pinky"]
            and up_floor - spans.get("pinky", 1.0) >= hand_state.TWO_CLIFF)


def pose_kind(hand):
    """Which pose a movement could be made from, or None.

    Two fingers, held up or aimed at the camera; which way they face does
    not matter.

    There was a pinch here too, for volume, and it went: defined by where
    the thumb is, it was mistaken in turn for a fist, an open hand and two
    fingers -- because the thumb is the landmark a camera loses first,
    whenever the hand turns or closes.  Volume is on the keyboard, where
    brightness already was.
    """

    fingers = _fingers_out(hand)

    if _two_up(fingers, _steady_spans(hand)):
        return POSE_TWO_FINGER

    if not all(fingers.values()):
        return None

    # An open hand has to be properly open, and has to be a palm shown
    # to the camera.  The back of a hand with the fingers straight is
    # what the camera sees of someone reaching for something, and a hand
    # edge-on is one reaching past it -- and reaching moves the hand,
    # which is the whole of what this pose is then asked about.
    screen = hand_state.screen_of(hand)

    if not hand_state.is_facing_palm(screen, getattr(hand, "handedness", None)):
        return None

    if not _open_enough(hand):
        return None

    return POSE_OPEN_PALM
