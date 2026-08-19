"""What a hand is doing, measured from its landmarks.

Everything here answers a question about a single frame -- how straight a
finger is, which way it points, how big the hand looks.  Nothing here
remembers anything; that is state_machine.py's job.

The measurements are deliberately built from 3D distances and directions
rather than from how the hand projects onto the screen.  Turning a hand
side-on changes every screen measurement -- the knuckle row collapses to
almost nothing -- while the hand itself has not changed at all.  Anything
scaled by the screen becomes meaningless at that angle, which is what used
to make a fist in profile go unrecognised.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------
# Landmarks
# ---------------------------------------------------------

WRIST = 0
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

# name -> (tip, pip, mcp)
FINGERS = {
    "index": (8, 6, 5),
    "middle": (12, 10, 9),
    "ring": (16, 14, 13),
    "pinky": (20, 18, 17),
}


# ---------------------------------------------------------
# Tuning
# ---------------------------------------------------------

#: A finger is out when the straight line from its knuckle to its tip is
#: nearly as long as the finger itself.  Curling a finger shortens that
#: line; turning the hand does not.
EXTENDED_RATIO = 0.82

#: A finger is deliberately down below this.  A different line from the
#: one above, because between them lies the hand at rest: its fingers
#: curl in order, index straightest and pinky deepest, so wherever the
#: out-line cuts, some depth of slack leaves exactly the index above it
#: -- and "down" meaning merely "not out" then read a resting hand as
#: POINT, one slacker finger from the swipe pose.  A pattern's down
#: fingers must be folded on purpose, which a resting finger never is.
#:
#: Shipped equal to EXTENDED_RATIO -- one line, exactly the old
#: behaviour -- because some hands hold the swipe pose's spare fingers
#: nearly straight, and a guessed gap would cost them the pose.
#: Calibration lowers it only when this user's own recordings show the
#: held-down fingers and the resting ones actually separate.
FOLDED_RATIO = 0.82

#: The same line drawn per finger, when a calibration has measured them
#: apart, or None for the single line above.  Fingers do not rest
#: equally: a ring finger at rest sits a good deal straighter than a
#: pinky at rest, so the one line either refuses a casually held-down
#: ring or lets a resting pinky count as folded -- it cannot avoid both.
FOLDED_RATIOS = None

#: The other way a finger can be deliberately down: dropped this far
#: below the weaker of the two up-fingers, in the same frame.  Measured
#: live, the absolute lines ran out of room -- one hand's pinky read
#: 0.72 held down in the pose and 0.73 at rest, a hundredth apart, and
#: no recorded bar can sit in a hundredth.  What separates the two is
#: shape, not height: rest is a gradient, each finger a small step below
#: the last (that pinky rested 0.16 under its own up-fingers), and the
#: pose is a cliff (the same pinky held down sat 0.30 under them).  A
#: relative measure is also immune to the posed-versus-casual disease
#: that broke every absolute bar here, because it never compares against
#: a recording -- only against the hand as it is this frame.
TWO_CLIFF = 0.25

#: Per-finger extension of this user's hand at rest, measured by the
#: calibration, or None before any has been.  A hand whose every finger
#: sits within REST_TOLERANCE of it is resting, whatever the lines above
#: would make of the pattern.
REST_SIGNATURE = None

#: How closely a live hand must match the signature to be called
#: resting.  Two sides bound it.  It must stay under the distance a
#: deliberate pose keeps from the rest -- an open palm sits about 0.20
#: from it at the pinky alone, so every real pose escapes on at least
#: one finger.  And it must reach *past* the open-hand line: the fate of
#: a resting hand rides on its slackest finger drifting over that line,
#: and at 0.09 the veto let go just before the line was crossed -- a
#: wall standing an inch inside the door it was built to guard.
REST_TOLERANCE = 0.12

#: A fist is judged by where the fingertips end up rather than by how
#: bent each finger is.  How far a tip sits from the wrist, against how
#: far its knuckle does: shut brings the tips back level with the
#: knuckles, a hand merely resting leaves them well beyond, and an open
#: hand further still.  Those separate cleanly, where "how bent is this
#: finger" does not -- and the knuckles and wrist stay well tracked even
#: when a closed hand hides the fingers themselves.
FIST_REACH = 1.15

#: The other way of asking the same question: how curled the fingers
#: themselves are.  A fist answers to both, and it takes only one.
#:
#: Reach alone was the whole test, and reach alone is a narrow thing to
#: rest a gesture on.  It is measured from the fingertips, which are the
#: landmarks a closed hand hides -- and it puts a tight fist and a loose
#: one far apart, though both are plainly fists, because how far the tips
#: end up depends on how hard the hand is squeezed.  Curl barely moves
#: between the two: a folded finger is folded.
FIST_CURL = 0.55

#: How close the thumb and fingertip must come to count as a pinch, as a
#: fraction of the palm's length.  Measured on the synthetic hands:
#: pinched 0.24, a closed fist 0.41, and anything open 0.81.  Set nearer
#: the open end than the middle, because people do not reliably touch --
#: a pinch held a centimetre apart is still a pinch, and the fist test
#: behind this one is what keeps a closed hand out.
PINCH_GAP = 0.45


#: How much further the middle finger must reach than the index, for a
#: pinch made with the other fingers held out.  A peace sign holds both
#: level; pinching pulls the index down to the thumb.
PINCH_BEHIND = 0.15

#: An open hand has to be properly open, not merely not-closed.  A hand at
#: rest has fingers straighter than a fist and slacker than a spread hand,
#: and with only one line to fall on it landed on "open" -- so a hand
#: doing nothing asked for something.  This is the second line.
OPEN_RATIO = 0.90

#: A palm seen edge-on gives a normal with almost no depth component, and
#: its sign is then decided by noise.  Below this the answer is "cannot
#: tell" rather than a coin flip.
PALM_CERTAINTY = 0.12

#: How much palm must be in view before the open-palm pose is believed.
#: Deliberately half the certainty line above: the question is not "is
#: this squarely a palm" but "is there palm at all in view".  A palm
#: relaxed at resting height tilts back and shows the camera less of
#: itself than one held square for a prompt, and demanding the full
#: certainty of it in use refused the pose as people actually hold it --
#: brightness armed only from a hand carried high and squared.  A true
#: side profile still reads under a tenth of this and stays refused.
FACING_CERTAINTY = 0.06

#: Smallest a hand can look on screen and still be read, as a fraction of
#: the frame width.  Anything smaller is discarded before a gesture is
#: looked for at all, so this is the range limit and nothing else is.
#:
#: Measured at the size the camera now gives, with landmark error added:
#: every pose reads correctly every time down to 0.028, and at 0.022 the
#: worst of them -- an open hand, which asks the most -- is still right
#: nine times in ten.  Below that they fall away quickly.  So the line
#: goes at 0.022, which is a hand about two metres off on a laptop
#: webcam, against 0.035 for one at a metre and a third.
#:
#: Lowering it cannot cost anything close up.  It only ever decides which
#: hands are too small to bother with, so a hand near the camera is
#: unaffected by where it sits.
#:
#: It was 0.09 briefly, to cut down false positives, which quietly capped
#: the whole app at arm's length.
MIN_HAND_ON_SCREEN = 0.022

#: What the fraction above actually is: a pixel budget.  Landmark error
#: is roughly fixed in pixels, so what decides whether a hand can be
#: read is how many pixels of the original frame it covers -- a property
#: of the model, not of any camera.  The two range measurements in this
#: project agree on it: poses held to 0.022 of frame at 640 wide (14
#: pixels of hand), and range "roughly doubles" at 1760, which is the
#: same 14 pixels at a smaller fraction.  Given the frame width the
#: camera actually delivers, the range gate follows -- which is what
#: lets one hand profile travel between cameras with nothing to redo.
MIN_HAND_PIXELS = 14


def set_camera(width):
    """Derive the range gate from the camera actually in use.

    Called when a camera opens, with the width it really delivers (not
    the one asked for).  Wins over any stored value by arriving later,
    deliberately: the stored number was measured through the calibrating
    camera, and it is the one threshold in the file that describes the
    window rather than the hand.  Clamped so a strange reported width
    cannot open the gate to noise or close it to arm's length.
    """

    global MIN_HAND_ON_SCREEN

    if not width or width <= 0:
        return MIN_HAND_ON_SCREEN

    MIN_HAND_ON_SCREEN = min(0.10, max(0.008, MIN_HAND_PIXELS / width))

    return MIN_HAND_ON_SCREEN


# ---------------------------------------------------------
# Which landmarks to ask
# ---------------------------------------------------------

def shape_of(hand):
    """Landmarks for measuring what the hand is doing.

    MediaPipe returns two sets and they are not interchangeable.  The
    normalised set says where the hand is on screen, and its depth is only
    loosely scaled -- but a finger aimed at the lens has nearly all of its
    length in that depth, so measured there it comes out short, which reads
    as curled.  That is why the gun pose was mistaken for a fist and swipes
    never armed.

    The world set is in metres with real depth, which is what a question
    about shape needs.  Anything about position on screen must still use
    the normalised set: see screen_of.
    """

    world = getattr(hand, "world", None)

    if world:
        return world

    # A plain list of landmarks, as the tests provide.
    return getattr(hand, "landmarks", hand)


def screen_of(hand):
    """Landmarks for measuring where the hand is in the frame."""

    return getattr(hand, "landmarks", hand)


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

def distance(point1, point2):
    """3D distance between two landmarks."""

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )


def distance_2d(point1, point2):
    """Distance as it appears on screen, ignoring depth."""

    return math.hypot(
        point1.x - point2.x,
        point1.y - point2.y
    )


def hand_scale(hand_landmarks):
    """How big the hand looks on screen, for measuring movement across it.

    Only the sideways slide uses this, and a slide is a screen distance,
    so a screen measurement is the right one.  Nothing that has to survive
    the hand turning is scaled by it.

    Both the palm's width and its length are considered, because a turn
    that flattens one leaves the other.
    """

    return max(
        distance_2d(hand_landmarks[INDEX_MCP], hand_landmarks[PINKY_MCP]),
        distance_2d(hand_landmarks[WRIST], hand_landmarks[MIDDLE_MCP]),
        0.02
    )


#: The thumb tip landmark, for the thumb-gap measurement below.
THUMB_TIP = 4


def thumb_gap(hand):
    """How far the thumb tip sits from the index tip, in palm-lengths.

    Finger extension cannot tell a closed hole (thumb and index
    touching, an OK sign) from an open C (the same fingers curled but
    apart) -- the fingers are equally curled either way.  What separates
    them is this gap: near zero when they touch, wide when they do not.
    Adding it to a custom gesture's signature lets the two be told
    apart, which finger extension alone never could.

    Measured in 3D over palm length so it means the same at any distance
    or hand size.
    """

    shape = shape_of(hand)

    palm = distance(shape[WRIST], shape[MIDDLE_MCP]) or 0.01

    return distance(shape[THUMB_TIP], shape[INDEX_TIP]) / palm


#: A thumb reaching past the knuckle line by more than this, on a hand
#: whose fingers are all shut, is a thumbs-up -- and a thumbs-up is not
#: a fist.  A fist's thumb sits beside or across the fingers, at or
#: below the knuckles; only a deliberately raised thumb climbs past
#: them.
THUMBS_UP_RISE = 0.25

#: A thumb closer to the index knuckle than this, on a hand with all
#: four fingers straight, is tucked across the palm -- and four fingers
#: with the thumb tucked is not an open palm.  An open palm holds the
#: thumb out to the side, well clear of the knuckles.  Live hands put
#: an open thumb at 0.55-0.65 palms and a folded one well under 0.35;
#: the first line here (0.22) was calibrated on synthetic geometry and
#: let real folded thumbs through.
THUMB_TUCKED_SPREAD = 0.35

#: Below this hand size (as a fraction of frame width) the thumb is a
#: few pixels of guesswork, and the strict thumb tests above would be
#: refusing real gestures over noise.  They only ask their questions
#: of a hand close enough to answer honestly; far away, the four-finger
#: reading -- the one that holds at range -- stands alone, exactly as
#: it did before the thumb was consulted at all.
THUMB_READ_SCALE = 0.05


def thumb_rise(hand):
    """How far the thumb tip reaches past the knuckle line, along the
    hand's own up-axis, in palm-lengths.

    Positive is thumb-up territory; a thumb beside curled fingers or
    out to the side of a palm sits at or below zero.  Measured along
    the wrist-to-middle-knuckle axis so turning or tilting the whole
    hand does not change the answer.
    """

    shape = shape_of(hand)

    palm = distance(shape[WRIST], shape[MIDDLE_MCP]) or 0.01

    ux = (shape[MIDDLE_MCP].x - shape[WRIST].x) / palm
    uy = (shape[MIDDLE_MCP].y - shape[WRIST].y) / palm
    uz = (getattr(shape[MIDDLE_MCP], "z", 0.0)
          - getattr(shape[WRIST], "z", 0.0)) / palm

    dx = shape[THUMB_TIP].x - shape[INDEX_MCP].x
    dy = shape[THUMB_TIP].y - shape[INDEX_MCP].y
    dz = (getattr(shape[THUMB_TIP], "z", 0.0)
          - getattr(shape[INDEX_MCP], "z", 0.0))

    return (dx * ux + dy * uy + dz * uz) / palm


def thumb_spread(hand):
    """How far the thumb tip sits from the index knuckle, in
    palm-lengths: wide for a thumb held out, small for one tucked
    across the palm."""

    shape = shape_of(hand)

    palm = distance(shape[WRIST], shape[MIDDLE_MCP]) or 0.01

    return distance(shape[THUMB_TIP], shape[INDEX_MCP]) / palm


def thumb_readable(hand):
    """Whether the hand is near enough for the thumb tests to be
    honest rather than noise."""

    return hand_scale(screen_of(hand)) >= THUMB_READ_SCALE


def is_prominent(hand):
    """Whether the hand is close enough and whole enough to mean it."""

    return hand_scale(screen_of(hand)) >= MIN_HAND_ON_SCREEN


# ---------------------------------------------------------
# Palm orientation
# ---------------------------------------------------------

def palm_facing_strength(hand_landmarks, handedness):
    """How squarely the palm faces the camera, as a signed number.

    Positive means facing, negative means the back of the hand, and near
    zero means edge-on.  Nothing in the detectors depends on this any
    more: it is decisive for a hand held flat and meaningless otherwise,
    and putting it in front of the other tests is what once stopped a
    punch, a pointed finger and every swipe from registering.
    """

    wrist = hand_landmarks[WRIST]
    index_mcp = hand_landmarks[INDEX_MCP]
    pinky_mcp = hand_landmarks[PINKY_MCP]

    v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)
    v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y)

    normal_z = v1[0] * v2[1] - v1[1] * v2[0]

    # Scale out hand size so the number means the same at any distance.
    normal_z /= hand_scale(hand_landmarks) ** 2

    if handedness == "Right":
        return -normal_z

    return normal_z


def is_back_of_hand(hand_landmarks, handedness):
    """True only when the camera is clearly behind the hand.

    Separate from "is the palm facing us", and the useful one of the two
    for most poses.  A hand turned side-on is neither -- and a fist from
    the side is still a fist -- so the test that matters is whether we
    are looking at the back of it, which is what a hand on a keyboard
    shows.
    """

    return palm_facing_strength(
        hand_landmarks,
        handedness
    ) < -PALM_CERTAINTY


def is_facing_palm(hand_landmarks, handedness):
    """True only when the palm is confidently toward the camera.

    The stricter question, asked of the open palm alone.  The other
    poses are shapes -- a fist from the side is still a fist -- but an
    open palm shown side-on is not somebody showing their palm, it is a
    hand reaching past the camera with the fingers straight, and it was
    read as the palm anyway.  So the pose is only believed where the
    camera can actually tell it is looking at one: edge-on, the reading
    is noise, and noise is not a gesture.
    """

    return palm_facing_strength(
        hand_landmarks,
        handedness
    ) > FACING_CERTAINTY


# ---------------------------------------------------------
# Fingers
# ---------------------------------------------------------

def finger_extension(hand_landmarks, tip, pip, mcp):
    """How straight a finger is: about 1.0 straight, about 0.5 curled.

    The straight-line distance from knuckle to tip, divided by the length
    of the finger itself.  Both are measured in 3D, which is what makes
    this hold from any viewpoint: turning the hand changes how long the
    finger *looks*, but not how long it *is*.

    It also shrugs off MediaPipe's shaky depth.  If every z is off by the
    same factor, both distances are off by roughly that factor too, and
    the ratio between them barely moves.
    """

    dip = pip + 1

    span = distance(hand_landmarks[mcp], hand_landmarks[tip])

    length = (
        distance(hand_landmarks[mcp], hand_landmarks[pip]) +
        distance(hand_landmarks[pip], hand_landmarks[dip]) +
        distance(hand_landmarks[dip], hand_landmarks[tip])
    )

    if length == 0:
        return 0.0

    return span / length


def finger_is_extended(hand_landmarks, tip, pip, mcp):
    """Whether a finger is out, seen from any angle."""

    return finger_extension(
        hand_landmarks, tip, pip, mcp
    ) > EXTENDED_RATIO


def finger_reach(hand_landmarks, tip, mcp):
    """How far past its knuckle a fingertip sits, as a multiple.

    About 1 when the hand is shut, around 1.3 when it is merely resting,
    and 1.6 or more when the finger is out.
    """

    knuckle = distance(hand_landmarks[WRIST], hand_landmarks[mcp])

    if knuckle == 0:
        return 0.0

    return distance(hand_landmarks[WRIST], hand_landmarks[tip]) / knuckle


def detect_fingers(hand_landmarks):
    """Which fingers are out."""

    return {
        name: finger_is_extended(hand_landmarks, tip, pip, mcp)
        for name, (tip, pip, mcp) in FINGERS.items()
    }


def finger_span(hand):
    """How straight each finger is, taking the better of the two views.

    The screen view is exact for a hand held flat and hopeless for one
    aimed at the camera; the world view is the reverse.  Requiring both
    to agree loses real gestures -- which is what stopped two fingers
    registering at all -- so the larger reading wins.

    A number rather than a yes or no, because whoever asks may want to
    know how near the line it is.  ``fingers_out`` is the same thing
    thresholded.
    """

    shape = shape_of(hand)
    screen = screen_of(hand)

    return {
        name: max(finger_extension(shape, tip, pip, mcp),
                  finger_extension(screen, tip, pip, mcp))
        for name, (tip, pip, mcp) in FINGERS.items()
    }


def fingers_out(hand):
    """Which fingers are out, asking both views of the hand."""

    return {name: span > EXTENDED_RATIO
            for name, span in finger_span(hand).items()}


def looks_at_rest(spans):
    """Whether these finger readings match the calibrated resting hand.

    The calibration records the resting hand for one promise: that it
    asks for nothing.  The thresholds alone cannot always keep it --
    they are lines, and a resting hand drifts across them -- so the
    recording itself is the last word: a hand that looks like *your*
    hand at rest is resting.  Answers False until a calibration has
    provided the signature.
    """

    if not REST_SIGNATURE:
        return False

    return all(abs(spans[name] - rest) <= REST_TOLERANCE
               for name, rest in REST_SIGNATURE.items()
               if name in spans)


def is_open(hand_landmarks):
    """Whether every finger is properly out, not merely un-curled."""

    return all(
        finger_extension(hand_landmarks, tip, pip, mcp) > OPEN_RATIO
        for tip, pip, mcp in FINGERS.values()
    )


def is_clenched(hand):
    """Whether the hand is shut, not merely un-straight.

    The index finger has to be in, and two of the other three.  Requiring
    all four made a fist fail whenever one finger was misread, which on a
    closed hand is often -- three of them are half hidden behind it.  The
    index is not optional because that is what separates a fist from a
    pinch: pinching holds it out to meet the thumb.

    A finger is in if either measurement says so.  Where its tip ended up
    is the sharper answer when the hand is squeezed shut; how curled the
    finger is holds up better when it is not, and does not depend on the
    fingertip, which is the landmark a closed hand hides.  Asking both was
    what let a fist measured from a hard-clenched calibration go on
    recognising a fist made casually -- reach put those two a long way
    apart, and curl barely tells them apart at all.
    """

    screen = screen_of(hand)
    shape = shape_of(hand)

    closed = {
        name: (finger_reach(screen, tip, mcp) < FIST_REACH
               or finger_extension(shape, tip, pip, mcp) < FIST_CURL)
        for name, (tip, pip, mcp) in FINGERS.items()
    }

    if not closed["index"]:
        return False

    return sum(closed[name] for name in ("middle", "ring", "pinky")) >= 2


def reach_scores(hand_landmarks):
    """Every finger's reach, for the tuning overlay."""

    return {
        name: round(finger_reach(hand_landmarks, tip, mcp), 2)
        for name, (tip, _, mcp) in FINGERS.items()
    }


def finger_scores(hand_landmarks):
    """Every finger's extension score, for the tuning overlay."""

    return {
        name: round(finger_extension(hand_landmarks, tip, pip, mcp), 2)
        for name, (tip, pip, mcp) in FINGERS.items()
    }


def pointing_direction(hand):
    """Which way the two fingers lean: -1 hard left, 0 upright, +1 hard right.

    Measured purely on screen, in x and y, with no depth at all.  The
    gesture is a wrist rotation with the fingers staying toward the camera,
    so the whole of it happens in the plane of the image -- which is the
    part MediaPipe reports accurately.  Bringing depth into it would put
    the measurement back on the axis a single camera can only estimate,
    which is what made the earlier version unreliable.

    It stays relative to the wrist, so carrying the hand across the frame
    without turning it reads as nothing, and it is a ratio, so how far away
    you sit does not matter.
    """

    screen = screen_of(hand)

    wrist = screen[WRIST]

    tip_x = (screen[8].x + screen[12].x) / 2
    tip_y = (screen[8].y + screen[12].y) / 2

    sideways = tip_x - wrist.x
    reach = math.hypot(sideways, tip_y - wrist.y)

    # Fingers aimed at the camera collapse toward the wrist on screen, so
    # the reach shrinks and dividing by it would make small movements read
    # as huge -- enough that slowly lowering the hand counted as a swipe.
    # A hand and a half is the floor: comfortably under the reach of an
    # upright peace sign, so that gesture is measured as it stands.
    lean = sideways / max(reach, hand_scale(screen) * 1.5)

    # That ratio is the sine of the lean, and a sine flattens out.  Forty
    # degrees of wrist read 0.67 from upright and 0.26 from a wrist
    # already turned 55 -- the same movement, a third of the reading --
    # so a turn made from a hand that habitually sits tilted one way was
    # measured as barely happening, and did not register at all.  Reported
    # as swipe right not working, on a hand that rests leaning right.
    #
    # The angle itself does not flatten.  In radians it is within a
    # percent of the sine for the small leans where the thresholds were
    # measured, and only diverges where the sine was going wrong.
    return math.asin(max(-1.0, min(1.0, lean)))
