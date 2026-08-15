"""Drawing on the camera preview.

Two audiences.  During a demo the overlay explains what just happened, so
the room can see why the video paused.  While tuning it shows the raw
numbers behind a decision, so a gesture that will not fire becomes one
threshold to adjust rather than a guess.

The preview is a room, not a page: whatever is behind the text is
whatever you happen to be standing in front of, and pale text on a pale
wall is not readable.  So the top and bottom of the picture are shaded
down where the lines sit, and each line carries a shadow of its own.
Between them the text reads against a white wall and a dark one without
putting a box over either.

cv2 is passed in rather than imported, so this module stays importable --
and testable -- without OpenCV present.
"""

from __future__ import annotations

#: Softer than full-brightness primaries, which glare against a lit room
#: and bloom on a webcam's own exposure.
LIVE = (140, 255, 155)      # a hand is being read
ALERT = (95, 95, 255)       # nothing is being read, or something failed
GREY = (225, 225, 225)
DIM = (195, 195, 195)
SHADOW = (0, 0, 0)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX

#: Everything starts this far from the left, so the lines share an edge.
MARGIN = 20


#: How far the shading takes to disappear once it is past the text.
FADE = 58


def dim_edges(cv2, frame, top=94, bottom=92):
    """Darken the top and bottom of the picture, fading into the middle.

    Text over a camera has no background of its own, and the one it gets
    is whatever the room is.  A shadow copes with a busy wall; it does not
    cope with a bright one, where pale text has nothing to be paler than.

    ``top`` and ``bottom`` are how much room the text needs, not how far
    the shading reaches: each is held at full strength across its own
    lines and then fades out over ``FADE`` rows beyond them.  Giving the
    band the fade to share instead leaves the text sitting in the weakest
    part of the very thing put there to hold it.

    Nothing is hidden -- it is a darker picture at the edges, not a panel
    over it.
    """

    height = frame.shape[0]

    _shade(frame, 0, min(top, height), from_top=True)
    _shade(frame, max(0, height - bottom), height, from_top=False)


def _shade(frame, top, bottom, from_top, strength=0.45):
    """Darken these rows, and fade back out over the rows beyond them.

    A row at a time rather than in a few blocks: shading in blocks leaves
    visible steps across a plain wall, which reads as a rendering fault
    rather than as shading.

    numpy is imported here rather than at the top so this module keeps
    loading -- and its lines keep being testable -- on a machine with
    neither it nor OpenCV.
    """

    import numpy

    height = frame.shape[0]

    if bottom - top < 1:
        return

    solid = numpy.full(bottom - top, strength)
    fade = numpy.linspace(strength, 0.0, FADE)

    if from_top:
        rows, first = numpy.concatenate([solid, fade]), top
    else:
        rows, first = numpy.concatenate([fade[::-1], solid]), top - FADE

    # Clip to the picture, in case the bands are asked to overlap the edge.
    start = max(0, first)
    rows = rows[start - first:start - first + height - start]

    keep = (1.0 - rows).reshape(len(rows), 1, 1)

    frame[start:start + len(rows)] = (
        frame[start:start + len(rows)] * keep).astype(frame.dtype)


def text(cv2, frame, line, at, scale=0.6, colour=GREY, weight=1):
    """One line, with its own contrast.

    A dark copy sits a couple of pixels down and right, and the colour
    goes on top.  Without something the overlay is legible against a dark
    room and invisible against a white wall, and which one you get is not
    something the app chooses.

    Offset rather than an outline: an outline around a letter this size is
    thicker than the strokes of the letter, so it fills the counters in
    and the text turns into a smear that is still, technically, legible.
    """

    x, y = at
    away = 2 if weight > 1 else 1

    cv2.putText(frame, line, (x + away, y + away), cv2.FONT_HERSHEY_SIMPLEX,
                scale, SHADOW, weight, cv2.LINE_AA)
    cv2.putText(frame, line, at, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, weight, cv2.LINE_AA)


def draw_gesture(cv2, frame, gesture):
    """The gesture being seen, large, at the top."""

    text(cv2, frame, gesture or "no hand", (MARGIN, 52), 0.95,
         LIVE if gesture else ALERT, 2)


def draw_result(cv2, frame, result):
    """What the last command did, under the gesture."""

    if result is None:
        return

    # Throttled commands are normal and constant; showing them is noise.
    if result.status == "THROTTLED":
        return

    text(cv2, frame, f"{result.command}  {result.detail or result.error}",
         (MARGIN, 82), 0.55, LIVE if result.ok else ALERT)


def draw_hint(cv2, frame, line):
    """The bottom-left corner, for when nothing is happening.

    "It stopped working" and "you are out of range" look identical from
    where the user is standing, so the app says which.  In the corner
    because it is the answer to a question nobody has asked yet.
    """

    if not line:
        return

    text(cv2, frame, line, (MARGIN, frame.shape[0] - 20), 0.55, ALERT)


def legend_lines(mapping):
    """What each gesture does, in plain words.

    Not on screen by default.  It was, and a five-line list along the
    bottom of a camera preview is not a reference -- it is the thing you
    look past to see whether your hand is being read.  It is worth having
    on the first run and in the way on every one after, so it is a key
    press.
    """

    described = {
        "FIST": "fist",
        "OPEN_PALM": "open hand",
        "POINT": "point",
        "TWO_FINGER": "two fingers",
        "SWIPE_LEFT": "2 fingers, turn wrist left",
        "SWIPE_RIGHT": "2 fingers, turn wrist right",
        "SWIPE_UP": "2 fingers, raise hand",
        "SWIPE_DOWN": "2 fingers, lower hand",
    }

    # The command is an enum, and what str() makes of one changed in 3.11
    # -- ``.value`` is the name either way, and it is what gets drawn.
    return [(described.get(gesture, gesture.lower()),
             getattr(command, "value", str(command)))
            for gesture, command in mapping.items()]


#: Where the command column starts, in pixels.  Padding the description
#: out with spaces lines nothing up: the font is proportional, so a row of
#: spaces is a different width on every line.
COMMAND_AT = 250

#: Room one legend line takes, and what the whole list needs.
LINE = 24


def legend_height(mapping):
    """How much of the bottom the legend needs, hint line included."""

    return 52 + len(legend_lines(mapping)) * LINE


def draw_legend(cv2, frame, mapping, showing=True):
    """The mapping up the bottom-left, above the hint line.

    ``showing`` is the h key.  When it is off the reminder that h exists
    takes its place, in one dim line, so the list is findable without
    being permanent.
    """

    height = frame.shape[0]

    if not showing:
        text(cv2, frame, "h  what the gestures do", (MARGIN, height - 44),
             0.45, DIM)
        return

    lines = legend_lines(mapping)
    top = height - 52 - (len(lines) - 1) * LINE

    for number, (gesture, command) in enumerate(lines):
        at = top + number * LINE

        text(cv2, frame, gesture, (MARGIN, at), 0.5, GREY)
        text(cv2, frame, command, (MARGIN + COMMAND_AT, at), 0.5, DIM)


def draw_prompt(cv2, frame, prompt, note, remaining, hand_seen, purpose=""):
    """What to do next, while calibrating.

    Large, because you are meant to be standing where you would actually
    use it -- which may be several metres away.
    """

    height, width = frame.shape[:2]
    colour = LIVE if hand_seen else ALERT

    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), colour, 10)

    text(cv2, frame, prompt, (24, 56), 0.7, GREY, 2)
    text(cv2, frame, f"{note}  {max(0.0, remaining):.0f}", (24, 100),
         1.1, colour, 3)

    if purpose:
        text(cv2, frame, purpose, (24, 132), 0.6, DIM)

    if not hand_seen:
        text(cv2, frame, "no hand -- move into view", (24, height - 30),
             0.7, ALERT, 2)


def tuning_lines(state, motion, hand_state):
    """The numbers behind a swipe decision, against what they must reach.

    Whichever line is short of its target is the reason a swipe did not
    fire, which is the whole point of showing them.
    """

    scores = state.get("ext", {})

    def line(label, key, target, note=""):
        """Now, the best of the last couple of seconds, and what is needed.

        The peak is the number to read: everything spikes during the swipe
        and is back to nothing by the time you look up.  A peak short of
        its target is the reason the swipe did not fire.
        """

        peak = state.get(f"peak_{key}", 0)
        mark = "ok " if peak >= target else "   "

        return (f"{mark}{label} {state.get(key, 0):.2f}"
                f"  peak {peak:.2f} / {target}{note}")

    fps = state.get("fps")

    return [
        # Below about 15 the gestures suffer: a quick turn stops producing
        # enough frames to be seen as one.
        f"   fps   {fps:.0f}" if fps else "   fps   -",
        f"   pose  {state.get('pose')}   armed {state.get('armed')}",
        f"   aim   {state.get('aim', 0):+.2f}   (-1 left, +1 right)",
        line("turn ", "turn", motion.SWIPE_TURN),
        line("lift ", "lift", motion.SWIPE_LIFT),
        line("speed", "speed", motion.SWIPE_TURN_SPEED),
        line("agree", "agree", motion.SWIPE_CONSISTENCY),
        "   ext   " + "  ".join(
            f"{name[0]}{score:.2f}" for name, score in scores.items()
        ) + f"   (out above {hand_state.EXTENDED_RATIO})",
        "   reach " + "  ".join(
            f"{name[0]}{score:.2f}"
            for name, score in state.get("reach", {}).items()
        ) + f"   (fist below {hand_state.FIST_REACH})",
        "   why   " + str(state.get("why", "")),
        f"   open above {hand_state.OPEN_RATIO}",
    ]


def draw_tuning(cv2, frame, state, motion, hand_state):
    """Print those numbers down the left of the frame."""

    for number, line in enumerate(tuning_lines(state, motion, hand_state)):
        text(cv2, frame, line, (MARGIN, 122 + number * 26), 0.55, GREY)
