"""Drawing on the camera preview.

Two audiences.  During a demo the overlay explains what just happened, so
the room can see why the video paused.  While tuning it shows the raw
numbers behind a decision, so a gesture that will not fire becomes one
threshold to adjust rather than a guess.

The preview is a room, not a page: whatever is behind the text is
whatever you happen to be standing in front of, and pale text on a pale
wall is not readable.  Two attempts at fixing that -- shading the edges
of the picture, then a dark rim on every letter -- both looked like
something was wrong with the camera, which is worse than the problem.
So nothing is done to the picture, and what stands on it is chosen to
read on its own: strong colours, and dark where the room is bright.  The
rim survives only for the two lists that appear when asked for, where
there is a lot of small text and no room to lose any of it.

cv2 is passed in rather than imported, so this module stays importable --
and testable -- without OpenCV present.
"""

from __future__ import annotations

#: Softer than full-brightness primaries, which glare against a lit room
#: and bloom on a webcam's own exposure.
LIVE = (80, 220, 110)       # a hand is being read
ALERT = (95, 95, 255)       # something failed, or is about to
MISSING = (30, 30, 165)     # no hand at all
GREY = (225, 225, 225)
DIM = (195, 195, 195)
SHADOW = (0, 0, 0)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX

#: Everything starts this far from the left, so the lines share an edge.
MARGIN = 20


def text(cv2, frame, line, at, scale=0.6, colour=GREY, weight=1, rim=False):
    """One line, with its own contrast.

    Small lines are drawn in black four times, a pixel out in each
    direction, and in colour on top, so each letter keeps a thin dark rim
    whatever is behind it.  Without something they are legible against a
    dark room and invisible against a white wall, and which one you get is
    not something the app chooses.  Headings are left plain -- see below.

    Four copies rather than one heavier one underneath, which is the
    obvious way to do it and does not work: OpenCV spaces letters further
    apart as the stroke thickens, so the heavier copy is a good deal wider
    than the text it is meant to sit behind.  The two start together and
    drift, and by the end of a sentence the black is a legible second copy
    of the last few words.  Offsetting at the same thickness keeps every
    letter where it was.
    """

    x, y = at

    if rim:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.putText(frame, line, (x + dx, y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, SHADOW, weight,
                        cv2.LINE_AA)

    cv2.putText(frame, line, at, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, weight, cv2.LINE_AA)


def draw_gesture(cv2, frame, gesture):
    """The gesture being seen, large, at the top."""

    text(cv2, frame, gesture or "no hand", (MARGIN, 52), 0.95,
         LIVE if gesture else MISSING, 2)


def draw_result(cv2, frame, result):
    """What the command just did, under the gesture.

    Only just: the runner stops passing it after a couple of seconds.  It
    is news, and news that stays on screen until the next one is not news
    -- it is a label saying PLAY_PAUSE over your face while you are trying
    to see whether your hand is being read.
    """

    if result is None:
        return

    # Throttled commands are normal and constant; showing them is noise.
    if result.status == "THROTTLED":
        return

    text(cv2, frame, f"{result.command}  {result.detail or result.error}",
         (MARGIN, 82), 0.58, LIVE if result.ok else ALERT)


def draw_hint(cv2, frame, line):
    """The bottom-left corner, for when nothing is happening.

    "It stopped working" and "you are out of range" look identical from
    where the user is standing, so the app says which.  In the corner
    because it is the answer to a question nobody has asked yet.
    """

    if not line:
        return

    text(cv2, frame, line, (MARGIN, frame.shape[0] - 20), 0.58, ALERT)


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
COMMAND_AT = 275

#: Room one legend line takes.
LINE = 26


def draw_legend(cv2, frame, mapping, showing=True):
    """The mapping up the bottom-left, above the hint line.

    ``showing`` is the h key, and off is the whole of what is drawn then.
    A line on the picture reminding you that h exists is one more thing
    over your own face; the terminal says it once at startup, which is
    where you already are when you type the command.
    """

    if not showing:
        return

    height = frame.shape[0]
    lines = legend_lines(mapping)
    top = height - 52 - (len(lines) - 1) * LINE

    for number, (gesture, command) in enumerate(lines):
        at = top + number * LINE

        text(cv2, frame, gesture, (MARGIN, at), 0.55, GREY, rim=True)
        text(cv2, frame, command, (MARGIN + COMMAND_AT, at), 0.55, DIM,
             rim=True)


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
        text(cv2, frame, line, (MARGIN, 122 + number * 26), 0.55, GREY,
             rim=True)
