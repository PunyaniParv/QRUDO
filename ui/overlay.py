"""Drawing on the camera preview.

Two audiences.  During a demo the overlay explains what just happened, so
the room can see why the video paused.  While tuning it shows the raw
numbers behind a decision, so a gesture that will not fire becomes one
threshold to adjust rather than a guess.

cv2 is passed in rather than imported, so this module stays importable --
and testable -- without OpenCV present.
"""

from __future__ import annotations

GREEN = (0, 255, 0)
RED = (0, 0, 255)
GREY = (200, 200, 200)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


def draw_gesture(cv2, frame, gesture):
    """The gesture being seen, large, at the top."""

    cv2.putText(frame, gesture or "no hand", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                GREEN if gesture else RED, 2)


def draw_result(cv2, frame, result):
    """What the last command did, under the gesture."""

    if result is None:
        return

    # Throttled commands are normal and constant; showing them is noise.
    if result.status == "THROTTLED":
        return

    cv2.putText(frame, f"{result.command}: {result.detail or result.error}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                GREEN if result.ok else RED, 1)


def draw_prompt(cv2, frame, prompt, note, remaining, hand_seen, purpose=""):
    """What to do next, while calibrating.

    Large, because you are meant to be standing where you would actually
    use it -- which may be several metres away.
    """

    height, width = frame.shape[:2]
    colour = GREEN if hand_seen else RED

    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), colour, 10)

    cv2.putText(frame, prompt, (24, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)

    cv2.putText(frame, f"{note}  {max(0.0, remaining):.0f}",
                (24, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)

    if purpose:
        cv2.putText(frame, purpose, (24, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1)

    if not hand_seen:
        cv2.putText(frame, "no hand -- move into view",
                    (24, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)


def legend_lines(mapping):
    """What each gesture does, in plain words.

    Shown on the preview the whole time it is running.  A gesture app that
    does not say what its gestures are is unusable by anyone who did not
    write it, which during a demo includes the audience.
    """

    described = {
        "FIST": "fist",
        "OPEN_PALM": "open hand",
        "POINT": "point",
        "TWO_FINGER": "two fingers",
        "SWIPE_LEFT": "2 fingers, turn wrist left",
        "SWIPE_RIGHT": "2 fingers, turn wrist right",
        "PINCH_UP": "pinch, raise hand",
        "PINCH_DOWN": "pinch, lower hand",
        "PINCH": "pinch",
    }

    return [
        f"{described.get(gesture, gesture.lower()):<26} {command}"
        for gesture, command in mapping.items()
    ]


def draw_legend(cv2, frame, mapping):
    """Print the mapping down the bottom-left of the frame."""

    lines = legend_lines(mapping)
    height = frame.shape[0]
    top = height - 20 - (len(lines) * 22)

    for number, line in enumerate(lines):
        cv2.putText(frame, line, (20, top + number * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)


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
        line("slide", "slide", motion.SWIPE_SLIDE, "  (peace sign only)"),
        line("speed", "speed", motion.SWIPE_TURN_SPEED),
        line("agree", "agree", motion.SWIPE_CONSISTENCY),
        "   ext   " + "  ".join(
            f"{name[0]}{score:.2f}" for name, score in scores.items()
        ) + f"   (out above {hand_state.EXTENDED_RATIO})",
        "   reach " + "  ".join(
            f"{name[0]}{score:.2f}"
            for name, score in state.get("reach", {}).items()
        ) + f"   (fist below {hand_state.FIST_REACH})",
        f"   pinch {state.get('pinch', 0):.2f}"
        f"   (pinch below {hand_state.PINCH_GAP},"
        f" open above {hand_state.OPEN_RATIO})",
    ]


def draw_tuning(cv2, frame, state, motion, hand_state):
    """Print those numbers down the left of the frame."""

    for number, line in enumerate(tuning_lines(state, motion, hand_state)):
        cv2.putText(frame, line, (20, 130 + number * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1)
