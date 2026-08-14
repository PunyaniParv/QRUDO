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


def tuning_lines(state, motion, hand_state):
    """The numbers behind a swipe decision, against what they must reach.

    Whichever line is short of its target is the reason a swipe did not
    fire, which is the whole point of showing them.
    """

    scores = state.get("ext", {})

    return [
        f"pose  {state.get('pose')}   armed {state.get('armed')}",
        f"aim   {state.get('aim', 0):+.2f}   (-1 left, +1 right)",
        f"turn  {state.get('turn', 0):.2f} / {motion.SWIPE_TURN}",
        f"slide {state.get('slide', 0):.2f} / {motion.SWIPE_SLIDE}"
        f"   (peace sign only)",
        f"speed {state.get('speed', 0):.2f} / {motion.SWIPE_TURN_SPEED}",
        f"agree {state.get('agree', 0):.2f} / {motion.SWIPE_CONSISTENCY}",
        "ext   " + "  ".join(
            f"{name[0]}{score:.2f}" for name, score in scores.items()
        ) + f"   (out above {hand_state.EXTENDED_RATIO})",
    ]


def draw_tuning(cv2, frame, state, motion, hand_state):
    """Print those numbers down the left of the frame."""

    for number, line in enumerate(tuning_lines(state, motion, hand_state)):
        cv2.putText(frame, line, (20, 130 + number * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1)
