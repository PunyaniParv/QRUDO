"""Walk the user through showing SARV each gesture, and measure them.

Four poses held still, then a few movements repeated.  Each step gives a
moment to get into position before it starts recording, so the transition
into the pose is not measured as part of it.

Nothing here decides a threshold: it collects readings and hands them to
vision.calibration, which works out where the lines go.
"""

from __future__ import annotations

import sys
import time

#: name, what to ask for, what it is for, seconds of recording
POSES = [
    ("fist", "Make a FIST, palm toward the camera",
     "this plays and pauses", 3.0),
    ("open", "Hold your hand OPEN, fingers spread",
     "so a hand doing nothing is not read as one", 3.0),
    ("two", "Hold up TWO FINGERS",
     "the pose you swipe with", 3.0),
    ("rest", "Let your hand REST naturally, however it falls",
     "so a hand doing nothing is left alone", 3.0),
]

#: name, what to ask for, what it is for, how many times
MOVES = [
    ("turn", "Two fingers up: TURN YOUR WRIST left, then back",
     "this rewinds and skips forward", 3),
]

READY_SECONDS = 2.0
MOVE_SECONDS = 2.5


def run(args):
    """Record, work out the thresholds, and save them."""

    import cv2

    from vision import Camera, CameraError, HandTracker, TrackerError
    from vision import calibration, gestures, hand_state, motion
    import vision
    from ui import overlay

    try:
        tracker = HandTracker().open()
        camera = Camera(
            args.camera,
            width=1280 if getattr(args, "far", False) else 640,
            height=720 if getattr(args, "far", False) else 480,
        ).open()
    except (CameraError, TrackerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(banner())

    session = _Session(cv2, camera, tracker, vision, gestures, motion, overlay)

    try:
        poses = {name: session.record_pose(prompt, purpose, seconds)
                 for name, prompt, purpose, seconds in POSES}

        moves = {name: session.record_move(prompt, purpose, times)
                 for name, prompt, purpose, times in MOVES}
    except _Cancelled:
        print("\n  cancelled -- nothing was saved.")
        return 1
    except CameraError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

    measured, warnings = calibration.from_samples(
        poses, moves, calibration.current())

    print("\n  measured:\n")

    for line in measured.describe():
        print(f"    {line}")

    for warning in warnings:
        print(f"\n  ! {warning}"
              f"\n    keeping the existing value for it")

    path = measured.save()

    print(f"\n  saved to {path.name} -- it is used from now on.")
    print("  delete that file to go back to the defaults.\n")

    return 0


class _Cancelled(Exception):
    """The user pressed q."""


class _Session:
    """One pass in front of the camera."""

    def __init__(self, cv2, camera, tracker, vision, gestures, motion, overlay):
        self.cv2 = cv2
        self.camera = camera
        self.tracker = tracker
        self.vision = vision
        self.gestures = gestures
        self.motion = motion
        self.overlay = overlay

    def record_pose(self, prompt, purpose, seconds):
        """Hold still: collect the finger measurements frame by frame."""

        self._countdown(prompt, READY_SECONDS, purpose)

        readings = []
        until = time.time() + seconds

        while time.time() < until:
            hand = self._frame(prompt, "hold it", until - time.time(),
                               purpose)

            if hand is None:
                continue

            state = self.motion.debug_state()

            if "ext" in state and "reach" in state:
                readings.append({
                    "ext": state["ext"],
                    "reach": state["reach"],
                    "scale": state.get("scale", 0.1),
                })

        print(f"    {prompt}: {len(readings)} readings")

        return readings

    def record_move(self, prompt, purpose, times):
        """Move: collect the largest reading reached in each repetition."""

        peaks = []

        for attempt in range(1, times + 1):
            self._countdown(f"{prompt}  ({attempt} of {times})",
                            READY_SECONDS, purpose)

            self.vision.reset_state()

            biggest = (0.0, 0.0)
            until = time.time() + MOVE_SECONDS

            while time.time() < until:
                hand = self._frame(prompt, f"go  ({attempt} of {times})",
                                   until - time.time(), purpose)

                if hand is None:
                    continue

                state = self.motion.debug_state()

                if "turn" in state:
                    biggest = _bigger(biggest, state, prompt)

            if biggest[0] > 0:
                peaks.append(biggest)

            print(f"    attempt {attempt}: {biggest[0]:.2f} at {biggest[1]:.2f}/s")

        return peaks

    def _countdown(self, prompt, seconds, purpose=""):
        until = time.time() + seconds

        while time.time() < until:
            self._frame(prompt, "get ready", until - time.time(), purpose)

    def _frame(self, prompt, note, remaining, purpose=""):
        """One frame: track, draw, and check for q."""

        frame = self.camera.read()
        hand = self.tracker.track(frame)

        if hand is not None:
            self.gestures.detect_gesture(hand)
            self.motion.detect_swipe(hand)

        self.overlay.draw_prompt(self.cv2, frame, prompt, note, remaining,
                                 hand is not None, purpose)

        self.cv2.imshow("SARV - calibrating", frame)

        if self.cv2.waitKey(1) & 0xFF == ord("q"):
            raise _Cancelled

        return hand


def _bigger(biggest, state, prompt):
    """The larger of what we have and what this frame shows."""

    size, speed = abs(state.get("turn", 0)), state.get("speed", 0)

    return (max(biggest[0], size), max(biggest[1], speed))


def banner():
    return "\n".join([
        "",
        "  SARV setup",
        "",
        "  Four poses to hold, then one movement to repeat.  Stand where",
        "  you actually intend to use it -- the numbers depend on how big",
        "  your hand looks, so calibrating close up will not suit a demo",
        "  from across the room.",
        "",
        "  q in the window to stop without saving.",
        "",
    ])
