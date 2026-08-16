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
#:
#: Asked for as the things they do, not as shapes to hold.  People make a
#: gesture differently when told to pause a video than when told to close
#: their hand, and it is the first one the app has to recognise later.
POSES = [
    ("fist", "PLAY OR PAUSE:  make a fist",
     "the shape that plays and pauses", 3.0),
    ("two", "READY TO SEEK:  hold up two fingers",
     "the shape every swipe is made from", 3.0),
    ("open", "Hold your hand OPEN, fingers spread",
     "so a spread hand can be told from a slack one", 3.0),
    ("rest", "Let your hand REST naturally, however it falls",
     "so a hand doing nothing is left alone", 3.0),
]

#: name, what to ask for, what it is for, how many times, what to measure
#:
#: Both directions of each, because a wrist does not turn as far one way
#: as the other and an arm does not rise as far as it falls.  Measuring
#: only the easy direction sets a bar the hard one never clears.
MOVES = [
    ("turn left", "REWIND:  two fingers up, turn your wrist left, then back",
     "how far your wrist turns", 2, "turn"),
    ("turn right", "SKIP FORWARD:  two fingers up, turn your wrist right, then back",
     "the other way, which does not go as far", 2, "turn"),
    ("raise", "VOLUME UP:  two fingers up, raise your hand, then lower it",
     "how far your hand travels, and how briskly", 2, "lift"),
    ("lower", "VOLUME DOWN:  hand up, lower it, then raise it",
     "the same going down, which is not the same", 2, "lift"),
]

#: Everything measured during a movement, whichever movement was asked
#: for.  Turning the wrist raises the hand a little and raising it turns
#: the wrist a little, and telling the two gestures apart means knowing
#: how much -- which cannot be known from a recording that only kept the
#: half it went looking for.
MEASURED = ("turn", "speed", "lift", "lift_speed")

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
            width=1600 if getattr(args, "far", False) else 640,
            height=1200 if getattr(args, "far", False) else 480,
        ).open()
    except (CameraError, TrackerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(banner())

    session = _Session(cv2, camera, tracker, vision, gestures, motion, overlay)

    try:
        poses = {name: session.record_pose(prompt, purpose, seconds)
                 for name, prompt, purpose, seconds in POSES}

        moves = {name: session.record_move(prompt, purpose, times, axis)
                 for name, prompt, purpose, times, axis in MOVES}
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

    profile = calibration.Profile.from_samples(poses, moves)
    measured, warnings = profile.derive(calibration.current())

    # Checked before it is saved rather than on the way back in, so what
    # is printed here is what will actually be used, and anything that
    # had to be pulled back is said while you are still standing there.
    measured, pulled = measured.sensible()
    warnings = list(warnings) + list(pulled)

    print("\n  measured:\n")

    for line in measured.describe():
        print(f"    {line}")

    for warning in warnings:
        print(f"\n  ! {warning}"
              f"\n    keeping the existing value for it")

    for note in measured.advice:
        print(f"\n  - {note}")

    # The readings go in beside the thresholds, so that a correction to
    # how they are worked out reaches this session without it having to
    # be recorded again -- and so a gesture added later can be built from
    # a pose and a movement already measured here.
    path = measured.save(profile=profile)

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

    def record_move(self, prompt, purpose, times, axis="turn"):
        """Move: collect the largest readings reached in each repetition.

        Every measurement is kept, not only the one this movement is
        about.  ``axis`` says which one that is, so a repetition where
        nothing much happened can be told from one where it did.
        """

        peaks = []

        for attempt in range(1, times + 1):
            self._countdown(f"{prompt}  ({attempt} of {times})",
                            READY_SECONDS, purpose)

            self.vision.reset_state()

            biggest = dict.fromkeys(MEASURED, 0.0)
            until = time.time() + MOVE_SECONDS

            while time.time() < until:
                hand = self._frame(prompt, f"go  ({attempt} of {times})",
                                   until - time.time(), purpose)

                if hand is None:
                    continue

                state = self.motion.debug_state()

                if axis in state:
                    biggest = _bigger(biggest, state)

            wanted = biggest["turn" if axis == "turn" else "lift"]

            if wanted > 0:
                peaks.append(biggest)

            print(f"    attempt {attempt}: {wanted:.2f}")

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


def _bigger(biggest, state):
    """The larger of what we have and what this frame shows, in every
    measurement.

    Taken as sizes without a sign: lowering a hand is a negative lift,
    and it is as much of a movement as raising one.
    """

    return {name: max(biggest[name], abs(state.get(name, 0)))
            for name in MEASURED}


def banner():
    return "\n".join([
        "",
        "  SARV setup",
        "",
        "  Four shapes to hold, then four movements to repeat.  Stand where",
        "  you actually intend to use it -- the numbers depend on how big",
        "  your hand looks, so calibrating close up will not suit a demo",
        "  from across the room.",
        "",
        "  q in the window to stop without saving.",
        "",
    ])
