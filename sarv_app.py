#!/usr/bin/env python3
"""SARV -- control your computer with hand gestures.

    python sarv_app.py                 # camera on, preview window
    python sarv_app.py --no-window     # no preview, nothing steals focus
    python sarv_app.py --dry-run       # recognise gestures, touch nothing

This is the whole product: the camera watches your hand, the Vision Engine
names what it sees, and the Control Engine does it.  The two halves meet
in exactly one place -- GestureRouter, below -- and neither knows anything
about the other.

Press q in the preview window, or ctrl+c in the terminal, to stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

from sarv import Command, ControlConfig, ControlEngine, log

MODEL_PATH = ROOT / "models" / "hand_landmarker.task"


# ---------------------------------------------------------
# The mapping: gesture names in, commands out
# ---------------------------------------------------------

#: Held poses.  These fire once when you make them, not repeatedly while
#: you hold them -- see GestureRouter.
POSE_COMMANDS = {
    "FIST": Command.PLAY_PAUSE,
}

#: Movements.  These are already one-off events with their own cooldown.
SWIPE_COMMANDS = {
    "SWIPE_LEFT": Command.REWIND,
    "SWIPE_RIGHT": Command.FORWARD,
}

# Deliberately unmapped for now:
#
#   TWO_FINGER  is the swipe pose.  Binding it would fire a command every
#               time you got ready to swipe.
#   OPEN_PALM   is what a hand looks like on its way to and from every
#               other gesture, so it would fire constantly.
#   POINT       is free, but volume and brightness need four distinct
#               signals and there are not four left.  Vertical swipes are
#               the natural way to find them.


class GestureRouter:
    """Turn a stream of gesture names into commands.

    Held poses fire on the way in and then stay quiet.  Without that, a
    fist held for two seconds would toggle play/pause three times, because
    the vision side reports it on every frame -- it describes what your
    hand *is*, not what changed.
    """

    def __init__(self, poses=None, swipes=None):
        self.poses = POSE_COMMANDS if poses is None else poses
        self.swipes = SWIPE_COMMANDS if swipes is None else swipes
        self._held = None

    def update(self, gesture=None, swipe=None):
        """Return the command this frame should run, or None."""

        # A swipe is a movement that already happened, so it always counts.
        if swipe in self.swipes:
            # Whatever pose was being held was part of the swipe; make it
            # ask again rather than firing as the hand settles.
            self._held = None
            return self.swipes[swipe]

        if gesture == self._held:
            return None

        self._held = gesture

        return self.poses.get(gesture)

    def forget(self):
        """Hand left the frame; the next pose is a new one."""

        self._held = None


# ---------------------------------------------------------
# The app
# ---------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="sarv", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="recognise gestures but do not touch the machine")
    parser.add_argument("--no-window", action="store_true",
                        help="run without the preview, so nothing takes focus")
    parser.add_argument("--camera", type=int, default=0, metavar="N",
                        help="which camera to use (default 0)")
    parser.add_argument("--config", metavar="PATH", help="JSON config file")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    config = ControlConfig.load(args.config)
    if args.dry_run:
        config.dry_run = True

    log.setup(config.log_dir, console=False)

    # Importing these takes a moment and prints a good deal, so do it after
    # the arguments have been checked.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    import gesture_detection as gestures

    if not MODEL_PATH.exists():
        print(f"error: hand model missing at {MODEL_PATH}", file=sys.stderr)
        return 1

    engine = ControlEngine(config=config)
    router = GestureRouter()
    last_result = None

    def remember(result):
        nonlocal last_result
        last_result = result

    engine.on_result = remember

    for warning in engine.preflight():
        print(f"  ! {warning}\n")

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(MODEL_PATH)),
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
    )

    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        print(f"error: could not open camera {args.camera}", file=sys.stderr)
        return 1

    # Flushed, because the camera loop below never returns to do it and a
    # redirected stdout would otherwise hold the banner until the end.
    print(banner(engine, args), flush=True)

    try:
        run_loop(camera, landmarker, gestures, engine, router,
                 cv2, mp, args, lambda: last_result)
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        landmarker.close()
        engine.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    print("\n  bye.")
    return 0


def run_loop(camera, landmarker, gestures, engine, router,
             cv2, mp, args, get_result):
    while True:
        ok, frame = camera.read()

        if not ok:
            print("error: lost the camera", file=sys.stderr)
            return

        # Mirror, so moving right on screen means moving right in life.
        # The gesture code assumes this.
        frame = cv2.flip(frame, 1)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        found = landmarker.detect(image)

        gesture = None

        if found.hand_landmarks:
            landmarks = found.hand_landmarks[0]
            handedness = found.handedness[0][0].category_name

            gesture = gestures.detect_gesture(landmarks, handedness)
            swipe = gestures.detect_swipe(landmarks, handedness)

            command = router.update(gesture, swipe)

            if command is not None:
                # submit, not execute: a brightness change can take over a
                # second on Windows, and the camera must not wait for it.
                engine.submit(command)

            if swipe:
                gesture = swipe
        else:
            gestures.reset_state()
            router.forget()

        if args.no_window:
            continue

        draw(cv2, frame, gesture, get_result())
        cv2.imshow("SARV", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


def draw(cv2, frame, gesture, result):
    """Show what was seen and what it did."""

    cv2.putText(frame, gesture or "no hand", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if gesture else (0, 0, 255), 2)

    if result is None:
        return

    # Throttled commands are normal and constant; showing them would be
    # noise.
    if result.status == "THROTTLED":
        return

    cv2.putText(frame, f"{result.command}: {result.detail or result.error}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if result.ok else (0, 0, 255), 1)


def banner(engine, args):
    lines = [
        "",
        f"  SARV  --  {engine.controller.name}"
        f"{'  [DRY RUN]' if engine.config.dry_run else ''}",
        "",
    ]

    for gesture, command in {**POSE_COMMANDS, **SWIPE_COMMANDS}.items():
        lines.append(f"    {gesture:<12} {command}")

    lines += [
        "",
        "  q in the window to stop" if not args.no_window
        else "  ctrl+c to stop",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
