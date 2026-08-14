"""The live loop: camera to gesture to command, once per frame.

Everything it touches is somebody else's: vision finds the hand and names
what it is doing, bridge turns that name into a command, control performs
it, ui draws it.  This file only decides the order.
"""

from __future__ import annotations

import sys
import time

from .bridge import GestureRouter

#: How long a completed swipe stays on screen, in seconds.
SWIPE_SHOWN_FOR = 0.8


def run(engine, args, tuning=False):
    """Run SARV until q is pressed or the camera goes away.

    ``tuning`` shows the numbers behind each decision and performs
    nothing, which is the mode to use when a gesture will not fire.
    """

    import cv2

    from vision import Camera, CameraError, HandTracker, TrackerError
    from vision import gestures as gesture_module
    from vision import hand_state, motion
    import vision
    from ui import overlay

    router = GestureRouter()
    last_result = None
    last_swipe = None
    last_swipe_at = 0.0

    def remember(result):
        nonlocal last_result
        last_result = result

    engine.on_result = remember

    for warning in engine.preflight():
        print(f"  ! {warning}\n")

    try:
        tracker = HandTracker().open()
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        camera = Camera(args.camera).open()
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        tracker.close()
        return 1

    print(banner(engine, router, args, tuning), flush=True)

    show_window = not args.no_window

    try:
        for frame in camera.frames():
            hand = tracker.track(frame)
            gesture = None

            if hand is None:
                vision.reset_state()
                router.forget()
            else:
                gesture = gesture_module.detect_gesture(hand)
                swipe = motion.detect_swipe(hand)

                if not tuning:
                    command = router.update(gesture, swipe)

                    if command is not None:
                        # submit, not execute: a brightness change can take
                        # over a second on Windows, and the camera must not
                        # wait for it.
                        engine.submit(command)

                if swipe:
                    last_swipe = swipe
                    last_swipe_at = time.time()

            # A swipe is reported on the single frame it completes, which at
            # 30 fps is far too brief to read.  Hold it on screen for long
            # enough to see that it happened.
            if last_swipe and time.time() - last_swipe_at < SWIPE_SHOWN_FOR:
                gesture = last_swipe

            if not show_window:
                continue

            overlay.draw_gesture(cv2, frame, gesture)
            overlay.draw_result(cv2, frame, last_result)
            overlay.draw_legend(cv2, frame, router.mapping())

            if tuning:
                overlay.draw_tuning(
                    cv2, frame, motion.debug_state(), motion, hand_state)

            cv2.imshow("SARV", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        tracker.close()
        engine.close()

        if show_window:
            cv2.destroyAllWindows()

    print("\n  bye.")
    return 0


def banner(engine, router, args, tuning):
    lines = [
        "",
        f"  SARV  --  {engine.controller.name}"
        f"{'  [TUNING, nothing will run]' if tuning else ''}"
        f"{'  [DRY RUN]' if engine.config.dry_run and not tuning else ''}",
        "",
    ]

    if not tuning:
        for gesture, command in router.mapping().items():
            lines.append(f"    {gesture:<12} {command}")
        lines.append("")

    lines += [
        "  q in the window to stop" if not args.no_window else
        "  ctrl+c to stop",
        "",
    ]

    return "\n".join(lines)
