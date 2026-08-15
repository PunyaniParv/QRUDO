"""The live loop: camera to gesture to command, once per frame.

Everything it touches is somebody else's: vision finds the hand and names
what it is doing, bridge turns that name into a command, control performs
it, ui draws it.  This file only decides the order.
"""

from __future__ import annotations

import sys
import time
from collections import deque

from .bridge import GestureRouter

#: How long a completed swipe stays on screen, in seconds.
SWIPE_SHOWN_FOR = 0.8

#: How long what a command did stays on screen.  Long enough to read,
#: and then gone: it is news, and it stops being news.
RESULT_SHOWN_FOR = 2.5

#: How long with no hand at all before saying something about it.  Far
#: enough away and the camera simply stops reporting one, which from where
#: the user is standing looks exactly like the app having stopped.
NOTHING_SEEN_FOR = 6.0


def run(engine, args, tuning=False):
    """Run SARV until q is pressed or the camera goes away.

    ``tuning`` shows the numbers behind each decision and performs
    nothing, which is the mode to use when a gesture will not fire.
    """

    import cv2

    from vision import Camera, CameraError, HandTracker, Presence, TrackerError
    from vision import gestures as gesture_module
    from vision import hand_state, motion
    import vision
    from ui import overlay

    router = GestureRouter(cooldown=engine.config.gesture_cooldown_seconds)
    presence = Presence()
    frame_times = deque(maxlen=30)
    last_result = None
    last_result_at = 0.0
    last_swipe = None
    last_swipe_at = 0.0
    last_hand_at = time.time()

    def remember(result):
        nonlocal last_result, last_result_at
        last_result, last_result_at = result, time.time()

    engine.on_result = remember

    for warning in engine.preflight():
        print(f"  ! {warning}\n")

    try:
        tracker = HandTracker().open()
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        # More pixels reach further: a hand three metres off is about
        # twenty pixels across at 640, which is not much to find joints in.
        camera = Camera(
            args.camera,
            width=640 if getattr(args, "near", False) else 1280,
            height=480 if getattr(args, "near", False) else 720,
        ).open()
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        tracker.close()
        return 1

    print(banner(engine, router, args, tuning), flush=True)

    show_window = not args.no_window

    # Off by default: a list of gestures along the bottom of the preview
    # stops being a reference after the first minute and starts being
    # something to look past.
    show_legend = False

    try:
        for frame in camera.frames():
            frame_times.append(time.time())
            hand = tracker.track(frame)

            # Someone walking past at the back of the room is not
            # gesturing at us.
            if hand is not None and not hand_state.is_prominent(hand):
                hand = None
            gesture = None

            if hand is None:
                # Not immediately: a fast gesture blurs, and a blurred hand
                # is a hand MediaPipe misses for a frame or two.
                if presence.missing(time.time()):
                    vision.reset_state()
                    router.forget()
            else:
                last_hand_at = time.time()
                presence.seen(time.time())
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

            if last_result and time.time() - last_result_at > RESULT_SHOWN_FOR:
                last_result = None

            overlay.draw_gesture(cv2, frame, gesture)
            overlay.draw_result(cv2, frame, last_result)
            overlay.draw_legend(cv2, frame, router.mapping(), show_legend)
            overlay.draw_hint(cv2, frame,
                              out_of_range(time.time() - last_hand_at, args))

            if tuning:
                state = motion.debug_state()
                state["fps"] = frame_rate(frame_times)
                overlay.draw_tuning(cv2, frame, state, motion, hand_state)

            cv2.imshow("SARV", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("h"):
                show_legend = not show_legend

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


def out_of_range(quiet_for, args):
    """What to say when no hand has been seen for a while."""

    if quiet_for < NOTHING_SEEN_FOR:
        return ""

    if getattr(args, "near", False):
        return ("no hand seen -- move closer, or restart without --near "
                "for more range")

    return "no hand seen -- move closer, or into better light"


def frame_rate(times):
    """Frames per second over the last second or so of frames."""

    if len(times) < 2:
        return None

    elapsed = times[-1] - times[0]

    return (len(times) - 1) / elapsed if elapsed > 0 else None


def banner(engine, router, args, tuning):
    lines = [
        "",
        f"  SARV  --  {engine.controller.name}"
        f"{'  [TUNING: gestures shown, NO commands run]' if tuning else ''}"
        f"{'  [DRY RUN]' if engine.config.dry_run and not tuning else ''}",
        "",
    ]

    if not tuning:
        for gesture, command in router.mapping().items():
            lines.append(f"    {gesture:<12} {command}")
        lines.append("")

    if tuning:
        lines.append("  nothing will be controlled -- run without --tune for that")
        lines.append("")

    lines += [
        "  q in the window to stop, h for the gesture list"
        if not args.no_window else "  ctrl+c to stop",
        "",
    ]

    return "\n".join(lines)
