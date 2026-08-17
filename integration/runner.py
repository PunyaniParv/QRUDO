"""The live loop: camera to gesture to command, once per frame.

Everything it touches is somebody else's: vision finds the hand and names
what it is doing, bridge turns that name into a command, control performs
it, ui draws it.  This file only decides the order.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

from control import log as control_log

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

#: How long a visible hand may read as unknown before the screen says
#: which test is refusing it.  Short enough to answer while the pose is
#: still being held, long enough that transitions never flash it.
UNKNOWN_SAID_AFTER = 1.0


def run(engine, args, tuning=False):
    """Run QRUDO until q is pressed or the camera goes away.

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
    hand_present = False
    frame_times = deque(maxlen=30)
    last_result = None
    last_result_at = 0.0
    last_swipe = None
    last_swipe_at = 0.0
    last_hand_at = time.time()
    unknown_since = None
    refused = ""
    spans_logged_at = 0.0
    vision_log = control_log.get_logger("vision")

    def remember(result):
        nonlocal last_result, last_result_at
        last_result, last_result_at = result, time.time()

    engine.on_result = remember

    warnings = engine.preflight()

    for warning in warnings:
        print(f"  ! {warning}\n")

    # Launched from an icon, there is no terminal for those lines to
    # reach -- the camera window is the only place QRUDO can speak.  So
    # the one permission macOS never prompts for by itself rides the
    # overlay's hint line, and the packaged app walks the user to the
    # exact Settings pane, once per installation.
    permission_hint = next(
        ("play/pause and seeking need Accessibility -- System Settings "
         "> Privacy & Security > Accessibility"
         for warning in warnings if "Accessibility" in warning), "")

    walk_to_settings(warnings)

    # Keep the target fresh in the background, and let ctrl+shift+arrows
    # step it from any app.  Both are best effort: a machine that cannot
    # watch the keyboard still has the pointing gesture and the config.
    engine.targets.start()

    from control import hotkeys

    hotkeys.watch_targets(engine)

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
            width=1600 if getattr(args, "far", False) else 640,
            height=1200 if getattr(args, "far", False) else 480,
        ).open()
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        tracker.close()
        return 1

    print(banner(engine, router, args, tuning), flush=True)
    print(picture(camera), flush=True)

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
                    hand_present = False
            else:
                # Arriving is judged by the same grace as leaving, so a
                # hand lost to blur mid-gesture does not "arrive" twice.
                if not hand_present:
                    router.hand_arrived(time.time())
                    hand_present = True
                last_hand_at = time.time()
                presence.seen(time.time())
                gesture = gesture_module.detect_gesture(hand)
                swipe = motion.detect_swipe(hand)

                # A hand plainly shown and not recognised is a question
                # the user is already asking, so the answer goes on
                # screen -- which test refused, and what it measured --
                # and into the log, where a bug report can carry it.
                # This used to live in --tune alone, and every report of
                # a pose reading as unknown arrived without the one line
                # that said why.
                if gesture == "UNKNOWN":
                    if unknown_since is None:
                        unknown_since = time.time()
                    elif time.time() - unknown_since > UNKNOWN_SAID_AFTER:
                        refused = gesture_module.explain(hand)
                else:
                    unknown_since = None
                    refused = ""

                if time.time() - spans_logged_at > 1.0:
                    spans_logged_at = time.time()
                    vision_log.info(
                        "hand ext=%s gesture=%s why=%s",
                        {name: round(span, 2) for name, span
                         in hand_state.finger_span(hand).items()},
                        gesture, gesture_module.explain(hand))

                if not tuning:
                    command = router.update(gesture, swipe)

                    if command is not None:
                        # submit, not execute: a brightness change can take
                        # over a second on Windows, and the camera must not
                        # wait for it.
                        engine.submit(command, source="gesture")

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
                              refused
                              or out_of_range(time.time() - last_hand_at,
                                              args)
                              or permission_hint)

            if tuning:
                state = motion.debug_state()
                state["fps"] = frame_rate(frame_times)
                overlay.draw_tuning(cv2, frame, state, motion, hand_state)

            cv2.imshow("QRUDO", frame)

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


def walk_to_settings(warnings):
    """Open the Accessibility pane for the packaged app, once ever.

    Camera permission asks for itself the first time the camera opens,
    with QRUDO's name on the dialog.  Accessibility never asks -- it
    just silently does nothing -- and an app launched from an icon has
    no terminal to explain that in.  So the first launch that finds
    the permission missing opens the exact Settings pane, and a marker
    in the data folder keeps every later launch from nagging: from
    then on the overlay's hint line carries the reminder instead.

    Run from a terminal this does nothing at all -- the printed
    warning is readable there, and opening windows nobody asked for is
    not how a command line behaves.
    """

    if not getattr(sys, "frozen", False) or sys.platform != "darwin":
        return

    if not any("Accessibility" in warning for warning in warnings):
        return

    from paths import data_dir

    marker = data_dir() / ".accessibility-walked"

    if marker.exists():
        return

    marker.write_text("")

    import subprocess

    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference."
                 "security?Privacy_Accessibility"],
        check=False)


def picture(camera):
    """What the camera gave, and what that buys, said out loud.

    Worth the lines, because both facts are otherwise invisible.  A
    camera answering a request for a big picture with a widescreen one
    drops a quarter of its height, and the only symptom is close-up
    gestures failing while distant ones improve.  And the range gate is
    derived from the delivered width -- the hand profile carries the
    hand, the camera carries only this -- so plugging in a different
    camera should change this line and nothing else.
    """

    from vision import hand_state

    shape = camera.shape

    if not shape:
        return ""

    width, height = shape
    gate = hand_state.set_camera(width)
    line = (f"  camera : {width}x{height} -- hands under "
            f"{hand_state.MIN_HAND_PIXELS}px ignored"
            f" ({gate:.1%} of frame)")

    if height < camera.height:
        # Shorter than asked for: rows genuinely went missing, which is
        # a widescreen crop of a 4:3 sensor.  A native-widescreen sensor
        # never lands here -- its taller mode was taken at open, and
        # widescreen is its whole picture, not a crop.
        return (line + "\n           the view lost height, so close-up"
                       " gestures can run out of picture")

    if abs(width / height - 4 / 3) > 0.05:
        return line + "  (widescreen sensor, using its full view)"

    return line


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


def build_stamp():
    """Which code and whose hand this session is actually running.

    Three diagnosis sessions have now been spent on invisible state --
    a stale checkout shows last week's bugs in this week's report, and
    default thresholds pass for calibrated ones -- so both facts go on
    the banner, where every screenshot carries them for free.
    """

    import subprocess

    try:
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%h %ad", "--date=short"],
            capture_output=True, text=True, timeout=2.0,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        commit = ""

    from vision import calibration

    measured = calibration.Calibration.load() is not None

    profile = ("hand profile: measured" if measured
               else "hand profile: DEFAULTS -- run --calibrate")

    stamp = f"  build  : {commit or 'unknown'}  --  {profile}"

    behind = _commits_behind()

    if behind:
        stamp += (f"\n  ! this machine is {behind} commit(s) behind -- "
                  f"git pull to catch up")

    return stamp


def _commits_behind():
    """How far this checkout trails the shared main, or 0 if unknowable.

    A stale checkout shows last week's bugs in this week's report -- a
    Windows session once reported three of them at one sitting, all
    already fixed -- so the machine says so itself at startup.  Offline,
    slow, or not a checkout at all: silently nothing, never a delay
    worth noticing and never a crash.
    """

    import subprocess

    here = Path(__file__).resolve().parent

    try:
        subprocess.run(["git", "fetch", "--quiet"], cwd=here, timeout=2.5,
                       capture_output=True)
        count = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            cwd=here, timeout=2.0, capture_output=True, text=True,
        ).stdout.strip()

        return int(count) if count.isdigit() else 0
    except Exception:
        return 0


def banner(engine, router, args, tuning):
    lines = [
        "",
        f"  QRUDO  --  {engine.controller.name}"
        f"{'  [TUNING: gestures shown, NO commands run]' if tuning else ''}"
        f"{'  [DRY RUN]' if engine.config.dry_run and not tuning else ''}",
        build_stamp(),
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
