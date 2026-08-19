#!/usr/bin/env python3
"""QRUDO -- control your computer with hand gestures.

    python main.py                       # run QRUDO: camera on, gestures live
    python main.py --tune                # camera on, gestures shown, nothing done
    python main.py --check               # what this machine can control
    python main.py --selftest            # run all seven commands, then restore
    python main.py --simulate            # keyboard instead of camera
    python main.py --hotkeys             # ctrl+alt+U/D/P/L/R/B/N from any app
    python main.py --command VOLUME_UP   # fire one command and exit
    python main.py --voice               # the voice assistant alone -- no camera
    python main.py --gesture             # the camera gesture loop
    python main.py --voice --gesture     # both: camera gestures and voice

Add --dry-run to any of them to watch without touching the machine.
"""

from __future__ import annotations

import argparse
import sys
import time

from control import ControlConfig, ControlEngine, log, parse_command


def build_parser():
    parser = argparse.ArgumentParser(
        prog="qrudo", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tune", action="store_true",
                      help="camera with the gesture numbers shown, no commands")
    mode.add_argument("--calibrate", action="store_true",
                      help="measure the thresholds from your own hand")
    mode.add_argument("--check", action="store_true",
                      help="print backend capabilities and permission warnings")
    mode.add_argument("--selftest", action="store_true",
                      help="run every command once and restore the machine")
    mode.add_argument("--simulate", action="store_true",
                      help="keyboard simulator (U/D/P/L/R/B/N)")
    mode.add_argument("--hotkeys", action="store_true",
                      help="listen for ctrl+alt+U/D/P/L/R/B/N from any app")
    mode.add_argument("--command", metavar="NAME",
                      help="execute a single command, e.g. VOLUME_UP")
    mode.add_argument("--report", action="store_true",
                      help="reliability numbers from the command log: "
                           "misfire rate, per-command counts")
    mode.add_argument("--update", action="store_true",
                      help="fetch and install a newer QRUDO, if one exists")
    mode.add_argument("--ui", action="store_true",
                      help="the application window: two pages, buttons, "
                           "the camera in a corner")

    parser.add_argument("--voice", action="store_true",
                        help="listen for voice commands -- alone it is a "
                             "microphone assistant (needs "
                             "requirements-voice.txt)")
    parser.add_argument("--gesture", action="store_true",
                        help="the camera gesture loop (implicit when a "
                             "camera mode or --ui is asked for)")

    parser.add_argument("--dry-run", action="store_true",
                        help="log commands without performing them")
    parser.add_argument("--no-window", action="store_true",
                        help="run without the preview, so nothing takes focus")
    parser.add_argument("--camera", type=int, default=0, metavar="N",
                        help="which camera to use (default 0)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="start without the first-run setup, using the "
                             "default thresholds")
    parser.add_argument("--near", action="store_true",
                        help="a smaller picture, for a machine that cannot "
                             "keep up; costs about half the range")
    parser.add_argument("--far", action="store_true",
                        help=argparse.SUPPRESS)  # now the default; kept so
                                                 # older instructions still run
    parser.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                        help="count down before --command, so you can click "
                             "the video first")
    parser.add_argument("--config", metavar="PATH", help="JSON config file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="debug logging")
    return parser


def voice_only_requested(args) -> bool:
    """Whether ``--voice`` is the whole request: a voice assistant alone.

    Any camera/gesture surface -- the application window, an explicit
    ``--gesture``, or one of the camera mode flags -- means the gesture
    loop runs, and voice rides it as a second input.  With none of those,
    ``--voice`` is the entire program: no camera is opened, so a machine
    with a microphone but no camera still gets its commands.
    """

    camera_mode = (args.tune, args.calibrate, args.check, args.selftest,
                   args.simulate, args.hotkeys, args.command, args.ui)
    return bool(args.voice) and not args.gesture and not any(camera_mode)


def _warm_font_cache():
    """Point matplotlib's cache at a stable dir and build it once.

    Done before anything opens the camera.  It is cheap when the cache
    already exists (a stat and return) and slow exactly once -- the
    first launch on a device -- which is far better than that slowness
    landing on the main thread mid-startup with the camera waiting.
    Any failure here is swallowed: a missing font cache must never stop
    QRUDO starting.
    """

    import os
    import sys as _sys

    if "matplotlib" in _sys.modules:
        # Already imported before we could point it anywhere -- nothing
        # to do, and forcing the dir now would not move its cache.
        return

    try:
        from paths import data_dir
        cache = data_dir() / "mpl-cache"
        cache.mkdir(parents=True, exist_ok=True)
        # Force, not setdefault: a packaged app may already carry an
        # MPLCONFIGDIR pointing inside the read-only bundle, where the
        # cache cannot be written and so is rebuilt every launch.
        os.environ["MPLCONFIGDIR"] = str(cache)
        import matplotlib.font_manager  # noqa: F401  (builds the cache)
    except Exception:
        pass


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Keep matplotlib's font cache in a persistent, writable place, and
    # build it once now rather than on the first frame.  mediapipe
    # imports matplotlib while the hand tracker starts, and matplotlib
    # builds its font cache the first time it is imported without one --
    # for a packaged app that is the first launch after every build, on
    # the main thread, for tens of seconds, before the camera opens.
    # That was "the camera did not switch on" and "I have to open it
    # twice".  A stable cache dir means it is built once ever; warming
    # it here means it never blocks the camera.
    _warm_font_cache()

    # The report reads a log file with the standard library alone, so it
    # must never trigger the first-run install below.
    if args.report:
        from control import report
        return report.run(ControlConfig.load(args.config))

    # Updating is likewise stdlib-only, and it would be absurd for the
    # updater to first install an environment it is about to replace.
    if args.update:
        import selfupdate
        return selfupdate.run_cli()

    # A machine that has never seen QRUDO cannot import the vision stack.
    # This builds .venv/ and re-launches through it, once per device --
    # and costs a few milliseconds everywhere else.  It must come before
    # anything that touches the heavy imports.
    import bootstrap
    bootstrap.ensure()

    config = ControlConfig.load(args.config)

    if args.dry_run:
        config.dry_run = True

    # The camera modes and the simulator own stdout, so log to the file
    # only unless asked.
    log.setup(config.log_dir, console=args.verbose,
              level=10 if args.verbose else 20)

    # Voice alone is the whole app: a microphone assistant that never
    # imports vision and never opens a camera, so a machine with a
    # microphone but no camera still gets its commands.  Any camera
    # surface below (--ui, --gesture, a camera mode) runs the gesture
    # loop instead, with voice riding it as a second input.
    if voice_only_requested(args):
        from integration.voice import run_voice_only

        return run_voice_only(ControlEngine(config=config))

    # Thresholds measured from this machine's camera, if there are any.
    import vision

    measured = vision.load_and_apply()

    # Gestures the user has taught, loaded into the isolated matcher.
    # An empty or absent store is normal and costs nothing.
    from vision import custom
    custom.load()

    # The one gesture threshold that is a preference rather than a
    # measurement, so it lives with the settings and not the calibration.
    vision.motion.set_cooldown(config.gesture_cooldown_seconds)

    for note in getattr(measured, "pulled", ()):
        print(f"  ! {note}"
              f"\n    (outside the range QRUDO trusts, so it was pulled back)\n")

    for note in getattr(measured, "notes", ()):
        print(f"  ! {note}"
              f"\n    that one is still the built-in guess\n")

    if measured is not None and getattr(measured, "incomplete", ()):
        print(f"  ! your calibration predates "
              f"{', '.join(measured.incomplete)}; those are still guessed."
              f"\n    run --calibrate again to measure them.\n")

    if args.calibrate:
        from integration import calibrate
        return calibrate.run(args)

    engine = ControlEngine(config=config)

    if args.check:
        return show_capabilities(engine, config)

    if args.selftest:
        from control import selftest
        return selftest.run(engine)

    if args.simulate:
        from control import simulator
        return simulator.run(engine, seek_delay=args.delay or 3.0)

    if args.hotkeys:
        from control import hotkeys
        return hotkeys.run(engine)

    if args.command:
        return run_one(engine, args)

    # One QRUDO at a time, because there is one camera.  A second launch
    # -- a stray double-click while the window is behind others -- would
    # otherwise fail to open the camera the first one holds and show
    # "could not open camera 0" with no idea why.  The lock is held for
    # the life of the process and freed automatically when it ends.
    import singleton

    lock = singleton.SingleInstance()

    try:
        lock.acquire()
    except singleton.AlreadyRunning:
        print("  QRUDO is already running -- look for its window.\n"
              "  (only one copy can use the camera at a time.)",
              file=sys.stderr)
        return 0

    # First time in front of this camera: measure the hand before using
    # it.  Nobody discovers a --calibrate flag on their own, and the
    # thresholds are guesses until somebody runs it -- so it is part of
    # starting up, not an option.  It teaches the gestures on the way
    # through, since it has to ask for each one anyway.
    if measured is None and not args.skip_setup:
        print(first_run_notice())

        from integration import calibrate

        calibrate.run(args)
        vision.load_and_apply()

    # The packaged app opens the application window by default -- a
    # product is a window, not a video feed -- and any checkout gets it
    # with --ui.  If Tk is missing (some minimal Pythons), the classic
    # window still stands, so a broken toolkit never blocks the camera.
    if args.ui or (getattr(sys, "frozen", False) and not args.tune):
        try:
            from ui.app import run_app
        except Exception as exc:
            print(f"  ! the application window is unavailable ({exc}); "
                  f"classic window instead")
        else:
            return run_app(engine, args)

    from integration.runner import run
    return run(engine, args, tuning=args.tune)


def first_run_notice():
    return "\n".join([
        "",
        "  First run: a short setup, about a minute.",
        "",
        "  It measures your hand so the gestures suit your camera and how",
        "  far away you stand, and shows you what each one does.  Stand",
        "  where you actually mean to use QRUDO.",
        "",
        "  --skip-setup starts without it, using thresholds that were",
        "  guessed rather than measured.",
        "",
    ])


def show_capabilities(engine, config):
    print(f"backend: {engine.controller.name}")
    print(f"config : volume {config.volume_step}%, "
          f"brightness {config.brightness_step}%, "
          f"seek {config.seek_seconds}s ({config.seek_presses} key press(es)), "
          f"cooldown {config.cooldown_seconds}s")

    # The voice stack is optional, so the check only asks whether it is
    # installed -- asking for a microphone here would be rude.
    from integration import voice as voice_mod

    if voice_mod.available():
        print("voice  : requirements-voice.txt installed "
              "(faster-whisper, openwakeword, sounddevice)")
    else:
        print("voice  : requirements-voice.txt not installed -- "
              "say a command and nothing happens")

    warnings = engine.preflight()

    for warning in warnings:
        print(f"  ! {warning}")

    if not warnings:
        print("  all controls available")

    return 0


def run_one(engine, args):
    try:
        command = parse_command(args.command)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Seeking sends arrow keys, which go to whichever window has keyboard
    # focus.  Run from a terminal, that is the terminal -- so give the user
    # a moment to click the video they actually want to seek.
    for remaining in range(int(args.delay), 0, -1):
        print(f"  {command} in {remaining}s -- click the window you want it "
              f"to hit", end="\r", flush=True)
        time.sleep(1)

    if args.delay:
        print(" " * 70, end="\r")

    result = engine.execute(command, force=True, source="cli")
    print(result)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
