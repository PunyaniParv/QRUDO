#!/usr/bin/env python3
"""SARV -- control your computer with hand gestures.

    python main.py                       # run SARV: camera on, gestures live
    python main.py --tune                # camera on, gestures shown, nothing done
    python main.py --check               # what this machine can control
    python main.py --selftest            # run all seven commands, then restore
    python main.py --simulate            # keyboard instead of camera
    python main.py --hotkeys             # ctrl+alt+U/D/P/L/R/B/N from any app
    python main.py --command VOLUME_UP   # fire one command and exit

Add --dry-run to any of them to watch without touching the machine.
"""

from __future__ import annotations

import argparse
import sys
import time

from control import ControlConfig, ControlEngine, log, parse_command


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sarv", description=__doc__,
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


def main(argv=None):
    args = build_parser().parse_args(argv)

    config = ControlConfig.load(args.config)

    if args.dry_run:
        config.dry_run = True

    # The camera modes and the simulator own stdout, so log to the file
    # only unless asked.
    log.setup(config.log_dir, console=args.verbose,
              level=10 if args.verbose else 20)

    # Thresholds measured from this machine's camera, if there are any.
    import vision

    measured = vision.load_and_apply()

    # The one gesture threshold that is a preference rather than a
    # measurement, so it lives with the settings and not the calibration.
    vision.motion.SWIPE_COOLDOWN = config.gesture_cooldown_seconds

    for note in getattr(measured, "pulled", ()):
        print(f"  ! {note}"
              f"\n    (outside the range SARV trusts, so it was pulled back)\n")

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

    from integration.runner import run
    return run(engine, args, tuning=args.tune)


def first_run_notice():
    return "\n".join([
        "",
        "  First run: a short setup, about a minute.",
        "",
        "  It measures your hand so the gestures suit your camera and how",
        "  far away you stand, and shows you what each one does.  Stand",
        "  where you actually mean to use SARV.",
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

    result = engine.execute(command, force=True)
    print(result)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
