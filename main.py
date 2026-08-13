#!/usr/bin/env python3
"""SARV control-layer CLI.

    python main.py --simulate            # keyboard-driven command simulator
    python main.py --selftest            # verify all seven commands
    python main.py --command VOLUME_UP   # fire one command and exit
    python main.py --check               # report permissions/capabilities only

Add --dry-run to any of these to log commands without touching the OS.
"""

from __future__ import annotations

import argparse
import sys

from sarv import ControlConfig, ControlEngine, log, parse_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sarv", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true",
                      help="keyboard simulator (U/D/P/L/R/B/N)")
    mode.add_argument("--selftest", action="store_true",
                      help="run every command once and restore the machine")
    mode.add_argument("--command", metavar="NAME",
                      help="execute a single command, e.g. VOLUME_UP")
    mode.add_argument("--check", action="store_true",
                      help="print backend capabilities and permission warnings")

    parser.add_argument("--dry-run", action="store_true",
                        help="log commands without performing them")
    parser.add_argument("--config", metavar="PATH", help="path to a JSON config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = ControlConfig.load(args.config)
    if args.dry_run:
        config.dry_run = True

    # The simulator owns stdout, so send log lines to stderr only when asked.
    log.setup(config.log_dir, console=args.verbose,
              level=10 if args.verbose else 20)

    engine = ControlEngine(config=config)

    if args.check:
        warnings = engine.preflight()
        print(f"backend: {engine.controller.name}")
        print(f"config : volume {config.volume_step}%, brightness {config.brightness_step}%, "
              f"seek {config.seek_seconds}s ({config.seek_presses} key press(es)), "
              f"cooldown {config.cooldown_seconds}s")
        for warning in warnings:
            print(f"  ! {warning}")
        if not warnings:
            print("  all controls available")
        return 0

    if args.selftest:
        from sarv import selftest
        return selftest.run(engine)

    if args.simulate:
        from sarv import simulator
        return simulator.run(engine)

    try:
        command = parse_command(args.command)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = engine.execute(command, force=True)
    print(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
