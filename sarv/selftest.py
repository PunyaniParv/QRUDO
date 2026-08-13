"""Control testing (spec section F): verify every command independently.

Run with ``python main.py --selftest``.  Each of the seven commands is executed
on its own and the result printed, then the machine is put back the way it was
(volume and brightness restored, PLAY_PAUSE toggled back, a FORWARD undone by a
REWIND) so the test is safe to run mid-demo.
"""

from __future__ import annotations

import time

from .commands import NO_CHANGE, Command
from .controller import ControlEngine

PAUSE_BETWEEN = 0.4


def run(engine: ControlEngine | None = None) -> int:
    engine = engine or ControlEngine()

    print(f"\n  SARV control self-test  --  backend: {engine.controller.name}"
          f"{'  [DRY RUN]' if engine.config.dry_run else ''}\n")
    for warning in engine.preflight():
        print(f"  ! {warning}\n")

    before = _snapshot(engine)
    if before:
        print(f"  before: {before}\n")

    # (command, note) pairs.  The paired command after each one puts the
    # machine back where it started.
    plan = [
        (Command.VOLUME_UP, Command.VOLUME_DOWN),
        (Command.VOLUME_DOWN, Command.VOLUME_UP),
        (Command.BRIGHTNESS_UP, Command.BRIGHTNESS_DOWN),
        (Command.BRIGHTNESS_DOWN, Command.BRIGHTNESS_UP),
        (Command.PLAY_PAUSE, Command.PLAY_PAUSE),
        (Command.FORWARD, Command.REWIND),
        (Command.REWIND, Command.FORWARD),
    ]

    results = []
    for command, undo in plan:
        result = engine.execute(command, force=True)
        results.append(result)
        print(f"  {'PASS' if result.ok else 'FAIL'}  {command.value:<16} "
              f"{result.detail or result.error}  ({result.duration_ms:.0f} ms)")
        time.sleep(PAUSE_BETWEEN)
        # Only undo what actually moved.  At a limit -- brightness already at
        # 100%, volume already at 0% -- the command is a no-op, and undoing it
        # anyway would leave the machine dimmer or quieter than we found it.
        if NO_CHANGE not in result.detail:
            engine.execute(undo, force=True)
            time.sleep(PAUSE_BETWEEN)

    # Behaviour checks that do not touch the OS.
    print()
    _check("Command.NONE is a safe no-op",
           engine.execute(Command.NONE).status == "NOOP")
    _check("unknown command is reported, not raised",
           engine.execute("NOT_A_COMMAND").status == "ERROR")
    time.sleep(engine.config.cooldown_seconds + 0.05)  # clear the window first
    first = engine.execute(Command.VOLUME_UP)
    _check("repeat within cooldown is throttled",
           engine.execute(Command.VOLUME_UP).status == "THROTTLED")
    if first.ok:
        engine.execute(Command.VOLUME_DOWN, force=True)

    after = _snapshot(engine)
    if after:
        print(f"\n  after:  {after}")

    failed = [r for r in results if not r.ok]
    print(f"\n  {len(results) - len(failed)}/{len(results)} commands OK\n")
    return 1 if failed else 0


def _check(label: str, passed: bool) -> None:
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")


def _snapshot(engine: ControlEngine) -> str:
    """Read volume/brightness if the backend exposes them (macOS only)."""
    controller = engine.controller
    parts = []
    if hasattr(controller, "_volume_settings"):
        try:
            settings = controller._volume_settings()
            parts.append(f"volume {settings['volume']}%"
                         + (" (muted)" if settings["muted"] else ""))
        except Exception as exc:
            parts.append(f"volume unreadable ({exc})")
    if hasattr(controller, "_get_brightness"):
        try:
            level = controller._get_brightness()
            if level is not None:
                parts.append(f"brightness {level * 100:.0f}%")
        except Exception:
            pass
    return ", ".join(parts)


if __name__ == "__main__":
    raise SystemExit(run())
