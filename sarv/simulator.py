"""Command simulator (spec section E).

Generates commands from the keyboard so the whole control layer can be built
and demoed before the camera exists::

    python main.py --simulate

Keys follow the spec's U/D/P/L/R/B/N mapping.  When stdin is not a terminal the
simulator reads keys from stdin one line at a time instead, which makes it
scriptable::

    printf 'u\\nu\\nd\\n' | python main.py --simulate
"""

from __future__ import annotations

import sys

from .commands import Command
from .controller import ControlEngine

#: Spec E: "U/D/P/L/R/B/N for the seven V1 commands".
KEY_MAP: dict[str, Command] = {
    "u": Command.VOLUME_UP,
    "d": Command.VOLUME_DOWN,
    "p": Command.PLAY_PAUSE,
    "l": Command.REWIND,        # L for left
    "r": Command.FORWARD,       # R for right
    "b": Command.BRIGHTNESS_UP,
    "n": Command.BRIGHTNESS_DOWN,
}

QUIT_KEYS = {"q", "\x03", "\x04"}  # q, Ctrl-C, Ctrl-D


def help_text(engine: ControlEngine) -> str:
    lines = [
        "",
        f"  SARV command simulator  --  backend: {engine.controller.name}"
        f"{'  [DRY RUN]' if engine.config.dry_run else ''}",
        "",
    ]
    for key, command in KEY_MAP.items():
        lines.append(f"    {key}   {command.value}")
    lines += [
        "    ?   this help",
        "    q   quit",
        "",
        f"  volume step {engine.config.volume_step}%   "
        f"brightness step {engine.config.brightness_step}%   "
        f"seek {engine.config.seek_seconds}s",
        "",
    ]
    return "\n".join(lines)


def run(engine: ControlEngine | None = None) -> int:
    engine = engine or ControlEngine()

    for warning in engine.preflight():
        print(f"  ! {warning}\n", file=sys.stderr)

    print(help_text(engine))

    for key in _key_stream():
        if key in QUIT_KEYS:
            break
        if key in ("?", "h"):
            print(help_text(engine))
            continue
        command = KEY_MAP.get(key.lower())
        if command is None:
            if key.strip():
                print(f"    unmapped key {key!r} -- press ? for help")
            continue
        # force=True: a keypress is a deliberate human action, so it should
        # never be swallowed by the gesture debounce.
        result = engine.execute(command, force=True)
        marker = "ok " if result.ok else "ERR"
        print(f"    [{marker}] {result.command}: {result.detail or result.error}")

    print("\n  bye.")
    return 0


def _key_stream():
    """Yield one keypress at a time from a terminal, or one line from a pipe."""
    if not sys.stdin.isatty():
        for line in sys.stdin:
            yield line.strip()[:1] or "\n"
        return
    # Reading one key without Enter is done differently on each OS, and the
    # POSIX modules do not even exist on Windows.
    yield from (_windows_keys() if sys.platform == "win32" else _posix_keys())


def _windows_keys():
    import msvcrt

    while True:
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            msvcrt.getwch()  # arrow/function keys arrive as two reads; drop both
            continue
        yield char


def _posix_keys():
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # cbreak, not raw: Ctrl-C still interrupts
        while True:
            char = sys.stdin.read(1)
            if not char:
                return
            yield char
    except KeyboardInterrupt:
        return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


if __name__ == "__main__":
    raise SystemExit(run())
