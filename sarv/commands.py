"""Shared command vocabulary for SARV.

This module is the contract between the Vision Engine (Teammate 1) and the
Computer Control Engine (Teammate 2).  It deliberately imports nothing beyond
the standard library -- no MediaPipe, no OpenCV, no OS APIs -- so that both
sides can depend on it without dragging in each other's dependencies.
"""

from __future__ import annotations

from enum import Enum


class Command(str, Enum):
    """The seven V1 commands, plus an explicit "do nothing" signal.

    Inherits from ``str`` so a command can be logged, serialised to JSON, or
    sent over a socket as plain text: ``json.dumps(Command.VOLUME_UP) ==
    '"VOLUME_UP"'``.
    """

    VOLUME_UP = "VOLUME_UP"
    VOLUME_DOWN = "VOLUME_DOWN"
    PLAY_PAUSE = "PLAY_PAUSE"
    REWIND = "REWIND"
    FORWARD = "FORWARD"
    BRIGHTNESS_UP = "BRIGHTNESS_UP"
    BRIGHTNESS_DOWN = "BRIGHTNESS_DOWN"

    # Emitted by the Vision Engine when no gesture is recognised.  Executing it
    # is always a safe no-op, so the vision side never needs a null check.
    NONE = "NONE"

    def __str__(self) -> str:  # nicer log output than "Command.VOLUME_UP"
        return self.value


#: The commands that actually do something, in a stable order (useful for
#: self-tests, menus and the simulator's help text).
ACTIONABLE_COMMANDS: tuple[Command, ...] = (
    Command.VOLUME_UP,
    Command.VOLUME_DOWN,
    Command.PLAY_PAUSE,
    Command.REWIND,
    Command.FORWARD,
    Command.BRIGHTNESS_UP,
    Command.BRIGHTNESS_DOWN,
)


#: Backends put this in their detail string when a command succeeded but moved
#: nothing -- volume already at 100%, brightness already at 0%.  The self-test
#: relies on it: undoing a command that did nothing would leave the machine
#: quieter or dimmer than it found it.
NO_CHANGE = "already at"


class Status(str, Enum):
    """Outcome of an execution attempt."""

    OK = "OK"                    # the OS action was performed
    NOOP = "NOOP"                # Command.NONE -- nothing to do
    THROTTLED = "THROTTLED"      # suppressed by the cooldown / debounce
    UNSUPPORTED = "UNSUPPORTED"  # this platform cannot do it
    ERROR = "ERROR"              # the OS action was attempted and failed

    def __str__(self) -> str:
        return self.value


def parse_command(value: str) -> Command:
    """Parse a command name case-insensitively.

    Raises ``ValueError`` with a helpful message on an unknown name, which is
    what the CLI and any future network layer want.
    """
    try:
        return Command[value.strip().upper()]
    except KeyError:
        known = ", ".join(c.value for c in Command)
        raise ValueError(f"unknown command {value!r}; expected one of: {known}") from None
