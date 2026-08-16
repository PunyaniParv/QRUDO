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

    # Not OS actions: these move which app the targeted commands land in.
    # They live in the same vocabulary so a gesture, a hotkey and the
    # simulator can all reach them through the one pipe -- cooldown,
    # logging and results included.
    TARGET_NEXT = "TARGET_NEXT"
    TARGET_PREV = "TARGET_PREV"

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
    Command.TARGET_NEXT,
    Command.TARGET_PREV,
)

#: The commands that change where commands go rather than touching the
#: OS.  They are handled inside the engine -- no backend method exists
#: for them, deliberately, so a platform backend stays a list of OS
#: actions and nothing else.
TARGET_COMMANDS: tuple[Command, ...] = (
    Command.TARGET_NEXT,
    Command.TARGET_PREV,
)


#: Seeking has no system-wide key on either platform, so it is sent as arrow
#: keys -- which land in whichever window has keyboard focus.  Everything else
#: is a media key or a system call and ignores focus entirely.
FOCUS_SENSITIVE: tuple[Command, ...] = (Command.REWIND, Command.FORWARD)

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
