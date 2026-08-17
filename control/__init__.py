"""QRUDO Control Engine: commands in, OS actions out.

The Computer Control Engine's public surface is deliberately tiny::

    from control import Command, ControlEngine

    engine = ControlEngine()
    result = engine.execute(Command.VOLUME_UP)
    if not result.ok:
        print(result.error)

Nothing here imports MediaPipe or OpenCV, so this half can be developed and
tested without a camera.  The two halves meet only in integration/bridge.py.
"""

from .commands import ACTIONABLE_COMMANDS, Command, Status, parse_command
from .config import ControlConfig
from .executor import (
    CommandResult,
    ControlEngine,
    ControlError,
    Controller,
    UnsupportedCommand,
    get_controller,
)

__all__ = [
    "ACTIONABLE_COMMANDS",
    "Command",
    "CommandResult",
    "ControlConfig",
    "ControlEngine",
    "ControlError",
    "Controller",
    "Status",
    "UnsupportedCommand",
    "get_controller",
    "parse_command",
]

__version__ = "1.0.0"
