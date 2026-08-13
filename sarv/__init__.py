"""SARV -- gesture-controlled computer control.

The Computer Control Engine's public surface is deliberately tiny::

    from sarv import Command, ControlEngine

    engine = ControlEngine()
    result = engine.execute(Command.VOLUME_UP)
    if not result.ok:
        print(result.error)

Nothing here imports MediaPipe or OpenCV, so the Vision Engine and the Control
Engine can be developed and tested independently.
"""

from .commands import ACTIONABLE_COMMANDS, Command, Status, parse_command
from .config import ControlConfig
from .controller import (
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
