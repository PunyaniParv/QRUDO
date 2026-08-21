"""Safe builtin tool wrappers around existing ControlEngine commands.

Each tool is a declarative Tool instance registered with the global ToolRegistry.
Handlers call the ControlEngine with source="ai" and never directly access OS APIs.

Available tools:
- volume_up, volume_down
- brightness_up, brightness_down
- play_pause, rewind, forward
- target_next, target_prev
"""

from __future__ import annotations

import json
from typing import Any, Dict

from control import Command, ControlEngine
from .registry import Tool


#: Schema definitions for each builtin tool's parameters
VOLUME_UP_SCHEMA = {"type": "object", "properties": {"step": {"type": "integer"}}, "required": []}
VOLUME_DOWN_SCHEMA = {"type": "object", "properties": {"step": {"type": "integer"}}, "required": []}
BRIGHTNESS_UP_SCHEMA = {"type": "object", "properties": {"step": {"type": "integer"}}, "required": []}
BRIGHTNESS_DOWN_SCHEMA = {"type": "object", "properties": {"step": {"type": "integer"}}, "required": []}
PLAY_PAUSE_SCHEMA = {"type": "object", "properties": {}, "required": []}
REWIND_SCHEMA = {"type": "object", "properties": {"seconds": {"type": "integer"}}, "required": []}
FORWARD_SCHEMA = {"type": "object", "properties": {"seconds": {"type": "integer"}}, "required": []}
TARGET_NEXT_SCHEMA = {"type": "object", "properties": {}, "required": []}
TARGET_PREV_SCHEMA = {"type": "object", "properties": {}, "required": []}


def _make_handler(command: Command) -> Any:
    """Create a handler function for a ControlEngine Command."""

    def handler(engine: ControlEngine, args: Dict[str, Any], *, config: ControlConfig | None = None) -> Dict[str, Any]:
        step = args.get("step", 1) if command in (Command.VOLUME_UP, Command.VOLUME_DOWN, Command.BRIGHTNESS_UP, Command.BRIGHTNESS_DOWN) else 1
        seconds = args.get("seconds", 10) if command in (Command.REWIND, Command.FORWARD) else None
        engine.execute(command, force=False, source="ai", payload=json.dumps({}))
        return {"command": command.value, "source": "ai", "args": args, "status": "executed"}

    return handler


def register_tools(registry: registry.ToolRegistry) -> None:
    """Register all builtin safe tools with the given registry."""
    # Volume tools
    registry.register(Tool(
        name="volume_up",
        description="Increase volume by one step",
        parameters=VOLUME_UP_SCHEMA,
        handler=_make_handler(Command.VOLUME_UP),
        confirmation_required=False,
    ))

    registry.register(Tool(
        name="volume_down",
        description="Decrease volume by one step",
        parameters=VOLUME_DOWN_SCHEMA,
        handler=_make_handler(Command.VOLUME_DOWN),
        confirmation_required=False,
    ))

    # Brightness tools
    registry.register(Tool(
        name="brightness_up",
        description="Increase brightness by one step",
        parameters=BRIGHTNESS_UP_SCHEMA,
        handler=_make_handler(Command.BRIGHTNESS_UP),
        confirmation_required=False,
    ))

    registry.register(Tool(
        name="brightness_down",
        description="Decrease brightness by one step",
        parameters=BRIGHTNESS_DOWN_SCHEMA,
        handler=_make_handler(Command.BRIGHTNESS_DOWN),
        confirmation_required=False,
    ))

    # Media tools
    registry.register(Tool(
        name="play_pause",
        description="Play or pause media",
        parameters=PLAY_PAUSE_SCHEMA,
        handler=_make_handler(Command.PLAY_PAUSE),
        confirmation_required=False,
    ))

    registry.register(Tool(
        name="rewind",
        description="Rewind media by N seconds",
        parameters=REWIND_SCHEMA,
        handler=_make_handler(Command.REWIND),
        confirmation_required=False,
    ))

    registry.register(Tool(
        name="forward",
        description="Forward media by N seconds",
        parameters=FORWARD_SCHEMA,
        handler=_make_handler(Command.FORWARD),
        confirmation_required=False,
    ))

    # Target tools
    registry.register(Tool(
        name="target_next",
        description="Move to next target (next track/app tab)",
        parameters=TARGET_NEXT_SCHEMA,
        handler=_make_handler(Command.TARGET_NEXT),
        confirmation_required=False,
    ))

    registry.register(Tool(
        name="target_prev",
        description="Move to previous target (previous track/app tab)",
        parameters=TARGET_PREV_SCHEMA,
        handler=_make_handler(Command.TARGET_PREV),
        confirmation_required=False,
    ))