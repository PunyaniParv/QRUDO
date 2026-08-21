"""Read-only context tools that are already safely available.

These tools provide the AI with information about the current state
without any surveillance or arbitrary system inspection.  All are
read-only and depend only on what is already safely exposed by QRUDO.

Available tools:
- capabilities: list of supported capabilities
- current_target: the currently targeted app/target
- available_commands: list of commands the engine knows
"""

from __future__ import annotations

from control import ControlEngine, ControlConfig, Command, ACTIONABLE_COMMANDS
from control.catalog import job_names, apps_for
from ai.tools.registry import Tool


def _get_capabilities() -> List[str]:
    """Return the capabilities this QRUDO instance supports."""
    # Read from the control engine's available commands and config
    caps = ["volume_control", "brightness_control", "media_control", "target_cycling"]

    # Check if catalog is available
    try:
        from control import catalog
        caps.append("catalog_jobs")
    except Exception:
        pass

    # Check if actions are available
    try:
        from control import actions
        caps.append("action_chains")
    except Exception:
        pass

    return caps


def _get_current_target(engine: ControlEngine | None = None) -> str:
    """Return the currently targeted app or target."""
    if engine is not None:
        # Try to get target from engine config
        try:
            config = engine.config
            target = config.app or config.seek_target_app
            if target:
                return target
        except Exception:
            pass
    return ""


def _get_available_commands() -> List[str]:
    """Return the list of commands the control engine knows about."""
    return [c.value for c in ACTIONABLE_COMMANDS]


def register_tools(tool_registry: registry.ToolRegistry) -> None:
    """Register read-only context tools with the given registry."""
    # Capabilities tool - read-only, no engine needed
    tool_registry.register(Tool(
        name="capabilities",
        description="List QRUDO's supported capabilities (read-only)",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda engine, args: {"capabilities": _get_capabilities()},
        confirmation_required=False,
    ))

    # Current target tool - read-only
    tool_registry.register(Tool(
        name="current_target",
        description="Get the currently targeted app/target (read-only)",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda engine, args: {"current_target": _get_current_target(engine)},
        confirmation_required=False,
    ))

    # Available commands tool - read-only
    tool_registry.register(Tool(
        name="available_commands",
        description="List commands the control engine knows (read-only)",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda engine, args: {"available_commands": _get_available_commands()},
        confirmation_required=False,
    ))