"""Tool registry: the single bridge between AI tool calls and QRUDO's control layer.

The registry is the ONLY entry point for translating an AI-generated
ToolCall into a ControlEngine command.  It enforces whitelisting,
argument validation, and confirmation requirements.  Unknown tools are
rejected, malformed arguments are rejected, and handlers never bypass
the registry.

Dependency direction: ai → control (not the reverse).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from control import Command, ControlEngine, commands, catalog, actions as actions_mod
from control.config import ControlConfig
from ..schema import Message, ToolCall, ToolResult


@dataclass
class Tool:
    """A declarative tool that the AI can call.

    Each tool has a name, description, parameter schema, handler, and
    optionally a confirmation requirement.  The registry is the sole
    bridge between tool calls and QRUDO's control execution.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    """JSON schema describing valid arguments."""

    handler: Any
    """Callable that takes (engine, args) and returns a dict or ToolResult."""

    confirmation_required: bool = False
    """When True, the tool cannot execute without explicit user confirmation."""

    # Internal state, not part of the public schema
    _registered: bool = field(init=False, default=False)


class ToolRegistry:
    """Declarative registry of whitelisted tools.

    This is the single bridge between future AI tool calls and
    QRUDO's control layer.  It enforces:
      - Tool names are explicitly whitelisted (unknown tools rejected)
      - Malformed arguments are rejected (schema validation)
      - Handlers never directly access OS APIs
      - Handlers must use the existing ControlEngine/actions safety
      - No arbitrary shell, Python, or filesystem tools
      - Deterministic behavior
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._confirmation_required: set[str] = set()

    def register(self, tool: Tool) -> None:
        """Register a tool with the registry.

        Raises ValueError if a tool with the same name already exists.
        """
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        if tool.confirmation_required:
            self._confirmation_required.add(tool.name)

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
        self._confirmation_required.discard(name)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name, or None if not registered."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def call(self, engine: ControlEngine, tool_call: ToolCall, *, config: ControlConfig | None = None,
             confirmed: bool | None = None) -> ToolResult:
        """Execute a tool call through the registry.

        This is the ONLY path from AI tool calls to ControlEngine execution.

        Returns a ToolResult with the execution outcome.
        """
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                data={},
                error=f"unknown tool '{tool_call.name}'; not whitelisted",
            )

        # Validate arguments against the tool's parameter schema
        try:
            args = self._validate_arguments(tool, tool_call.arguments)
        except (ValueError, TypeError) as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                data={},
                error=f"malformed arguments for '{tool_call.name}': {exc}",
            )

        # Check confirmation requirement
        if tool.confirmation_required and confirmed is None:
            # No confirmation provided yet - reject with instruction
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                data={},
                error=f"tool '{tool_call.name}' requires confirmation (set confirmed=true or provide confirmation)",
            )

        if tool.confirmation_required and confirmed is not True:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                data={},
                error=f"tool '{tool_call.name}' was not confirmed",
            )

        # Execute the handler through the ControlEngine
        try:
            result = tool.handler(engine, args, config=config)
            if isinstance(result, dict):
                # Convert dict result to ToolResult
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=True,
                    data=result,
                    error=None,
                )
            elif isinstance(result, ToolResult):
                return result
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=True,
                    data={"output": str(result)},
                    error=None,
                )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                data={},
                error=f"tool '{tool_call.name}' raised {type(exc).__name__}: {exc}",
            )

    def _validate_arguments(self, tool: Tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool arguments against the parameter schema.

        The schema is a JSON-schema-like dict.  We enforce:
          - required fields present
          - types match
          - no unexpected top-level keys (beyond schema properties)
        """
        schema = tool.parameters

        if "type" in schema and schema["type"] != "object":
            # If schema is not an object type, just try to use args as-is
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be an object")
            return arguments

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for field_name in required:
            if field_name not in arguments:
                raise ValueError(f"required argument '{field_name}' is missing")

        # Validate types for provided fields
        for field_name, field_schema in properties.items():
            if field_name in arguments:
                value = arguments[field_name]
                expected_type = field_schema.get("type")
                if expected_type:
                    type_map = {
                        "string": str,
                        "number": (int, float),
                        "integer": int,
                        "boolean": bool,
                        "array": list,
                        "object": dict,
                    }
                    python_type = type_map.get(expected_type)
                    if python_type and not isinstance(value, python_type):
                        raise ValueError(
                            f"argument '{field_name}' expected type {expected_type}, got {type(value).__name__}"
                        )

        # Reject any keys not in the schema properties
        allowed_keys = set(properties.keys())
        for key in arguments:
            if key not in allowed_keys:
                raise ValueError(f"unexpected argument '{key}'")

        return arguments


#: The global tool registry instance - the single bridge between AI tool
#: calls and QRUDO's control layer.
_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance, creating it if needed."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (use for testing)."""
    global _global_registry
    _global_registry = ToolRegistry()


#: Pre-built tools that are automatically registered when the tools module
#: is imported.  These are safe wrappers around existing ControlEngine
#: commands and catalog actions.
_pre_built_tools_registered: bool = False


def ensure_prebuilt_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Ensure pre-built safe tools are registered.

    This registers all builtin tools (volume, brightness, media, targets)
    and catalog actions.  Call once during application initialization.
    Always registers tools regardless of previous state.
    """
    reg = registry or get_registry()

    # Check if tools are already registered by looking at the tool list
    # If already registered, just return
    existing_tools = list(reg._tools.keys())
    expected = ["volume_up", "volume_down", "brightness_up",
                "brightness_down", "play_pause", "rewind", "forward",
                "target_next", "target_prev"]
    if all(t in existing_tools for t in expected):
        # Also check catalog/actions tools
        catalog_actions = ["catalog_action", "custom_action"]
        if all(t in existing_tools for t in catalog_actions):
            return reg

    # Register builtin tools
    from . import builtins
    builtins.register_tools(reg)

    # Register catalog/action tools
    from . import actions
    actions.register_tools(reg)

# Register context tools (always - they are read-only and safe)
    from . import context
    context.register_tools(reg)

    return reg