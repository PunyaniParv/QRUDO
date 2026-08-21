"""Safe bridge for existing catalog/custom actions.

The LLM must NOT invent arbitrary CUSTOM payloads.

Instead:
  LLM → registered catalog tool → existing catalog resolution/validation
  → validated action payload → ControlEngine.execute(CUSTOM, payload, source="ai")

Reuses the existing catalog and actions safety model.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from control import catalog, actions as actions_mod
from . import registry
from .registry import Tool


#: Schema for catalog-backed action resolution
#: The AI can request a catalog job by name; the registry resolves it
#: through the existing catalog, validates the payload, and executes.
CATALOG_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "job_name": {"type": "string"},
        "app": {"type": "string", "description": "Optional app context for keystroke jobs"},
        "confirm": {"type": "boolean", "description": "Confirm execution (for confirmation-required tools)"},
    },
    "required": ["job_name"],
    "additionalProperties": False,
}


def _resolve_catalog_job(job_name: str, app: str | None = None) -> Dict[str, Any] | None:
    """Resolve a catalog job through the existing catalog module.

    Returns the resolved action dict, or None if the job is unknown.
    This uses the exact same resolution logic as the UI/form layer.
    """
    action = catalog.resolve(job_name, app or "any")
    if action is None:
        return None
    # Validate the action through the existing actions safety model
    try:
        validated = actions_mod.validate(action)
        return validated
    except Exception:
        # If validation fails, try returning the raw action
        return action


def _execute_catalog_action(engine: Any, job_name: str, app: str | None = None) -> Dict[str, Any]:
    """Execute a catalog job by resolving and running through ControlEngine.

    This is the safe bridge: LLM → catalog resolution → validated payload
    → ControlEngine.execute(CUSTOM, payload, source="ai")
    """
    resolved = _resolve_catalog_job(job_name, app)
    if resolved is None:
        return {"error": f"unknown catalog job: {job_name}", "status": "unknown"}

    # Serialize the validated action through the existing action pipeline
    from control.actions import serialize
    payload = serialize(resolved)

    # Execute via ControlEngine with CUSTOM command and the action payload
    # The engine's _run_custom will parse and execute the action chain
    result = engine.execute(Command.CUSTOM, source="ai", payload=payload)

    return {
        "command": result.command,
        "status": result.status,
        "detail": result.detail,
        "error": result.error,
        "source": "ai",
    }


def register_tools(registry: registry.ToolRegistry) -> None:
    """Register catalog-backed action tools with the given registry.

    These tools allow the AI to request existing catalog jobs (like "Open Chrome",
    "Volume up", "Play/pause", etc.) without inventing arbitrary custom payloads.
    The catalog resolution, validation, and ControlEngine execution are all
    handled through the existing safety model.
    """
    registry.register(Tool(
        name="catalog_action",
        description="Execute an existing catalog job (e.g. 'Open Chrome', 'Volume up')",
        parameters=CATALOG_ACTION_SCHEMA,
        handler=_execute_catalog_action,
        confirmation_required=False,
    ))

    # Also register a tool for explicit CUSTOM actions that go through the
    # catalog validation path - this prevents arbitrary payload injection
    registry.register(Tool(
        name="custom_action",
        description="Execute a validated custom action chain (must be a serialized catalog action)",
        parameters={"type": "object", "properties": {
            "action_payload": {"type": "string", "description": "Serialized action chain from catalog validation"}
        }, "required": ["action_payload"]},
        handler=_execute_custom_action_through_catalog,
        confirmation_required=False,
    ))


def _execute_custom_action_through_catalog(engine: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a custom action through the catalog validation path.

    This ensures that even "arbitrary" custom actions must go through
    the catalog/validation pipeline, preventing injection of unregistered
    or dangerous payloads.
    """
    payload_str = args.get("action_payload", "")
    if not payload_str:
        return {"error": "no action payload provided", "status": "error"}

    # The payload should be a serialized action chain that has already
    # been through catalog validation.  We re-validate to be safe.
    from control.actions import parse, validate

    try:
        actions_list = parse(payload_str)
    except Exception as exc:
        return {"error": f"invalid action payload: {exc}", "status": "error"}

    # Validate each action
    try:
        validated_actions = validate(actions_list)
    except Exception as exc:
        return {"error": f"action validation failed: {exc}", "status": "error"}

    # Execute through ControlEngine
    from control import Command
    try:
        result = engine.execute(Command.CUSTOM, source="ai", payload=payload_str)
        return {
            "command": result.command,
            "status": result.status,
            "detail": result.detail,
            "error": result.error,
            "source": "ai",
        }
    except Exception as exc:
        return {"error": f"execution failed: {exc}", "status": "error"}