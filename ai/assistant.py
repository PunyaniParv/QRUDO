"""Assistant orchestrator for the QRUDO AI tool foundation.

The ``Assistant`` is responsible for:

- receiving an unmatched user request
- building context / messages
- obtaining the tool manifest from ToolRegistry
- calling AssistantProvider
- processing structured tool calls
- dispatching tools ONLY through ToolRegistry
- feeding ToolResult back to the provider
- continuing for at most AIConfig.ai_max_turns
- returning a final textual response

The Assistant must NEVER:
- touch the OS directly
- import Windows/macOS backend code
- execute shell commands
- bypass ControlEngine
- bypass ToolRegistry
- reinterpret deterministic catalog phrases

The architecture is:

  Assistant
      ↓
  Provider
      ↓
  ToolRegistry
      ↓
  ControlEngine
      ↓
  existing safety/action infrastructure
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from ai.config import AIConfig
from ai.memory import Memory, NullMemory
from ai.provider import AssistantProvider, NullProvider
from ai.schema import Message, ToolCall, Turn
from ai.tools.registry import ToolRegistry, get_registry, reset_registry, ensure_prebuilt_tools


class Assistant:
    """Orchestrates one AI turn: provider → tools → results → final response.

    The Assistant is entered only when the voice pipeline's deterministic
    routing has failed to match a known command.  It should NOT be invoked
    for built-in phrases ("increase volume", "open Chrome", etc.) since
    those are handled entirely by the VoiceIntentRouter → ControlEngine
    path.

    Dependency direction: Assistant → Provider → ToolRegistry → ControlEngine.
    Nothing flows from ControlEngine back into the Assistant except
    ToolResult objects carrying execution outcomes.
    """

    #: Maximum number of back-and-forth turns before the Assistant gives up.
    #: Controlled by AIConfig.ai_max_turns (default 5).
    max_turns: int

    #: Optional memory interface.  NullMemory is the default (completely
    #: read-only, no persistence).
    memory: Memory

    #: Provider used to generate turns.  NullProvider is the default.
    provider: AssistantProvider

    #: ToolRegistry is the single bridge between AI tool calls and
    #: QRUDO's control layer.  The Assistant never executes tools directly.
    registry: ToolRegistry

    def __init__(
        self,
        *,
        config: AIConfig | None = None,
        provider: AssistantProvider | None = None,
        memory: Memory | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config or AIConfig.load()
        self.provider = provider or NullProvider()
        self.memory = memory or NullMemory()
        self.registry = registry or get_registry()

        # Ensure pre-built tools are registered in the registry
        ensure_prebuilt_tools(self.registry)

        # Enforce AI disabled = no provider calls
        if not self.config.ai_enabled:
            self.provider = NullProvider()
            # Replace registry with empty one that safely returns nothing
            self.registry = type('EmptyRegistry', (), {
                '_tools': {},
                'call': lambda self, engine, tool_call, **kw: ToolResult(
                    tool_call_id=tool_call.id, success=False, data={},
                    error="AI disabled; no tools available"
                ),
            })()

    # ------------------------------------------------------------------
    # Public orchestration entry point
    # ------------------------------------------------------------------

    def escalate(self, command_text: str, context: Dict[str, Any] | None = None) -> str:
        """Process one AI escalation cycle.

        This is called from the voice pipeline when the deterministic router
        returns None for a given command text.

        Returns the final textual response string.
        """
        if not self.config.ai_enabled:
            return "AI is currently unavailable.  No supported command matched."

        # Build conversation context
        messages: List[Message] = self._build_messages(command_text, context or {})

        # Obtain the tool manifest from the registry
        tools = self._tool_manifest()

        # Track turn count
        turn_count = 0

        # Main tool-call loop
        while turn_count < self.config.max_turns:
            turn_count += 1

            # Ask the provider for a response
            turn: Turn = self.provider.respond(messages, tools, config=self.config.__dict__)

            # If the provider produced tool calls, dispatch them
            if turn.tool_calls:
                for tool_call in turn.tool_calls:
                    # Validate and execute through the registry only
                    result = self.registry.call(
                        engine=None,
                        tool_call=tool_call,
                        config=self.config,
                    )

                    # Feed the result back to the provider
                    messages.append(turn.message)  # type: ignore[arg-type]
                    messages.append(
                        Message(role="tool", content="", tool_calls=[tool_call])  # type: ignore[arg-type]
                    )
                    # The ToolResult gets fed back; we reconstruct a minimal
                    # message structure the provider can read.  The actual
                    # result data is appended as a user-visible note.
                    # The provider's next respond() call will see the result.
                    # We store the result data in the assistant's internal
                    # state; here we just ensure the loop continues.
                    # Re-append the provider's previous message if needed.
                    # For simplicity, we just push a placeholder and let
                    # the provider handle it.

                    # Actually, let's do this properly: record the result and
                    # continue the loop
                    # The provider's next respond() will see the tool result
                    # in the conversation history.

                    # Push a tool-role message carrying the result data
                    if result.success:
                        messages.append(
                            Message(
                                role="user",
                                content=f"Tool '{tool_call.name}' executed: {result.data}",
                            )
                        )
                    else:
                        messages.append(
                            Message(
                                role="user",
                                content=f"Tool '{tool_call.name}' failed: {result.error or 'unknown error'}",
                            )
                        )

                # After processing all tool calls in this turn, loop back
                # to let the provider see the results and produce the next turn.
                continue

            # No tool calls: the provider produced a final text response
            if turn.message and turn.message.content:
                return turn.message.content

            # Empty response - end the loop
            return "AI did not produce a response."

        # Max turns exceeded
        return f"AI response exceeded maximum of {self.config.max_turns} turns."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, command_text: str, context: Dict[str, Any]) -> List[Message]:
        """Build the message list to pass to the provider.

        Includes:
          - system instruction (AI is an assistant for QRUDO control)
          - available tool manifest
          - the user's command text
          - relevant context (current target, capabilities, etc.)
          - recent memory if available
        """
        msgs: List[Message] = []

        # System instruction
        msgs.append(
            Message(
                role="system",
                content=(
                    "You are QRUDO's AI assistant.  Help the user control their "
                    "computer using the available tools.  You may call tools to "
                    "perform actions such as adjusting volume, changing brightness, "
                    "playing media, or executing catalog jobs.  Only use the "
                    "tool names that appear in the 'tools' declaration below.  "
                    "Do NOT invent new tools or execute arbitrary code.  Every "
                    "tool call goes through the ToolRegistry, which enforces "
                    "whitelisting, schema validation, and confirmation requirements. "
                    "The deterministic voice path (STT -> VoiceIntentRouter -> "
                    "ControlEngine) handles built-in commands directly and must "
                    "not be duplicated here.  Only genuinely unmatched requests "
                    "reach this assistant."
                ),
            )
        )

        # Capabilities / tool manifest
        # We inject the currently registered tool names and their schemas
        # so the provider knows what it may call.
        try:
            reg = self.registry
            # Gather tool info from the registry
            tool_summaries: List[str] = []
            for name in reg._tools.keys():
                tool_summaries.append(name)
            tool_desc = "Available tools: " + ", ".join(tool_summaries) if tool_summaries else "No tools available."
        except Exception:
            tool_desc = "Available tools: (error gathering manifest)"

        msgs.append(Message(role="system", content=tool_desc))

        # User request
        msgs.append(Message(role="user", content=command_text))

        # Context (current target, capabilities, etc.)
        if context:
            ctx_parts: List[str] = []
            if "current_target" in context:
                ctx_parts.append(f"Current target: {context['current_target']}")
            if "available_commands" in context:
                ctx_parts.append(f"Available commands: {context['available_commands']}")
            if ctx_parts:
                msgs.append(
                    Message(role="system", content="Context: " + " | ".join(ctx_parts))
                )

        # Memory snapshot (read-only; NullMemory provides nothing)
        try:
            recent = self.memory.get_recent(3)
            if recent:
                mem_parts: List[str] = []
                for msg in recent:
                    role = getattr(msg, "role", "unknown")
                    content = str(getattr(msg, "content", ""))[:60]
                    mem_parts.append(f"{role}: {content}")
                if mem_parts:
                    msgs.append(
                        Message(
                            role="system",
                            content="Recent memory: " + " | ".join(mem_parts),
                        )
                    )
        except Exception:
            pass

        return msgs

    def _tool_manifest(self) -> List[Dict[str, Any]]:
        """Build a list of tool definitions (JSON-schema-like) from the registry.

        This is the manifest the AI sees when deciding what it may call.
        It is derived entirely from the ToolRegistry's whitelisted tools.
        """
        try:
            reg = self.registry
            manifests: List[Dict[str, Any]] = []
            for name, tool in reg._tools.items():
                # Build a minimal schema from the tool's parameters
                params = tool.parameters
                schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

                if "properties" in params:
                    schema["properties"] = params["properties"]
                if "required" in params:
                    schema["required"] = params["required"]

                manifests.append(
                    {
                        "name": name,
                        "description": tool.description,
                        "parameters": schema,
                        "confirmation_required": tool.confirmation_required,
                    }
                )
            return manifests
        except Exception:
            return []


# ------------------------------------------------------------------
# Convenience: quick-start assistant (for testing / CLI integration)
# ------------------------------------------------------------------

def create_assistant(
    *,
    config: AIConfig | None = None,
    provider: AssistantProvider | None = None,
    memory: Memory | None = None,
) -> Assistant:
    """Factory for an Assistant with sensible defaults.

    Use this when wiring the assistant into the voice pipeline or other
    integration points.  All dependencies are injectable for testing.
    """
    return Assistant(
        config=config,
        provider=provider or NullProvider(),
        memory=memory or NullMemory(),
        registry=get_registry(),
    )