"""Provider abstraction for the QRUDO AI Assistant orchestration layer.

This module defines the interface that any LLM (or mock) must implement
to participate in the Assistant's tool-call loop.  It is deliberately
provider-neutral: no OpenAI, Ollama, LangChain, or other SDK dependencies.

The only concrete implementation shipped here is ``NullProvider``, which
safely does nothing -- never calls the network, never requires an API key,
and never imports an LLM SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ai.schema import Message, Turn
from ai.tools.registry import ToolResult
from ai.config import AIConfig


class AssistantProvider(ABC):
    """Protocol for AI providers in the QRUDO AI orchestration layer.

    Subclasses must implement ``available()`` and ``respond()``.
    The interface is purposefully minimal and provider-neutral.

    Concrete implementations are responsible for:
      - returning ``available=False`` when the provider cannot operate
        (e.g. no API key, no model loaded)
      - producing a ``Turn`` that may contain ``text`` and/or ``tool_calls``
      - never reaching outside the QRUDO sandbox (no shell, no OS, no
        arbitrary Python execution)
    """

    @abstractmethod
    def available(self) -> bool:
        """Return True if the provider can generate responses.

        Returns False when:
          - no API key is configured
          - no LLM model is loaded
          - the provider is otherwise unable to operate
        """
        ...

    @abstractmethod
    def respond(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        **config: Any,
    ) -> Turn:
        """Generate a ``Turn`` from the given conversation ``messages`` and
        available ``tool`` definitions.

        Args:
          messages: conversation history as ``Message`` objects (role, content).
          tools: list of tool definitions (JSON-schema-like dicts) that the
            AI may call.  These come from ``ToolRegistry`` and are purely
            declarative -- no handler executes anything until the registry
            processes a ``ToolCall``.
          **config: provider-specific configuration (e.g. ``model``, ``temperature``).
            The ``Assistant`` may pass ``AIConfig`` values through this path.

        Returns:
          A ``Turn`` containing either plain ``text``, structured
          ``tool_calls``, or both.  The ``Turn.message.content`` field
          holds the final textual response when no tool calls are produced.
        """
        ...


class NullProvider(AssistantProvider):
    """No-op provider: AI is unavailable.  Never calls the network.

    Satisfies the ``AssistantProvider`` protocol entirely for environments
    where no LLM is desired or configured.  All operations are deterministic
    and side-effect free.
    """

    def available(self) -> bool:
        return False

    def respond(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        **config: Any,
    ) -> Turn:
        """Return a deterministic Turn indicating AI is unavailable.

        This method never performs network I/O, never imports an LLM SDK,
        and never executes a tool.  It simply returns a ``Turn`` with an
        appropriate "AI unavailable" message and no tool calls.
        """
        return Turn(
            message=Message(
                role="assistant",
                content="AI is currently unavailable",
            ),
            tool_calls=[],
        )