"""Provider-neutral dataclasses for the future AI layer.

These types carry tool calls and results between the AI and QRUDO's
control layer without depending on any external SDK (OpenAI, Ollama,
LangChain, etc.).  They are deliberately kept simple so that any
provider can convert to/from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A message in the AI conversation.

    Roles:
      - "system": instructions / prompt
      - "user": user input / transcript
      - "assistant": model output / tool calls
    """

    role: str
    content: str
    tool_calls: List["ToolCall"] = field(default_factory=list)
    name: str = ""


@dataclass
class ToolCall:
    """A request for the system to run a tool.

    After the AI produces a ToolCall, QRUDO's tool registry resolves it
    and executes the handler through the ControlEngine.
    """

    id: str
    name: str
    arguments: Dict[str, Any]
    # Resolved after execution; filled in by the registry
    result: Optional["ToolResult"] = None


@dataclass
class ToolResult:
    """The result of a tool execution, fed back to the AI."""

    tool_call_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Turn:
    """One complete exchange: user message → AI response → tool calls → results."""

    message: Message
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)