"""AI configuration for QRUDO Phase A.

All settings default to safe no-op values so that importing this package
requires no API key, no external service, and causes zero behavior change
to the existing QRUDO application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AIConfig:
    """Configuration for the AI tool foundation.

    All defaults are explicitly chosen so that:
    - ai_enabled=False by default (zero behavior change)
    - No API key or external service required
    - All environment variable lookups are optional / graceful
    """

    ai_enabled: bool = False
    """When False, the AI tool foundation is a no-op no-op."""

    provider: str = ""
    """Name of the LLM provider (e.g. "openai", "ollama"). Empty by default."""

    endpoint: str = ""
    """API endpoint URL. Empty by default."""

    model: str = ""
    """Model name. Empty by default."""

    confirm_actions: bool = True
    """Whether tool actions require explicit confirmation before execution."""

    max_turns: int = 5
    """Maximum conversation turns before cutoff."""

    @classmethod
    def load(cls) -> "AIConfig":
        """Load configuration from environment variables.

        Environment variables (all optional, all graceful):
          AI_ENABLED=1|0
          AI_PROVIDER=...
          AI_ENDPOINT=...
          AI_MODEL=...
          AI_CONFIRM_ACTIONS=1|0
          AI_MAX_TURNS=...
        """
        return cls(
            ai_enabled=os.environ.get("AI_ENABLED", "").lower() in ("1", "true", "yes"),
            provider=os.environ.get("AI_PROVIDER", ""),
            endpoint=os.environ.get("AI_ENDPOINT", ""),
            model=os.environ.get("AI_MODEL", ""),
            confirm_actions=os.environ.get("AI_CONFIRM_ACTIONS", "").lower() in ("1", "true", "yes")
                    if os.environ.get("AI_CONFIRM_ACTIONS") is not None
                    else True,
            max_turns=int(os.environ.get("AI_MAX_TURNS", "5")),
        )


#: The default configuration -- ai_enabled=False so importing this package
#: is completely silent w.r.t. QRUDO behavior.
DEFAULT = AIConfig.load()