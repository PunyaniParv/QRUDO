"""QRUDO AI Tools package.

This package provides the safe AI tool foundation for Phase A.

DO NOT import real LLM providers, HTTP clients, or external SDKs from this
package.  All tools are declarative, whitelisted, and bridge through
the existing ControlEngine safety model.

Typical usage:
    from ai import AIConfig, registry, builtins, actions, context, schema, memory
"""

from __future__ import annotations

from ._aiconfig import AIConfig as AIConfig
from . import schema
from . import memory
from . import tools

__all__ = [
    "AIConfig",
    "schema",
    "memory",
    "tools",
    "registry",
    "builtins",
    "actions",
    "context",
]