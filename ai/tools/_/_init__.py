"""QRUDO AI Tools package.

This package provides the safe AI tool foundation for Phase A.

DO NOT import real LLM providers, HTTP clients, or external SDKs from this
package.  All tools are declarative, whitelisted, and bridge through
the existing ControlEngine safety model.
"""

from __future__ import annotations

__all__ = [
    "registry",
    "builtins",
    "actions",
    "context",
]