"""Minimal memory contract and NullMemory implementation.

The contract is purposefully tiny: only get_recent, remember, and clear.
NullMemory safely does nothing, so importing the AI package never
requires persistent state or a database.
"""

from __future__ import annotations


class Memory:
    """Interface for AI memory.

    Implementations must provide get_recent, remember, and clear.
    All operations must be deterministic and non-blocking.
    """

    def get_recent(self, n: int = 1) -> List[dict]:
        """Return the last ``n`` messages remembered.

        Returns an empty list by default (NullMemory).
        """
        return []

    def remember(self, role: str, content: str) -> None:
        """Store a message for the conversation context.

        By default this is a no-op (NullMemory).
        """
        pass

    def clear(self) -> None:
        """Forget all remembered messages.

        By default this is a no-op (NullMemory).
        """
        pass


class NullMemory(Memory):
    """Safe no-op memory: every operation is a no-op.

    Use this as the default when no persistent memory is desired.
    """

    def get_recent(self, n: int = 1) -> List[dict]:
        return []

    def remember(self, role: str, content: str) -> None:
        pass

    def clear(self) -> None:
        pass