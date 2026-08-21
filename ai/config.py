"""AI configuration for the QRUDO assistant orchestration layer.

Re-exports :class:`AIConfig` from ``ai._aiconfig`` so that callers can
import it as ``from ai.config import AIConfig``.
"""

from ai._aiconfig import AIConfig  # type: ignore[no-redef]