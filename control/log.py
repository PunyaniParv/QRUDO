"""Logging for the control layer (spec F: log command, timestamp, result, errors).

Two sinks:

* ``logs/sarv.log``     -- human-readable rotating text log.
* ``logs/commands.jsonl`` -- one JSON object per executed command, easy to load
  into a notebook when we want to show hit/miss rates in the demo.

Console output is opt-in so that a curses-style UI (or the vision preview
window) is not scribbled over.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "sarv"
_configured = False
_event_path: Path | None = None


def setup(log_dir: str | Path = "logs", *, console: bool = True,
          level: int = logging.INFO) -> logging.Logger:
    """Configure and return the shared ``sarv`` logger.  Safe to call twice."""
    global _configured, _event_path

    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _event_path = directory / "commands.jsonl"

    logger.setLevel(level)
    logger.propagate = False

    file_handler = RotatingFileHandler(
        directory / "sarv.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        logger.addHandler(stream)

    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Child logger, e.g. ``get_logger("macos")`` -> ``sarv.macos``."""
    setup()
    return logging.getLogger(_LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}")


def log_event(result) -> None:
    """Append a ``CommandResult`` to the JSONL event log.

    Logging must never take the app down, so any failure here is swallowed
    after one warning.
    """
    setup()
    if _event_path is None:
        return
    payload = asdict(result) if is_dataclass(result) else dict(result)
    try:
        with _event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except OSError as exc:  # disk full, permissions, ...
        get_logger().warning("could not write event log: %s", exc)
