"""Centralized voice logging: USER / DEBUG / TRACE separation.

Every message the voice pipeline emits goes through one of these three
functions so the CLI can look like a calm assistant in NORMAL mode while the
full engineering telemetry stays available on demand.  No module outside this
one calls ``print()`` for voice-pipeline output.

* :func:`voice_log`   -- USER: lifecycle states the user should see.  Always
  printed.  "Listening", "Wake word detected", "Command: ...", "Executing: ...",
  "Done", "No command detected", startup/shutdown and error lines.
* :func:`voice_debug` -- DEBUG: useful per-utterance diagnostics (wake stats,
  capture stats, STT stats, handoff, timing).  Printed when the debug flag is
  on.  Never frame-by-frame.
* :func:`voice_trace` -- TRACE: high-frequency telemetry ([mic-frame],
  [mic-pop], [capture-frame], raw scores, per-second wake lines, thresholds,
  noise floor, counters).  Printed only when debug AND trace are both on, so
  DEBUG mode alone never floods the terminal.

Levels are controlled by :class:`~voice.config.VoiceConfig`:

* NORMAL: ``python main.py --voice``  -- USER only.
* DEBUG:  ``QRUDO_VOICE_DEBUG=1``     -- USER + DEBUG.
* TRACE:  ``QRUDO_VOICE_DEBUG=1 QRUDO_VOICE_TRACE=1`` -- USER + DEBUG + TRACE.

An explicit ``enabled`` override lets callers that already resolved the debug
flag (such as the pipeline's injectable ``debug`` argument) decide directly;
TRACE additionally always requires ``CONFIG.trace`` so frame-level output can
never leak into DEBUG-only mode.
"""

from __future__ import annotations

from voice import config as _config


def voice_log(message: str) -> None:
    """USER-facing lifecycle message.  Always printed."""
    print(message)


def voice_debug(message: str, *, enabled: bool | None = None) -> None:
    """DEBUG diagnostic.  Printed when debug is on (or ``enabled`` is True)."""
    if enabled if enabled is not None else _config.CONFIG.debug:
        print(message)


def voice_trace(message: str, *, enabled: bool | None = None) -> None:
    """TRACE telemetry.  Printed when debug AND trace are both on.

    ``enabled`` overrides the debug half but never the trace half, so
    frame-level output is impossible unless ``CONFIG.trace`` is set.
    """
    if (enabled if enabled is not None else _config.CONFIG.debug) and _config.CONFIG.trace:
        print(message)