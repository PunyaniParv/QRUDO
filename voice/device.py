"""Microphone device selection for SARV's voice pipeline.

Both the wake-word listener and the speech-recording capture open an input
stream, and both must use the *same* microphone. Rather than have two separate
selection systems, the whole voice module routes through :class:`MicrophoneStream`,
which resolves and opens the active input device once.

Behavior:
  * ``CONFIG.microphone_device`` is ``None``  -> use the OS default input device.
  * It is an ``int``                        -> that device index.
  * It is a ``str``                         -> a case-insensitive substring of a
                                               device name (mirrors sounddevice).
  * If the configured device cannot be found or opened -- e.g. a Bluetooth
    headset that just disconnected -- SARV does not crash: it logs the failure
    and falls back to the OS default input device.

This module never hard-codes a particular laptop microphone and adds no
dependencies beyond sounddevice.
"""

from __future__ import annotations

import logging

import sounddevice as sd

from voice.config import CONFIG

logger = logging.getLogger("sarv.voice.device")


def _resolve(configured) -> tuple:
    """Return ``(device_spec, human_name)`` for the input device to use.

    ``device_spec`` is what sounddevice wants (an index, or ``None`` for the
    OS default); ``human_name`` is a readable description for logging.
    """
    if configured is None:
        return _default_input()
    try:
        info = sd.query_devices(configured, kind="input")
        return info["index"], info["name"]
    except Exception as exc:  # no such index/name, or no input device at all
        logger.warning(
            "configured microphone %r unavailable (%s); falling back to OS default",
            configured, exc,
        )
        return _default_input()


def _default_input() -> tuple:
    """The OS default input device as ``(index_or_None, name)``."""
    try:
        info = sd.query_devices(kind="input")
        return info["index"], info["name"]
    except Exception as exc:
        logger.warning("no default input device available: %s", exc)
        return None, "OS default (none available)"


class MicrophoneStream:
    """A sounddevice ``InputStream`` on the active input device.

    ``MicrophoneStream`` resolves ``CONFIG.microphone_device`` (or the OS
    default), reports which device is in use, and transparently falls back to
    the OS default if the configured device cannot be opened. Both the wake
    word and audio capture use it so they always hear from the same mic.

    Usage is the same as a raw ``sd.InputStream`` context manager::

        with MicrophoneStream(samplerate=16000, channels=1,
                              dtype="int16", blocksize=480) as mic:
            frame, _overflowed = mic.read(480)

    Attributes set after ``__enter__``:
      * ``device_spec`` -- the device passed to sounddevice (index or None).
      * ``device_name``  -- the human-readable name of the device in use.
    """

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self.device_spec: int | None = None
        self.device_name: str = ""
        self._stream = None

    def __enter__(self) -> "MicrophoneStream":
        configured = CONFIG.microphone_device
        spec, name = _resolve(configured)
        try:
            self._open(spec)
        except Exception as exc:  # e.g. Bluetooth headset went away mid-close
            logger.warning(
                "input device %r failed to open (%s); falling back to OS default",
                spec, exc,
            )
            spec, name = _default_input()
            self._open(spec)
        self.device_spec, self.device_name = spec, name
        logger.info("input device in use: %r (%s)", spec, name)
        return self

    def _open(self, spec) -> None:
        self._stream = sd.InputStream(device=spec, **self._kwargs)
        self._stream.start()

    def read(self, frames: int):
        if self._stream is None:
            raise RuntimeError("MicrophoneStream not entered")
        return self._stream.read(frames)

    def __exit__(self, *exc_info) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None