"""
Offline text-to-speech for QRUDO (Windows-first).

This is the wake-word -> audible-response side of the voice milestone. It is
deliberately small and dependency-free so it can never send audio or text to
the internet: ``speak()`` drives the synthesizer that ships *inside Windows*
and blocks until the phrase has finished playing.

Backend
-------
``System.Speech`` (a .NET framework assembly that is present on every
desktop Windows 10/11 install) is driven through ``powershell.exe``:

    Add-Type -AssemblyName System.Speech
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.Rate = 1
    $s.Speak($text)

The text is passed base64-encoded so arbitrary phrases (quotes, ``$``,
newlines) can never break the PowerShell command line. The subprocess blocks
until the phrase is done, so ``speak()`` blocks for about as long as the
utterance lasts. Everything runs locally: no API key, no cloud, no network.

Why not a Python TTS package? ``pyttsx3`` is the usual offline choice and
could be layered in later as a faster (in-process) backend behind the same
``speak()`` interface, but it is not installed in QRUDO's venv today and adds
a dependency that must be pip-installed. ``System.Speech`` needs nothing.
On macOS/Linux this module raises :class:`TTSError` with a clear message
until a native backend is added.

Future work: if latency matters, run ``speak()`` on a worker thread (the wake
loop currently calls it synchronously and pauses listening while it talks).
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess

logger = logging.getLogger("qrudo.voice.tts")

# Give PowerShell plenty of time even for a long phrase; SpeechSynthesizer is
# synchronous, so this bounds only pathological hangs.
_PROCESS_TIMEOUT_SECONDS = 60


class TTSError(Exception):
    """Speech synthesis failed or no offline backend is available."""


def speak(text: str) -> None:
    """Say ``text`` through the default audio output. Blocks until finished."""
    text = str(text)
    if not text:
        return
    if os.name != "nt":
        raise TTSError(
            "the System.Speech TTS backend only works on Windows; "
            "a macOS/Linux backend is not implemented yet"
        )
    if shutil.which("powershell") is None:
        raise TTSError("powershell.exe was not found; no offline TTS backend available")

    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$t=[System.Text.Encoding]::UTF8.GetString("
        "[Convert]::FromBase64String('{encoded}'));"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.Rate=1;"
        "$s.Speak($t);"
        "$s.Dispose()"
    ).format(encoded=encoded)
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
            timeout=_PROCESS_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise TTSError(f"System.Speech playback failed: {exc}") from exc
