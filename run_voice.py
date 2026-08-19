"""
Integrated voice command pipeline entry point for QRUDO.

Run this on your Windows machine (not in a sandbox) with a working mic:

    python run_voice.py

Flow: say "Hey Jarvis" -> QRUDO answers "Hey, how are you?" -> say a command
(e.g. "turn the volume up") -> faster-whisper transcribes it locally ->
VoiceIntentRouter -> the existing ControlEngine executes it. Unrecognized
speech executes nothing; silence after the wake word times out cleanly and
returns to wake-word listening.

Mic selection is unchanged: ``SARV_MICROPHONE_DEVICE`` accepts an explicit
device index, a device-name substring, or is unset for the OS default
(Bluetooth headsets included). TTS and STT are fully local/offline.

Because ``ControlEngine.submit`` is non-blocking, the voice loop stays
responsive; the OS action runs on the ControlEngine's own worker thread.
"""

import logging

from control import ControlEngine
from voice.bridge import VoiceIntentRouter
from voice.pipeline import run_voice_command_loop

# Make voice/device.py's device selection / fallback messages (and the
# engine's own command outcomes) visible on the console.
logging.basicConfig(level=logging.INFO)

# One router, one engine, for the whole session.
_router = VoiceIntentRouter()
_engine: ControlEngine | None = None


def _on_result(result) -> None:
    # ControlEngine already logs every result via the control log module; this
    # line keeps the voice transcript next to its outcome here.
    print(f"[control] {result}")


def main() -> None:
    global _engine
    print("[voice] Constructing control engine...")
    _engine = ControlEngine(on_result=_on_result)
    try:
        run_voice_command_loop(router=_router, control_engine=_engine)
    finally:
        _engine.close()


if __name__ == "__main__":
    main()