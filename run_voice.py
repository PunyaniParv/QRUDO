"""
Standalone entry point for testing SARV's voice pipeline (Milestone 3).

Run this on your Windows machine (not in a sandbox) with a working mic:

    python run_voice.py

Say "Hey Qrudo", then speak a command. Recognized text is routed through the
fast intent router (voice/bridge.py) into the existing ControlEngine, e.g.
"turn the volume up" -> Command.VOLUME_UP -> ControlEngine.submit(...).

Because ``submit`` is non-blocking, the voice loop stays responsive; the OS
action runs on the ControlEngine's own worker thread. Optionally set
``SARV_MICROPHONE_DEVICE`` to select a specific input device (see voice/device.py).
"""

import logging

from control import ControlEngine
from voice.bridge import VoiceIntentRouter
from voice.pipeline import run_voice_loop

# Make voice/device.py's device selection / fallback messages (and the
# engine's own command outcomes) visible on the console.
logging.basicConfig(level=logging.INFO)

# One router, one engine, for the whole session.
_router = VoiceIntentRouter()
_engine: ControlEngine | None = None


def _on_result(result) -> None:
    # ControlEngine already logs every result via the control log module; this
    # line keeps the voice transcript next to its outcome here.
    print(f"[SARV] {result}")


def handle_transcript(text: str) -> None:
    """Route a transcript to a supported command, if any.

    All phrase interpretation lives in VoiceIntentRouter; this function just
    executes the result. Unrecognized speech executes nothing.
    """
    command = _router.classify(text)
    if command is None:
        print(f"[SARV] Not a supported command, nothing executed: {text!r}")
        return
    print(f"[SARV] Routing -> {command}")
    _engine.submit(command)


def main() -> None:
    global _engine
    print("[SARV] Constructing control engine...")
    _engine = ControlEngine(on_result=_on_result)
    try:
        run_voice_loop(on_text=handle_transcript)
    finally:
        _engine.close()


if __name__ == "__main__":
    main()