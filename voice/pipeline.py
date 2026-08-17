"""
Ties wake word detection, audio capture, and STT together into a loop.

This is deliberately dumb: it does not route intents, does not call tools,
does not talk to an LLM. It just reliably produces text from speech and
hands each result to a callback. Wiring that callback into your existing
control/ commands (or later, the tools/ layer and an LLM) happens outside
this file — keep this module's job narrow so it stays fast and testable.
"""

import time
from typing import Callable

from voice.wake_word import WakeWordListener
from voice.audio_capture import record_until_silence
from voice.stt import SpeechToText


TextHandler = Callable[[str], None]


def run_voice_loop(on_text: TextHandler, on_listening: Callable[[], None] | None = None) -> None:
    """
    Blocks forever. On each loop:
      1. wait for "Hey Qrudo"
      2. record until the user stops talking
      3. transcribe
      4. call on_text(transcript) if transcript is non-empty

    on_listening: optional callback fired right after wake word detection,
    e.g. to play a short beep or flash a UI indicator — useful feedback so
    the user knows SARV is actually recording.
    """
    print("[SARV] Loading speech-to-text model...")
    stt = SpeechToText()

    print("[SARV] Starting wake word listener...")
    listener = WakeWordListener()

    print('[SARV] Ready. Say "Hey Qrudo" to speak a command.')
    try:
        while True:
            listener.wait_for_wake_word()

            if on_listening:
                on_listening()
            else:
                print("[SARV] Listening...")

            t0 = time.monotonic()
            audio = record_until_silence()
            transcript = stt.transcribe(audio)
            elapsed_ms = (time.monotonic() - t0) * 1000

            if not transcript:
                print("[SARV] (heard nothing usable)")
                continue

            print(f"[SARV] Heard: \"{transcript}\"  ({elapsed_ms:.0f}ms)")
            on_text(transcript)
    finally:
        listener.close()