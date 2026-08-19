"""
Ties wake word detection, audio capture, and STT together into a loop.

Two entry points share the same building blocks (wake-word engine,
:func:`~voice.audio_capture.record_until_silence`, :class:`~voice.stt.SpeechToText`,
:class:`~voice.bridge.VoiceIntentRouter`, :class:`~control.ControlEngine`):

* :func:`run_voice_loop` -- the original raw-text loop: wake word -> record ->
  transcribe -> ``on_text(transcript)``. Kept for compatibility and for
  programmatic callers that want the transcript and nothing else.
* :func:`run_voice_command_loop` -- the deterministic *voice command* loop:
  wake word -> spoken "Hey, how are you?" -> silence-gated command capture ->
  STT -> VoiceIntentRouter -> existing ControlEngine command. Prints a clear
  diagnostic line at every stage.

Neither talks to an LLM, does conversational AI, or routes anything outside
the existing supported Command vocabulary. Everything here is local and
offline.
"""

import time
from typing import Callable

from control import ControlEngine
from voice import tts
from voice.audio_capture import record_until_silence
from voice.bridge import VoiceIntentRouter
from voice.config import CONFIG
from voice.stt import SpeechToText
from voice.wake_word import WakeWordError, create_wake_word_engine


TextHandler = Callable[[str], None]

#: Short reply spoken after the wake word, before command-listening begins.
WAKE_RESPONSE = "Hey, how are you?"


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
    try:
        listener = create_wake_word_engine()
        listener.initialize()
    except WakeWordError as exc:
        print(f"[SARV] Wake-word unavailable: {exc}")
        print("[SARV] Voice control will not start. No commands executed.")
        return
    except Exception as exc:  # genuine (non-anticipated) startup failure
        print(f"[SARV] Failed to start wake-word listener: {exc}")
        raise

    print(f'[SARV] Ready. Say the wake phrase ({listener.name}).')
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


def _wake_label(wake_engine) -> str:
    """Human-facing name for the engine/model that fired (e.g. ``hey_jarvis``)."""
    names = getattr(wake_engine, "_model_names", None)
    if names:
        return names[0]
    return str(getattr(wake_engine, "name", "wake-word"))


def _capture_and_route(
    router: VoiceIntentRouter,
    control_engine: ControlEngine,
    stt: SpeechToText,
    on_transcript: Callable[[str], None] | None,
) -> dict | None:
    """One command utterance: record -> transcribe -> route -> execute.

    Never raises on expected problems: a mic/STT hiccup prints an error and
    returns (a bad utterance must not kill the voice session); an unrecognized
    transcript prints and executes nothing.

    Pure instrumentation (measurement only -- no behavior change):
      * ``record_until_silence(report=True)`` prints a ``[record-stats]`` block.
      * a ``[whisper-stats]`` block is printed for every transcription.
      * returns ``{t3, t5, t6, t7, t8, t9}`` (monotonic timestamps) when a
        command is executed, else ``None`` -- the caller prints the
        ``[voice-timing]`` summary line.
    """
    print("[voice] Listening for command...")
    try:
        t3 = time.monotonic()
        audio = record_until_silence(report=True)
        t5 = time.monotonic()
        t6 = time.monotonic()
        transcript = stt.transcribe(audio)
        t7 = time.monotonic()
    except Exception as exc:
        print(f"[voice] ERROR: command capture failed: {exc}")
        return None

    model_name = getattr(stt, "model_name", CONFIG.whisper_model_size)
    print("[whisper-stats]")
    print(f"model={model_name}")
    print(f"duration={t7 - t6:.2f}s")
    print(f'text="{transcript}"')
    print(f"empty={'yes' if not transcript else 'no'}")

    if not transcript:
        print("[voice] No speech detected after the wake word. Back to wake-word listening.")
        return None
    print(f'[voice] Transcription: "{transcript}"')
    if on_transcript:
        on_transcript(transcript)

    t8 = time.monotonic()
    command = router.classify(transcript)
    if command is None:
        print("[voice] No supported command matched. Nothing executed.")
        return None
    print(f"[voice] Intent: {command.name}")
    print(f"[voice] Executing: {command.name}")
    t9 = time.monotonic()
    control_engine.submit(command)
    return {"t3": t3, "t5": t5, "t6": t6, "t7": t7, "t8": t8, "t9": t9}


def run_voice_command_loop(
    *,
    router: VoiceIntentRouter | None = None,
    control_engine: ControlEngine | None = None,
    stt: SpeechToText | None = None,
    wake_engine=None,
    tts_speak: Callable[[str], None] = tts.speak,
    wake_response: str = WAKE_RESPONSE,
    on_listening: Callable[[], None] | None = None,
    on_transcript: Callable[[str], None] | None = None,
    max_cycles: int | None = None,
) -> None:
    """
    Blocks forever (or for ``max_cycles`` wake->command cycles) running the
    deterministic voice command loop::

        mic -> wake word -> spoken response -> silence-gated command capture
             -> faster-whisper STT -> VoiceIntentRouter -> ControlEngine

    Every dependency is injectable so unit tests run with zero hardware
    (pass fakes for ``wake_engine``, ``stt``, ``router``, ``control_engine``
    and ``tts_speak``); anything omitted is built from the existing
    implementations. An engine/router/STT built here is owned and closed here;
    a caller-supplied ``control_engine`` is left for the caller to close.

    Measurement-only instrumentation (no behavior change): each executed
    command prints one ``[voice-timing]`` block (monotonic wake-to-tts,
    tts-duration, tts-to-record, record-duration, stt-duration, routing,
    execution, total-command); ``record_until_silence(report=True)`` prints
    ``[record-stats]``; every transcription prints ``[whisper-stats]``; the
    wake listener prints ``[wake-stats]`` per detection.

    Anti-duplicate / settle design (no blind sleeps):
      * ``wait_for_wake_word()`` fires at most once per cycle and the engine's
        ``reset()`` clears openWakeWord's internal prediction buffer right
        after detection, so the next listening session cannot re-trigger on
        the wake phrase's stale high-score frames.
      * The spoken reply is synchronous (``tts_speak`` blocks), so the command
        mic stream opens only after the reply has fully played -- QRUDO can
        never record its own response as the user's command. A short,
        configurable ``CONFIG.post_tts_settle_s`` pause follows so any TTS
        echo on a (Bluetooth) headset mic decays before the RMS gate opens.
      * Command recording ends on its own via silence detection or
        ``CONFIG.max_recording_s``; an empty transcript prints a message and
        the loop returns to wake-word listening.
    """
    router = router if router is not None else VoiceIntentRouter()
    if stt is None:
        print("[voice] Loading speech-to-text model...")
        stt = SpeechToText()
    wake = wake_engine if wake_engine is not None else create_wake_word_engine()

    owns_engine = control_engine is None
    if owns_engine:
        control_engine = ControlEngine()

    try:
        wake.initialize()
    except WakeWordError as exc:
        print(f"[wake-word] Wake-word unavailable: {exc}")
        print("[voice] Voice control will not start. No commands executed.")
        if owns_engine:
            control_engine.close()
        return

    label = _wake_label(wake)
    settle_s = float(getattr(CONFIG, "post_tts_settle_s", 0.5))
    print("[voice] Ready. Say the wake phrase.")

    cycles = 0
    try:
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            wake.wait_for_wake_word()
            t0 = time.monotonic()
            print(f"[wake-word] WAKE WORD DETECTED: {label}")
            if on_listening:
                on_listening()
            # Clear stale detection state before the reply/command phase.
            _reset = getattr(wake, "reset", None)
            if callable(_reset):
                _reset()
            print(f"[qrudo] {wake_response}")
            t1 = time.monotonic()
            tts_speak(wake_response)
            t2 = time.monotonic()
            time.sleep(settle_s)
            timings = _capture_and_route(router, control_engine, stt, on_transcript)
            if timings is not None:
                # One compact measurement line per executed command.
                print("[voice-timing]")
                print(f"wake_to_tts={t1 - t0:.2f}s")
                print(f"tts_duration={t2 - t1:.2f}s")
                print(f"tts_to_record={timings['t3'] - t2:.2f}s")
                print(f"record_duration={timings['t5'] - timings['t3']:.2f}s")
                print(f"stt_duration={timings['t7'] - timings['t6']:.2f}s")
                print(f"routing={timings['t8'] - timings['t7']:.2f}s")
                print(f"execution={timings['t9'] - timings['t8']:.2f}s")
                print(f"total_command={timings['t9'] - t0:.2f}s")
    except KeyboardInterrupt:
        print("\n[voice] Stopped by user.")
    finally:
        wake.close()
        if owns_engine:
            control_engine.close()