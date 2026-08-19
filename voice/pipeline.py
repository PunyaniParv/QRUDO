"""
Ties wake word detection, audio capture, and STT together into a loop.

Two entry points share the same building blocks (wake-word engine,
:func:`~voice.audio_capture.capture_command`, :class:`~voice.stt.SpeechToText`,
:class:`~voice.bridge.VoiceIntentRouter`, :class:`~control.ControlEngine`,
:class:`~voice.stream.MicMonitor`):

* :func:`run_voice_loop` -- the original raw-text loop: wake word -> record ->
  transcribe -> ``on_text(transcript)``.  Kept for compatibility and for
  programmatic callers that want the transcript and nothing else.
* :func:`run_voice_command_loop` -- the deterministic *voice command* loop,
  and the one ``main.py --voice`` runs.  It is an explicit, testable state
  machine over one shared microphone owner::

      WAKE_LISTENING -> WAKE_DETECTED -> COMMAND_CAPTURE -> TRANSCRIBING
                     -> ROUTING -> EXECUTING -> WAKE_LISTENING

  Failure paths (WAKE_FALSE_POSITIVE, EMPTY_COMMAND, UNSUPPORTED_COMMAND,
  STT_FAILURE, AUDIO_FAILURE) all return to WAKE_LISTENING; SHUTDOWN closes
  the mic, wake engine and engine.

Audio architecture (see ``voice/stream.py`` and ``voice/audio_capture.py``):

* A single :class:`~voice.stream.MicMonitor` owns the microphone for the whole
  session.  Wake detection and command capture read the *same* live frames, so
  no gap exists between the wake phrase and the command that follows it in the
  same breath -- "hey jarvis increase the volume" works without repeating
  anything.  A rolling pre-roll keeps the audio just before the wake point.
* No conversational TTS in the command path: the default ``wake_response`` is
  empty, so nothing blocks between wake detection and command capture, and the
  assistant never records its own reply.  A caller that wants a short spoken
  ack passes ``wake_response`` + ``tts_speak`` and accepts the (bounded)
  two-stage flow.

The transcript is routed through :class:`~voice.bridge.VoiceIntentRouter` into
the existing ControlEngine command vocabulary (built-in commands and, through
the catalog, CUSTOM action chains).  Everything is local and offline.

Diagnostics (``[wake-debug]``, ``[wake-stats]``, ``[record-stats]``,
``[whisper-stats]``, ``[voice-timing]``) print only when ``debug=True``; the
normal mode prints the one-line lifecycle.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from control import ControlEngine
from voice.audio_capture import capture_command, record_until_silence
from voice.bridge import VoiceIntentRouter, normalize
from voice.config import CONFIG
from voice.stt import SpeechToText
from voice.stream import MicMonitor
from voice.wake_word import WakeWordError, create_wake_word_engine


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
    the user knows the assistant is actually recording.
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


def _wake_phrases(label: str) -> list[str]:
    """Candidate wake phrases to strip from a transcript, longest first.

    ``"hey_jarvis_v0.1"`` -> ``["hey jarvis v0 1", "hey jarvis", "jarvis"]``
    (the version suffix is dropped before deriving the spoken phrase).
    """
    base = re.sub(r"_v?\d+(\.\d+)*$", "", label or "").replace("_", " ").strip()
    candidates = [base] if base else []
    for extra in ("jarvis", "hey jarvis"):
        if extra not in candidates:
            candidates.append(extra)
    return candidates


def strip_wake_phrase(text: str, wake_label: str) -> str:
    """Remove a leading wake phrase from a transcript; return the rest.

    ``"hey jarvis increase the volume"`` -> ``"increase the volume"``,
    ``"hey jarvis"`` -> ``""``.  Commas/punctuation are already normalised
    away by the router's ``normalize``.  A transcript that does not start
    with a wake phrase is returned normalized unchanged, so a two-stage
    interaction (wake, pause, then the command alone) routes as before.
    """
    norm = normalize(text)
    for phrase in _wake_phrases(wake_label):
        phrase = normalize(phrase)
        if not phrase:
            continue
        if norm == phrase:
            return ""
        if norm.startswith(phrase + " "):
            return norm[len(phrase) + 1:].strip()
    return norm


def _speak_ack(tts_speak: Callable[[str], None], text: str) -> None:
    """Speak an opt-in acknowledgement, then drain the mic echo.

    Only called when a caller explicitly passes both ``wake_response`` and
    ``tts_speak`` (a two-stage interaction).  The default command path has no
    acknowledgement at all, so it never blocks and never records a reply.
    """
    try:
        tts_speak(text)
    except Exception:
        pass


def run_voice_command_loop(
    *,
    router: VoiceIntentRouter | None = None,
    control_engine: ControlEngine | None = None,
    stt: SpeechToText | None = None,
    wake_engine=None,
    tts_speak: Callable[[str], None] | None = None,
    wake_response: str | None = None,
    on_listening: Callable[[], None] | None = None,
    on_transcript: Callable[[str], None] | None = None,
    source: str = "",
    should_stop: Callable[[], bool] | None = None,
    max_cycles: int | None = None,
    debug: bool | None = None,
    monitor: MicMonitor | None = None,
) -> None:
    """
    Blocks forever (or for ``max_cycles`` wake->command cycles) running the
    deterministic voice command loop::

        mic -> wake word -> (optional short ack) -> same-utterance command
             -> faster-whisper STT -> VoiceIntentRouter -> ControlEngine

    Every dependency is injectable so unit tests run with zero hardware (pass
    fakes for ``wake_engine``, ``stt``, ``router``, ``control_engine``,
    ``tts_speak`` and ``monitor``); anything omitted is built from the existing
    implementations.  An engine/router/STT/monitor built here is owned and
    closed here; a caller-supplied ``control_engine`` is left for the caller
    to close.

    ``wake_response`` defaults to ``""``: no spoken acknowledgement, so the
    command starts being captured immediately after the wake phrase and the
    assistant never records its own reply.  Pass a short phrase and a
    ``tts_speak`` to opt into a spoken two-stage ack (bounded: the ack is
    spoken, then the microphone's echo is drained before capture).

    ``should_stop``, when given, is asked at every state boundary; answering
    true ends the loop cleanly (the current blocking audio read is bounded by
    a timeout, so shutdown is prompt).  ``debug=True`` (defaults to
    ``CONFIG.debug``) prints the per-frame/wake/record/whisper/timing
    diagnostics; the normal mode prints the one-line lifecycle.

    ``monitor``, when given, is used (and not closed) as the single mic owner
    -- tests inject a fake; when omitted a real :class:`MicMonitor` is opened
    here and closed here.
    """
    router = router if router is not None else VoiceIntentRouter()
    if stt is None:
        print("[voice] Loading speech-to-text model...")
        stt = SpeechToText()
    wake = wake_engine if wake_engine is not None else create_wake_word_engine()
    debug = CONFIG.debug if debug is None else debug
    response = ("" if wake_response is None else wake_response)

    owns_engine = control_engine is None
    if owns_engine:
        control_engine = ControlEngine()

    owns_monitor = monitor is None
    if owns_monitor:
        monitor = MicMonitor()

    try:
        wake.initialize()
    except WakeWordError as exc:
        print(f"[wake-word] Wake-word unavailable: {exc}")
        print("[voice] Voice control will not start. No commands executed.")
        if owns_monitor:
            monitor.close()
        if owns_engine:
            control_engine.close()
        wake.close()
        return

    label = _wake_label(wake)

    try:
        monitor.open()
    except Exception as exc:
        print(f"[voice] ERROR: no microphone available: {exc}")
        print("[voice] Voice control will not start.")
        if owns_engine:
            control_engine.close()
        wake.close()
        return

    print("[voice] Ready. Say the wake phrase.")

    cycles = 0
    try:
        while max_cycles is None or cycles < max_cycles:
            if should_stop is not None and should_stop():
                break
            cycles += 1

            # --- WAKE_LISTENING -> WAKE_DETECTED -------------------------
            try:
                heard = wake.wait_for_wake_word(
                    frame_source=monitor.next_frame,
                    stop=should_stop,
                    debug=debug,
                )
            except WakeWordError as exc:
                # AUDIO_FAILURE: the mic died mid-session.
                print(f"[voice] ERROR: wake-word listening failed: {exc}")
                break
            if not heard:
                break  # shutdown requested
            print(f"[wake-word] WAKE WORD DETECTED: {label}")

            # Clear the wake model's buffers so the phrase tail and the
            # command audio cannot re-trigger it.
            _reset = getattr(wake, "reset", None)
            if callable(_reset):
                _reset()

            if on_listening:
                on_listening()

            if tts_speak is not None and response:
                # Opt-in two-stage ack: speak it (bounded, then drain the
                # speaker echo from the mic before the command is captured).
                _speak_ack(tts_speak, response)
                monitor.drain()

            # --- COMMAND_CAPTURE -----------------------------------------
            t_capture = time.monotonic()
            try:
                captured = capture_command(monitor, stop=should_stop, stats=debug)
            except Exception as exc:
                # AUDIO_FAILURE: recover and keep listening.
                print(f"[voice] ERROR: command capture failed: {exc}")
                continue
            t_stt = time.monotonic()
            if captured is None:
                # EMPTY_COMMAND: nothing said after the wake phrase.
                print("[voice] No speech after the wake phrase; "
                      "back to wake-word listening.")
                continue
            audio, record_stats = captured if debug else (captured, None)

            # --- TRANSCRIBING --------------------------------------------
            t_stt_done = time.monotonic()
            try:
                transcript = stt.transcribe(audio)
            except Exception as exc:
                # STT_FAILURE: recover and keep listening.
                print(f"[voice] ERROR: speech-to-text failed: {exc}")
                continue
            if debug:
                print("[whisper-stats]")
                print(f"model={getattr(stt, 'model_name', CONFIG.whisper_model_size)}")
                print(f"duration={t_stt_done - t_stt:.2f}s")
                print(f'text="{transcript}"')
                empty = "yes" if not transcript else "no"
                print(f"empty={empty}")

            # Wake+command in one utterance: strip the wake phrase so the
            # remainder routes as a plain command.
            command_text = strip_wake_phrase(transcript, label)
            if not command_text:
                # EMPTY_COMMAND (wake word only, or a hallucination).
                print("[voice] No speech detected after the wake word. "
                      "Back to wake-word listening.")
                continue
            print(f'[voice] Transcription: "{command_text}"')
            if on_transcript:
                on_transcript(command_text)

            # --- ROUTING ---------------------------------------------------
            t_route = time.monotonic()
            route = router.route(command_text)
            if route is None:
                # UNSUPPORTED_COMMAND: malformed or unsupported speech --
                # nothing is executed, ever.
                print("[voice] No supported command matched. Nothing executed.")
                continue
            command = route.command
            payload = route.payload
            print(f"[voice] Intent: {command.name}")
            if payload:
                print(f"[voice] Executing: {command.name} (payload={payload})")
            else:
                print(f"[voice] Executing: {command.name}")

            # --- EXECUTING ------------------------------------------------
            t_exec = time.monotonic()
            control_engine.submit(command, source=source, payload=payload)

            if debug:
                if record_stats is not None:
                    print("[record-stats]")
                    print(f"first_speech_after={record_stats['first_speech_after']:.2f}s")
                    print(f"duration={record_stats['duration']:.2f}s")
                    print(f"final_silence={record_stats['final_silence']:.2f}s")
                    print(f"samples={record_stats['samples']}")
                print("[voice-timing]")
                print(f"capture_duration={t_stt - t_capture:.2f}s")
                print(f"stt_duration={t_stt_done - t_stt:.2f}s")
                print(f"routing={t_route - t_stt_done:.2f}s")
                print(f"execution={t_exec - t_route:.2f}s")
                print(f"total_command={t_exec - t_capture:.2f}s")
    except KeyboardInterrupt:
        print("\n[voice] Stopped by user.")
    finally:
        if owns_monitor:
            monitor.close()
        wake.close()
        if owns_engine:
            control_engine.close()
