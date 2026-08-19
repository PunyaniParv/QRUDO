"""Voice as a second input to the same engine, beside the camera.

Gesture and voice both submit to one ControlEngine, so cooldown,
dry-run, logging and the reliability report apply to both equally --
only the source differs, and the report never judges a deliberate
input.  This module is the glue: it runs the voice command loop on a
background thread, sharing the engine the camera already uses.

The voice stack (faster-whisper, openwakeword, sounddevice) is
optional -- it lives in requirements-voice.txt, not requirements.txt
-- so every import here is lazy and every failure is a printed line,
never a crash.  A machine without the voice requirements, without a
microphone, or without a wake-word model simply runs camera-only,
exactly as it did before voice existed.
"""

from __future__ import annotations

import threading
import time

#: The top-level packages the voice pipeline needs, checked without
#: importing any of them.  numpy is pinned in both requirement files and
#: is a hard dependency of the voice stack anyway; listing it keeps
#: ``available()`` honest on a machine that somehow has only some of them.
VOICE_DEPS = ("faster_whisper", "openwakeword", "sounddevice", "numpy")


def available() -> bool:
    """Whether the voice stack is installed.  Cheap: imports nothing.

    A true answer does not mean a microphone or a wake-word model is
    present -- those are discovered later, and their absence is handled
    the same way (a message, and camera-only continues).
    """

    import importlib.util

    return all(importlib.util.find_spec(name) is not None
               for name in VOICE_DEPS)


def run_voice_only(engine, should_stop=None):
    """Run QRUDO as a voice assistant alone -- no camera is touched.

    This is what ``--voice`` (with no camera mode) runs.  It starts the
    voice loop sharing ``engine``, keeps the process alive until told
    to stop, and on the way out puts the microphone away and closes the
    engine exactly as the gesture loop does.  It never imports vision,
    never opens a camera and never takes the singleton lock, so a
    machine with a microphone but no camera still gets its commands.

    Returns 0 on a clean stop, 1 when the voice stack is not installed.
    """

    if engine.config.dry_run:
        print("  QRUDO voice  --  DRY RUN (commands are logged, not performed)")
    else:
        print(f"  QRUDO voice  --  {engine.controller.name}")
    print("  say the wake phrase, then a command -- ctrl+c to stop.\n")

    if not available():
        print("  voice: requirements-voice.txt not installed\n"
              "         (install them, then use --voice)")
        engine.close()
        return 1

    voice = start(engine)

    if should_stop is None:
        should_stop = lambda: False

    try:
        while not should_stop():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            voice.stop()
        except KeyboardInterrupt:
            pass
        try:
            engine.close()
        except KeyboardInterrupt:
            pass

    print("  bye.")
    return 0


def start(engine, *, source: str = "voice", stop_event=None):
    """Run the voice command loop on a background thread, sharing ``engine``.

    Returns a handle with ``stop()``, or None when the voice stack is
    not installed -- the caller then runs camera-only.  The loop is
    best-effort: if the wake-word model or microphone is missing it
    prints why and exits quietly; it can never raise into the caller.

    ``stop_event`` may be supplied by the caller (so several things can
    share one stop signal); otherwise the handle owns a fresh one.
    """

    if not available():
        print("  voice: requirements-voice.txt not installed -- camera only\n"
              "         (say the wake phrase and nothing happens; install\n"
              "         them and use --voice to turn voice on)")
        return None

    stop_event = stop_event if stop_event is not None else threading.Event()
    thread = threading.Thread(
        target=_run, args=(engine, source, stop_event),
        name="qrudo-voice", daemon=True)
    thread.start()

    return _VoiceHandle(thread, stop_event)


def _run(engine, source, stop_event):
    """The voice thread: wake word -> spoken command -> the shared engine.

    Everything optional is imported here, on the thread, so a machine
    that never asks for voice never imports faster-whisper.  Expected
    failures -- no model, no mic -- are printed by the pipeline and end
    the thread; an unexpected one is caught here so the camera loop is
    never taken down with it.
    """

    try:
        from voice.pipeline import run_voice_command_loop
    except Exception as exc:
        print(f"  voice: cannot start ({exc})")
        return

    try:
        run_voice_command_loop(
            control_engine=engine,
            source=source,
            should_stop=stop_event.is_set,
        )
    except Exception as exc:
        print(f"  voice: stopped unexpectedly ({exc})")


class _VoiceHandle:
    """What the caller holds: a way to ask the voice thread to end."""

    def __init__(self, thread, stop_event):
        self.thread = thread
        self._stop = stop_event

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def stop(self, timeout: float = 3.0):
        """Ask the loop to end and wait for it to put the mic away.

        The loop checks the stop signal at every state boundary and bounds
        its audio reads by a short timeout, so a shutdown is prompt.  A
        second Ctrl+C during the join must not produce a traceback, so a
        KeyboardInterrupt raised here is swallowed and the join proceeds.
        """

        self._stop.set()
        try:
            self.thread.join(timeout)
        except KeyboardInterrupt:
            pass
