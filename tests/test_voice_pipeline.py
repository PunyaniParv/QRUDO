"""Tests for the deterministic voice command pipeline (voice/pipeline.py).

Everything here runs with zero hardware: the wake-word engine, STT, mic
capture, TTS, router and ControlEngine are fakes/mocks. The real
VoiceIntentRouter is used (it is pure) so recognized/unknown routing is
verified against the actual command vocabulary.

The pipeline's single mic owner (``monitor``) is injected as a fake; in
production it is a real ``MicMonitor`` shared by wake detection and command
capture, which is what makes a same-utterance "hey jarvis + command" work.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from control import Command

# The pipeline imports faster_whisper at module load; without the voice
# extras (requirements-voice.txt) this whole module is a skip, not an
# error -- the suite must run green on both machines and both CI
# platforms.
try:
    from voice import pipeline
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"voice extras not installed: {exc}")

from voice.bridge import Route as _Route
from voice.bridge import VoiceIntentRouter
from voice.config import CONFIG
from voice.wake_word import WakeWordError

#: A plausible mono float32 audio buffer for faked mic recordings.
_AUDIO = np.zeros(16000, dtype=np.float32)

#: capture_command() returns ``(audio, stats)`` when stats are requested.
_AUDIO_STATS = (_AUDIO, {
    "first_speech_after": 0.10,
    "duration": 0.30,
    "final_silence": 0.10,
    "samples": 12800,
})


class _FakeMonitor:
    """Fake MicMonitor: never touches hardware, records lifecycle calls."""

    def __init__(self):
        self.opened = False
        self.closed = False
        self.drained = 0
        self._frames = []
        self.pre_roll_frames = []

    def open(self):
        self.opened = True

    def next_frame(self, timeout=0.25, stop=None, consumer=""):
        if self._frames:
            return self._frames.pop(0)
        return None

    def pre_roll(self):
        return list(self.pre_roll_frames)

    def pending_count(self):
        return len(self._frames)

    def drain(self):
        self.drained += 1
        self._frames.clear()
        self.pre_roll_frames.clear()

    def close(self):
        self.closed = True


class _FakeWake:
    """Wake engine that always fires; counts calls, resets and close."""

    name = "openwakeword"
    _model_names = ["hey_jarvis"]

    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.closed = False
        self.frame_source = None

    def initialize(self):
        pass

    def wait_for_wake_word(self, frame_source=None, stop=None, debug=False):
        self.calls += 1
        self.frame_source = frame_source
        return True

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


class _CompatWake(_FakeWake):
    """Raises StopIteration after one detection to end run_voice_loop's loop."""

    def wait_for_wake_word(self, frame_source=None, stop=None, debug=False):
        self.calls += 1
        if self.calls > 1:
            raise StopIteration("test: end of compat loop")


class _FakeSTT:
    def __init__(self, transcript="volume up"):
        self.transcript = transcript
        self.calls = 0
        self.model_name = "base"

    def transcribe(self, audio):
        self.calls += 1
        return self.transcript


class _FakeEngine:
    def __init__(self):
        self.submitted = []
        self.sources = []
        self.payloads = []
        self.closed = False

    def submit(self, command, source="", payload=""):
        self.submitted.append(command)
        self.sources.append(source)
        self.payloads.append(payload)

    def close(self):
        self.closed = True


class _FakeRouter:
    def __init__(self, command=None):
        self.command = command
        self.texts = []

    def route(self, text):
        self.texts.append(text)
        if self.command is None:
            return None
        return _Route(self.command)


_DEFAULT = object()  # sentinel: pick capture_command's return based on debug


def _run(capture=_DEFAULT, monitor=None, **kwargs):
    """Run the command loop once with hardware-free defaults; capture stdout."""
    kwargs.setdefault("tts_speak", mock.Mock())
    kwargs.setdefault("stt", _FakeSTT())
    max_cycles = kwargs.pop("max_cycles", 1)
    if capture is _DEFAULT:
        capture = _AUDIO_STATS if kwargs.get("debug") else _AUDIO
    monitor = monitor or _FakeMonitor()
    buf = io.StringIO()
    with mock.patch.object(
        pipeline, "capture_command", return_value=capture
    ), redirect_stdout(buf):
        pipeline.run_voice_command_loop(
            wake_engine=_FakeWake(),
            control_engine=_FakeEngine(),
            monitor=monitor,
            max_cycles=max_cycles,
            **kwargs,
        )
    return buf.getvalue()


class VoiceCommandPipelineCase(unittest.TestCase):
    def test_recognized_command_full_flow(self):
        wake = _FakeWake()
        engine = _FakeEngine()
        tts = mock.Mock()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                router=VoiceIntentRouter(),
                control_engine=engine,
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=tts,
                max_cycles=1,
            )
        out = buf.getvalue()
        self.assertIn("[voice] Wake word detected.", out)
        self.assertIn('[voice] Command: "volume up"', out)
        self.assertIn("[voice] Intent: VOLUME_UP", out)
        self.assertIn("[voice] Executing: VOLUME_UP", out)
        # Default: no spoken acknowledgement, so TTS is never invoked and
        # nothing blocks between the wake phrase and command capture.
        tts.assert_not_called()
        self.assertEqual(engine.submitted, [Command.VOLUME_UP])
        self.assertTrue(wake.closed)

    def test_wake_and_command_in_one_utterance(self):
        # "hey jarvis increase the volume" in a single breath: the wake phrase
        # is stripped so the remainder routes as a plain command.
        engine = _FakeEngine()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                router=VoiceIntentRouter(),
                control_engine=engine,
                stt=_FakeSTT("hey jarvis increase the volume"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                max_cycles=1,
            )
        out = buf.getvalue()
        self.assertIn('[voice] Command: "increase the volume"', out)
        self.assertIn("[voice] Intent: VOLUME_UP", out)
        self.assertEqual(engine.submitted, [Command.VOLUME_UP])

    def test_wake_phrase_alone_routes_as_empty(self):
        # Just the wake phrase (no command after it) must execute nothing.
        out = _run(stt=_FakeSTT("hey jarvis"))
        self.assertIn("[voice] No command detected.", out)

    def test_transcription_callback_receives_transcript(self):
        on_transcript = mock.Mock()
        _run(on_transcript=on_transcript)
        on_transcript.assert_called_once_with("volume up")

    def test_transcription_callback_gets_stripped_text(self):
        on_transcript = mock.Mock()
        _run(stt=_FakeSTT("jarvis next track"), on_transcript=on_transcript)
        on_transcript.assert_called_once_with("next track")

    def test_on_listening_fires_after_detection_with_no_ack_by_default(self):
        order = []

        def listening():
            order.append("listening")

        _run(on_listening=listening)
        self.assertEqual(order, ["listening"])

    def test_opt_in_ack_speaks_before_draining_and_capture(self):
        # A caller that opts into a spoken ack (wake_response + tts_speak) gets
        # a bounded two-stage flow: speak -> drain echo -> capture command.
        order = []
        monitor = _FakeMonitor()

        def tts_speak(_text):
            order.append("tts")

        def capture(mon, **kwargs):
            order.append("capture")
            return _AUDIO

        with mock.patch.object(pipeline, "capture_command", side_effect=capture), \
                redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=_FakeWake(),
                tts_speak=tts_speak,
                wake_response="Yes?",
                monitor=monitor,
                max_cycles=1,
            )
        self.assertEqual(order, ["tts", "capture"])
        self.assertGreaterEqual(monitor.drained, 1)

    def test_timing_instrumentation_printed_on_successful_command(self):
        out = _run(capture=_AUDIO_STATS, debug=True)
        self.assertIn("[record-stats]", out)
        self.assertIn("[voice-timing]", out)
        for field in (
            "capture_duration=",
            "stt_duration=",
            "routing=",
            "execution=",
            "total_command=",
        ):
            self.assertIn(field, out)
        self.assertIn("first_speech_after=0.10s", out)
        self.assertIn("samples=12800", out)

    def test_no_diagnostics_without_debug_flag(self):
        out = _run()
        self.assertNotIn("[wake-stats]", out)
        self.assertNotIn("[record-stats]", out)
        self.assertNotIn("[whisper-stats]", out)
        self.assertNotIn("[voice-timing]", out)

    def test_no_timing_line_without_executed_command(self):
        # Unknown and empty transcripts must not print the [voice-timing] block.
        self.assertNotIn("[voice-timing]", _run(stt=_FakeSTT("open Televator"), debug=True))
        self.assertNotIn("[voice-timing]", _run(stt=_FakeSTT(""), debug=True))

    def test_whisper_stats_printed_for_each_transcription(self):
        out = _run(debug=True)
        self.assertIn("[whisper-stats]", out)
        self.assertIn("model=base", out)
        self.assertIn("empty=no", out)
        out_empty = _run(stt=_FakeSTT(""), debug=True)
        self.assertIn("empty=yes", out_empty)

    def test_stt_input_debug_prints_the_exact_audio_buffer(self):
        # Ground truth for "did captured PCM actually reach Whisper?": the
        # buffer handed to transcribe(), independent of the gate/decode.
        out = _run(capture=_AUDIO_STATS, debug=True)
        self.assertIn("[stt-input-debug]", out)
        self.assertIn("samples=16000", out)
        self.assertIn("duration=1.000s", out)
        self.assertIn("dtype=float32", out)
        self.assertIn("shape=", out)
        self.assertIn("min=", out)
        self.assertIn("max=", out)
        self.assertIn("rms=", out)
        self.assertIn("finite=True", out)

    def test_whisper_stats_reports_audio_duration_not_decode_time(self):
        # Regression: the old [whisper-stats] "duration" was the STT elapsed
        # time, so a fast gate rejection printed 0.00s and looked like the
        # audio itself was empty.  Now it reports the actual PCM duration.
        out = _run(capture=_AUDIO_STATS, debug=True)
        self.assertIn("audio_duration=1.000s", out)
        self.assertIn("decode=", out)

    def test_unknown_command_executes_nothing(self):
        engine = _FakeEngine()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                router=VoiceIntentRouter(),
                control_engine=engine,
                stt=_FakeSTT("open Televator"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=1,
            )
        out = buf.getvalue()
        self.assertIn('[voice] Command: "open televator"', out)
        self.assertIn("[voice] No supported command matched. Nothing executed.", out)
        self.assertEqual(engine.submitted, [])
        self.assertNotIn("Executing", out)

    def test_injected_router_is_used(self):
        router = _FakeRouter(command=Command.FORWARD)
        engine = _FakeEngine()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                router=router,
                control_engine=engine,
                stt=_FakeSTT("skip forward"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=1,
            )
        self.assertEqual(router.texts, ["skip forward"])
        self.assertEqual(engine.submitted, [Command.FORWARD])

    def test_catalog_phrase_submits_custom_with_payload_and_voice_source(self):
        """"next track" -> catalog job -> CUSTOM payload -> engine, source voice."""

        from control.actions import parse

        engine = _FakeEngine()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                router=VoiceIntentRouter(),
                control_engine=engine,
                stt=_FakeSTT("next track"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                source="voice",
                monitor=_FakeMonitor(),
                max_cycles=1,
            )
        out = buf.getvalue()
        self.assertIn("[voice] Intent: CUSTOM", out)
        self.assertEqual(engine.submitted, [Command.CUSTOM])
        self.assertEqual(engine.sources, ["voice"])
        self.assertEqual(len(engine.payloads), 1)
        self.assertEqual(parse(engine.payloads[0]),
                         [{"type": "keystroke", "combo": "shift+n"}])

    def test_no_speech_times_out_and_returns_to_wake(self):
        wake = _FakeWake()
        engine = _FakeEngine()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                stt=_FakeSTT(""),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=2,
            )
        out = buf.getvalue()
        self.assertIn("[voice] No command detected.", out)
        self.assertEqual(out.count("Wake word detected."), 2)
        self.assertEqual(wake.calls, 2)
        self.assertEqual(engine.submitted, [])

    def test_capture_returns_none_and_recovers(self):
        # capture_command returned None (no speech within pre-speech timeout);
        # the loop keeps listening.
        out = _run(capture=None, max_cycles=2)
        self.assertIn("[voice] No command detected.", out)
        self.assertEqual(out.count("Wake word detected."), 2)

    def test_capture_error_does_not_kill_loop(self):
        wake = _FakeWake()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline,
            "capture_command",
            side_effect=RuntimeError("mic went away"),
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=2,
            )
        out = buf.getvalue()
        self.assertIn("[voice] ERROR: command capture failed: mic went away", out)
        self.assertEqual(out.count("Wake word detected."), 2)
        self.assertEqual(wake.calls, 2)

    def test_stt_failure_recovers_and_keeps_listening(self):
        class _BrokenSTT(_FakeSTT):
            def transcribe(self, audio):
                raise RuntimeError("model blew up")

        wake = _FakeWake()
        engine = _FakeEngine()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                stt=_BrokenSTT("volume up"),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=2,
            )
        out = buf.getvalue()
        self.assertIn("[voice] ERROR: speech-to-text failed: model blew up", out)
        self.assertEqual(wake.calls, 2)
        self.assertEqual(engine.submitted, [])

    def test_malformed_transcription_executes_nothing(self):
        # A hallucinated/garbled transcript (e.g. "leum" on quiet audio) must
        # never execute anything, only fall back to wake-word listening.
        engine = _FakeEngine()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                router=VoiceIntentRouter(),
                control_engine=engine,
                stt=_FakeSTT("leum"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=1,
            )
        self.assertEqual(engine.submitted, [])

    def test_one_command_produces_exactly_one_execution(self):
        # "Hey Jarvis, increase the volume" must submit VOLUME_UP exactly once
        # -- the wake phrase tail and the command are ONE capture and ONE
        # routing, never two executions.
        engine = _FakeEngine()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                router=VoiceIntentRouter(),
                control_engine=engine,
                stt=_FakeSTT("hey jarvis increase the volume"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=1,
            )
        out = buf.getvalue()
        self.assertEqual(engine.submitted, [Command.VOLUME_UP])
        self.assertEqual(engine.submitted.count(Command.VOLUME_UP), 1)
        self.assertEqual(out.count("[voice] Done."), 1)

    def test_single_cycle_per_wake_detection(self):
        wake = _FakeWake()
        tts = mock.Mock()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=tts,
                monitor=_FakeMonitor(),
                max_cycles=2,
            )
        self.assertEqual(wake.calls, 2)
        self.assertEqual(wake.resets, 2)
        self.assertEqual(tts.call_count, 0)  # no ack by default

    def test_wake_state_settles_before_each_command_session(self):
        wake = _FakeWake()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=1,
            )
        self.assertEqual(wake.resets, 1)
        self.assertGreaterEqual(wake.calls, wake.resets)

    def test_default_dependencies_built_and_owned(self):
        wake = _FakeWake()
        engine = _FakeEngine()
        monitor = _FakeMonitor()
        with mock.patch.object(
            pipeline, "SpeechToText", return_value=_FakeSTT("volume up")
        ), mock.patch.object(
            pipeline, "create_wake_word_engine", return_value=wake
        ), mock.patch.object(
            pipeline, "ControlEngine", return_value=engine
        ), mock.patch.object(
            pipeline, "MicMonitor", return_value=monitor
        ), mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(max_cycles=1)
        self.assertTrue(monitor.opened)
        self.assertTrue(monitor.closed)
        self.assertTrue(wake.closed)
        self.assertTrue(engine.closed)
        self.assertEqual(engine.submitted, [Command.VOLUME_UP])

    def test_caller_provided_engine_is_not_closed_by_loop(self):
        engine = _FakeEngine()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                stt=_FakeSTT("volume up"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=1,
            )
        self.assertFalse(engine.closed)

    def test_injected_monitor_is_not_closed_by_loop(self):
        monitor = _FakeMonitor()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                monitor=monitor,
                max_cycles=1,
            )
        self.assertTrue(monitor.opened)
        self.assertFalse(monitor.closed)

    def test_source_tag_rides_with_the_command(self):
        """A spoken command arrives as source="voice", like a gesture's
        "gesture", so the reliability report can tell the routes apart."""

        engine = _FakeEngine()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                stt=_FakeSTT("volume up"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                source="voice",
                monitor=_FakeMonitor(),
                max_cycles=1,
            )
        self.assertEqual(engine.sources, ["voice"])

    def test_source_defaults_to_untagged(self):
        """Callers that never set a source get today's behaviour: untagged."""

        engine = _FakeEngine()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                stt=_FakeSTT("volume up"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                max_cycles=1,
            )
        self.assertEqual(engine.sources, [""])

    def test_should_stop_ends_the_loop_before_any_cycle(self):
        """A stop requested before the first wake word is honoured at once,
        and the wake engine is still closed."""

        wake = _FakeWake()
        engine = _FakeEngine()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                should_stop=lambda: True,
            )
        self.assertEqual(wake.calls, 0)
        self.assertTrue(wake.closed)
        self.assertEqual(engine.submitted, [])

    def test_should_stop_ends_the_loop_between_cycles(self):
        """One completed cycle, then the stop signal lands and the loop ends."""

        wake = _FakeWake()
        engine = _FakeEngine()
        asked = [0]

        def maybe_stop():
            asked[0] += 1
            return asked[0] >= 2

        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                monitor=_FakeMonitor(),
                should_stop=maybe_stop,
            )
        self.assertEqual(wake.calls, 1)
        self.assertTrue(wake.closed)
        self.assertEqual(engine.submitted, [Command.VOLUME_UP])

    def test_wake_unavailable_stops_cleanly(self):
        class _BrokenWake(_FakeWake):
            def initialize(self):
                raise WakeWordError("no model on disk")

        engine = _FakeEngine()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "ControlEngine", return_value=engine
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                wake_engine=_BrokenWake(),
                stt=_FakeSTT("volume up"),
                tts_speak=mock.Mock(),
                max_cycles=1,
            )
        out = buf.getvalue()
        self.assertIn("[wake-word] Wake-word unavailable: no model on disk", out)
        self.assertTrue(engine.closed)

    def test_no_microphone_stops_cleanly(self):
        class _BrokenMonitor(_FakeMonitor):
            def open(self):
                raise RuntimeError("no input device")

        engine = _FakeEngine()
        wake = _FakeWake()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "ControlEngine", return_value=engine
        ), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                wake_engine=wake,
                stt=_FakeSTT("volume up"),
                tts_speak=mock.Mock(),
                monitor=_BrokenMonitor(),
                max_cycles=1,
            )
        out = buf.getvalue()
        self.assertIn("[voice] ERROR: no microphone available: no input device", out)
        self.assertFalse(engine.closed)  # caller-provided engine stays with caller
        self.assertTrue(wake.closed)

    def test_shutdown_closes_monitor_and_wake(self):
        monitor = _FakeMonitor()
        wake = _FakeWake()
        with mock.patch.object(
            pipeline, "capture_command", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                monitor=monitor,
                should_stop=lambda: True,
            )
        self.assertTrue(wake.closed)


class StripWakePhraseCase(unittest.TestCase):
    def test_strips_leading_phrase(self):
        self.assertEqual(
            pipeline.strip_wake_phrase("hey jarvis increase the volume", "hey_jarvis_v0.1"),
            "increase the volume",
        )

    def test_strips_jarvis_only(self):
        self.assertEqual(
            pipeline.strip_wake_phrase("jarvis next track", "hey_jarvis_v0.1"),
            "next track",
        )

    def test_phrase_alone_becomes_empty(self):
        self.assertEqual(
            pipeline.strip_wake_phrase("Hey Jarvis", "hey_jarvis_v0.1"),
            "",
        )

    def test_punctuation_is_normalised_away(self):
        self.assertEqual(
            pipeline.strip_wake_phrase("hey jarvis, open chrome", "hey_jarvis_v0.1"),
            "open chrome",
        )

    def test_plain_command_passes_through(self):
        self.assertEqual(
            pipeline.strip_wake_phrase("volume up", "hey_jarvis_v0.1"),
            "volume up",
        )

    def test_empty_input_returns_empty(self):
        self.assertEqual(pipeline.strip_wake_phrase("", "hey_jarvis_v0.1"), "")


class RunVoiceLoopCompatibilityCase(unittest.TestCase):
    def test_run_voice_loop_still_works(self):
        stt = _FakeSTT("volume up")
        listener = _CompatWake()
        seen = []
        with mock.patch.object(
            pipeline, "SpeechToText", return_value=stt
        ), mock.patch.object(
            pipeline, "create_wake_word_engine", return_value=listener
        ), mock.patch.object(
            pipeline, "record_until_silence", return_value=_AUDIO
        ), redirect_stdout(io.StringIO()):
            with self.assertRaises(StopIteration):
                pipeline.run_voice_loop(on_text=lambda text: seen.append(text))
        self.assertEqual(seen, ["volume up"])
        self.assertTrue(listener.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
