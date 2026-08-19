"""Tests for the deterministic voice command pipeline (voice/pipeline.py).

Everything here runs with zero hardware: the wake-word engine, STT, mic
recording, TTS, router and ControlEngine are fakes/mocks. The real
VoiceIntentRouter is used (it is pure) so recognized/unknown routing is
verified against the actual command vocabulary.
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
from voice import pipeline
from voice.bridge import VoiceIntentRouter
from voice.config import CONFIG
from voice.wake_word import WakeWordError

#: A plausible mono float32 audio buffer for faked mic recordings.
_AUDIO = np.zeros(16000, dtype=np.float32)


class _FakeWake:
    """Wake engine that always fires; counts calls, resets and close."""

    name = "openwakeword"
    _model_names = ["hey_jarvis"]

    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.closed = False

    def initialize(self):
        pass

    def wait_for_wake_word(self):
        self.calls += 1

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


class _CompatWake(_FakeWake):
    """Raises StopIteration after one detection to end run_voice_loop's loop."""

    def wait_for_wake_word(self):
        self.calls += 1
        if self.calls > 1:
            raise StopIteration("test: end of compat loop")


class _FakeSTT:
    def __init__(self, transcript="volume up"):
        self.transcript = transcript
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.transcript


class _FakeEngine:
    def __init__(self):
        self.submitted = []
        self.closed = False

    def submit(self, command):
        self.submitted.append(command)

    def close(self):
        self.closed = True


class _FakeRouter:
    def __init__(self, command=None):
        self.command = command
        self.texts = []

    def classify(self, text):
        self.texts.append(text)
        return self.command


def _run(**kwargs):
    """Run the loop once with hardware-free defaults and capture stdout."""
    kwargs.setdefault("tts_speak", mock.Mock())
    kwargs.setdefault("stt", _FakeSTT())
    buf = io.StringIO()
    with mock.patch.object(
        pipeline, "record_until_silence", return_value=_AUDIO
    ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(buf):
        pipeline.run_voice_command_loop(
            wake_engine=_FakeWake(),
            control_engine=_FakeEngine(),
            max_cycles=1,
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
            pipeline, "record_until_silence", return_value=_AUDIO
        ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                router=VoiceIntentRouter(),
                control_engine=engine,
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=tts,
                max_cycles=1,
            )
        out = buf.getvalue()
        self.assertIn("[wake-word] WAKE WORD DETECTED: hey_jarvis", out)
        self.assertIn("[qrudo] Hey, how are you?", out)
        self.assertIn("[voice] Listening for command...", out)
        self.assertIn('[voice] Transcription: "volume up"', out)
        self.assertIn("[voice] Intent: VOLUME_UP", out)
        self.assertIn("[voice] Executing: VOLUME_UP", out)
        tts.assert_called_once_with("Hey, how are you?")
        self.assertEqual(engine.submitted, [Command.VOLUME_UP])
        self.assertTrue(wake.closed)

    def test_transcription_callback_receives_transcript(self):
        on_transcript = mock.Mock()
        _run(on_transcript=on_transcript)
        on_transcript.assert_called_once_with("volume up")

    def test_on_listening_fires_after_detection_before_response(self):
        order = []

        def listening():
            order.append("listening")

        def tts_speak(_text):
            order.append("tts")

        _run(on_listening=listening, tts_speak=tts_speak)
        self.assertEqual(order, ["listening", "tts"])

    def test_timing_instrumentation_printed_on_successful_command(self):
        out = _run()
        self.assertIn("[voice-timing]", out)
        for field in (
            "wake_to_tts=",
            "tts_duration=",
            "tts_to_record=",
            "record_duration=",
            "stt_duration=",
            "routing=",
            "execution=",
            "total_command=",
        ):
            self.assertIn(field, out)

    def test_no_timing_line_without_executed_command(self):
        # Unknown and empty transcripts must not print the [voice-timing] line.
        self.assertNotIn("[voice-timing]", _run(stt=_FakeSTT("open Chrome")))
        self.assertNotIn("[voice-timing]", _run(stt=_FakeSTT("")))

    def test_whisper_stats_printed_for_each_transcription(self):
        out = _run()
        self.assertIn("[whisper-stats]", out)
        self.assertIn("model=base", out)
        self.assertIn("empty=no", out)
        out_empty = _run(stt=_FakeSTT(""))
        self.assertIn("empty=yes", out_empty)

    def test_unknown_command_executes_nothing(self):
        engine = _FakeEngine()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "record_until_silence", return_value=_AUDIO
        ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                router=VoiceIntentRouter(),
                control_engine=engine,
                stt=_FakeSTT("open Chrome"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                max_cycles=1,
            )
        out = buf.getvalue()
        self.assertIn('[voice] Transcription: "open Chrome"', out)
        self.assertIn("[voice] No supported command matched. Nothing executed.", out)
        self.assertEqual(engine.submitted, [])
        self.assertNotIn("Executing", out)

    def test_injected_router_is_used(self):
        router = _FakeRouter(command=Command.FORWARD)
        engine = _FakeEngine()
        with mock.patch.object(
            pipeline, "record_until_silence", return_value=_AUDIO
        ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                router=router,
                control_engine=engine,
                stt=_FakeSTT("skip forward"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                max_cycles=1,
            )
        self.assertEqual(router.texts, ["skip forward"])
        self.assertEqual(engine.submitted, [Command.FORWARD])

    def test_no_speech_times_out_and_returns_to_wake(self):
        wake = _FakeWake()
        engine = _FakeEngine()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline, "record_until_silence", return_value=_AUDIO
        ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                stt=_FakeSTT(""),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                max_cycles=2,
            )
        out = buf.getvalue()
        self.assertIn(
            "[voice] No speech detected after the wake word. Back to wake-word listening.",
            out,
        )
        self.assertEqual(out.count("WAKE WORD DETECTED"), 2)
        self.assertEqual(wake.calls, 2)
        self.assertEqual(engine.submitted, [])

    def test_capture_error_does_not_kill_loop(self):
        wake = _FakeWake()
        buf = io.StringIO()
        with mock.patch.object(
            pipeline,
            "record_until_silence",
            side_effect=RuntimeError("mic went away"),
        ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(buf):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                max_cycles=2,
            )
        out = buf.getvalue()
        self.assertIn("[voice] ERROR: command capture failed: mic went away", out)
        self.assertEqual(out.count("WAKE WORD DETECTED"), 2)
        self.assertEqual(wake.calls, 2)

    def test_single_response_per_wake_detection(self):
        wake = _FakeWake()
        tts = mock.Mock()
        with mock.patch.object(
            pipeline, "record_until_silence", return_value=_AUDIO
        ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=tts,
                max_cycles=2,
            )
        self.assertEqual(wake.calls, 2)
        self.assertEqual(wake.resets, 2)
        self.assertEqual(tts.call_count, 2)

    def test_wake_state_settles_before_each_command_session(self):
        wake = _FakeWake()
        with mock.patch.object(
            pipeline, "record_until_silence", return_value=_AUDIO
        ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=wake,
                tts_speak=mock.Mock(),
                max_cycles=1,
            )
        self.assertEqual(wake.resets, 1)
        self.assertGreaterEqual(wake.calls, wake.resets)

    def test_tts_completes_before_command_recording(self):
        order = []

        def tts_speak(_text):
            order.append("tts")

        def fake_record(*_args, **_kwargs):
            order.append("record")
            return _AUDIO

        sleep = mock.Mock(side_effect=lambda s: order.append(f"sleep:{s}"))
        with mock.patch.object(
            pipeline, "record_until_silence", side_effect=fake_record
        ), mock.patch.object(pipeline.time, "sleep", sleep), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=_FakeEngine(),
                stt=_FakeSTT("volume up"),
                wake_engine=_FakeWake(),
                tts_speak=tts_speak,
                max_cycles=1,
            )
        self.assertEqual(order, ["tts", "sleep:0.5", "record"])
        self.assertEqual(sleep.call_args.args[0], CONFIG.post_tts_settle_s)

    def test_default_dependencies_built_and_owned(self):
        wake = _FakeWake()
        engine = _FakeEngine()
        with mock.patch.object(
            pipeline, "SpeechToText", return_value=_FakeSTT("volume up")
        ), mock.patch.object(
            pipeline, "create_wake_word_engine", return_value=wake
        ), mock.patch.object(
            pipeline, "ControlEngine", return_value=engine
        ), mock.patch.object(
            pipeline, "record_until_silence", return_value=_AUDIO
        ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(max_cycles=1)
        self.assertTrue(wake.closed)
        self.assertTrue(engine.closed)
        self.assertEqual(engine.submitted, [Command.VOLUME_UP])

    def test_caller_provided_engine_is_not_closed_by_loop(self):
        engine = _FakeEngine()
        with mock.patch.object(
            pipeline, "record_until_silence", return_value=_AUDIO
        ), mock.patch.object(pipeline.time, "sleep"), redirect_stdout(io.StringIO()):
            pipeline.run_voice_command_loop(
                control_engine=engine,
                stt=_FakeSTT("volume up"),
                wake_engine=_FakeWake(),
                tts_speak=mock.Mock(),
                max_cycles=1,
            )
        self.assertFalse(engine.closed)

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