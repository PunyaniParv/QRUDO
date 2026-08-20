"""Tests for voice/audio_capture.py command capture and compatibility path.

Hardware-free: the monitor is faked with a scripted frame sequence. These
tests verify the adaptive-silence capture behaviour, the pre-roll seeding,
max-duration truncation, the no-speech fast path, and the record_until_silence
compatibility wrapper.
"""

from __future__ import annotations

import importlib
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice import audio_capture

# 80 ms frames at 16 kHz (1280 samples) -- the monitor frame size.
_FRAME_SAMPLES = 1280
_LOUD = np.full(_FRAME_SAMPLES, 1000, dtype=np.int16)   # RMS 1000 >= threshold
_SILENT = np.zeros(_FRAME_SAMPLES, dtype=np.int16)


def _cfg(**overrides):
    base = {
        "frame_samples": _FRAME_SAMPLES,
        "sample_rate": 16000,
        "silence_duration_s": 0.7,
        "min_recording_s": 0.3,
        "max_command_s": 8.0,
        "pre_speech_timeout_s": 3.0,
        "silence_threshold_rms": 300.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeMonitor:
    """Fake MicMonitor replaying a scripted frame sequence."""

    def __init__(self, frames, pre_roll=None):
        self._frames = list(frames)
        self.pre_roll_frames = list(pre_roll or [])
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def next_frame(self, timeout=0.25, stop=None, consumer=""):
        if stop is not None and stop():
            return None
        if self._frames:
            return self._frames.pop(0)
        return None

    def pre_roll(self):
        return list(self.pre_roll_frames)

    def pending_count(self):
        return 0

    def close(self):
        self.closed = True


class _Clock:
    """Scripted monotonic clock; each call advances by a frame interval."""

    def __init__(self, start=0.0, step=0.08):
        self.t = start
        self.step = step

    def __call__(self):
        t = self.t
        self.t += self.step
        return t


class CaptureCommandCase(unittest.TestCase):
    def test_ends_on_adaptive_silence_after_min_speech(self):
        # 5 loud frames (>= min_recording_s) then 12 silent frames
        # (>= silence_duration_s = 9 frames) ends the utterance.
        monitor = _FakeMonitor([_LOUD] * 5 + [_SILENT] * 12)
        audio = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNotNone(audio)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape[0], 14 * _FRAME_SAMPLES)  # 5 speech + 9 quiet

    def test_prints_nothing_without_stats(self):
        monitor = _FakeMonitor([_LOUD] * 5 + [_SILENT] * 12)
        buf = io.StringIO()
        with redirect_stdout(buf):
            audio_capture.capture_command(monitor, config=_cfg())
        self.assertEqual(buf.getvalue(), "")

    def test_debug_prints_capture_debug_block(self):
        # QRUDO_VOICE_DEBUG / debug=True prints one aggregate [capture-debug]
        # block with the handoff telemetry the regression needed to diagnose.
        monitor = _FakeMonitor(
            [_LOUD] * 12 + [_SILENT] * 12, pre_roll=[_LOUD]
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            audio_capture.capture_command(monitor, config=_cfg(), debug=True)
        out = buf.getvalue()
        for line in (
            "[capture-debug]",
            "frames_seen=",
            "frames_appended=",
            "pre_roll_frames=",
            "loud_frames=",
            "quiet_frames=",
            "gap_confirmed=",
            "post_wake_grace_s=",
            "command_started=",
            "ended_reason=",
            "[capture-audio-debug]",
            "total_samples=",
            "dtype=float32",
            "duration=",
            "rms=",
        ):
            self.assertIn(line, out)

    def test_synthetic_sequence_returns_real_pcm(self):
        # A deterministic 16 kHz scene: quiet frames, a wake-tail pre-roll,
        # speech frames with non-zero PCM, then trailing silence.  capture
        # must return real audio -- not an empty/zeroed buffer.
        cfg = _cfg()
        pre_roll = [_LOUD, _LOUD]                      # wake phrase tail
        live = (
            [_LOUD] * 8                                # wake tail run
            + [_SILENT] * 15                           # ~1.2 s pause
            + [_LOUD] * 12                             # the command
            + [_SILENT] * 12                           # trailing silence
        )
        monitor = _FakeMonitor(live, pre_roll=pre_roll)
        audio, stats = audio_capture.capture_command(
            monitor, config=cfg, stats=True
        )
        self.assertIsNotNone(audio)
        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(audio.size, 0)
        # pre-roll + tail + command + trailing quiet (pause skipped).
        self.assertEqual(audio.size, (2 + 8 + 12 + 9) * _FRAME_SAMPLES)
        self.assertGreater(float(np.max(np.abs(audio))), 0.0)
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        self.assertGreater(rms, 0.0)
        self.assertGreater(audio.size / cfg.sample_rate, 1.0)
        self.assertGreater(stats["samples"], 0)

    def test_captured_audio_survives_stt_gate(self):
        # The buffer capture_command returns must pass the STT preprocessing
        # path (trim + sustained-energy gate) unchanged -- capture-side VAD
        # already proved speech exists, so STT must see real samples.
        from voice.stt import _has_sustained_energy, _trim_silence

        monitor = _FakeMonitor(
            [_LOUD] * 10 + [_SILENT] * 12, pre_roll=[_LOUD, _LOUD]
        )
        audio = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNotNone(audio)
        trimmed = _trim_silence(audio, cfg=_cfg())
        self.assertGreater(trimmed.size, 0)
        self.assertTrue(
            _has_sustained_energy(trimmed, cfg=_cfg()),
            "captured speech must not be rejected by the STT energy gate",
        )

    def test_real_machine_evidence_shape(self):
        # Reproduce the shape from the failing real test: >= 30 frames
        # appended, command_started=1 -- the PCM must be > 1 s of real audio.
        monitor = _FakeMonitor(
            [_LOUD] * 19 + [_SILENT] * 12, pre_roll=[_LOUD, _LOUD]
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            audio, stats = audio_capture.capture_command(
                monitor, config=_cfg(), stats=True, debug=True
            )
        self.assertIsNotNone(audio)
        out = buf.getvalue()
        self.assertIn("command_started=1", out)
        self.assertIn("gap_confirmed=0", out)
        frames_appended = 2 + 19 + 9  # pre-roll + loud run + trailing quiet
        self.assertGreaterEqual(frames_appended, 30)
        self.assertGreater(audio.size, 0)
        self.assertGreater(audio.size / 16000.0, 1.0)
        self.assertGreater(stats["samples"], 0)

    def test_config_debug_flag_accepts_qrudo_env(self):
        # QRUDO_VOICE_DEBUG=1 (the requested variable) must enable diagnostics;
        # SARV_VOICE_DEBUG remains accepted for backward compatibility.
        from voice import config as config_module

        def reload_with(**env):
            with mock.patch.dict("os.environ", env, clear=False):
                return importlib.reload(config_module)

        # Reload with each candidate env and observe the debug flag.
        try:
            restored = dict(os.environ)
        except Exception:
            restored = {}
        config_module.CONFIG = reload_with(QRUDO_VOICE_DEBUG="1", SARV_VOICE_DEBUG="").CONFIG
        self.assertTrue(config_module.CONFIG.debug)
        config_module.CONFIG = reload_with(QRUDO_VOICE_DEBUG="", SARV_VOICE_DEBUG="1").CONFIG
        self.assertTrue(config_module.CONFIG.debug)
        config_module.CONFIG = reload_with(QRUDO_VOICE_DEBUG="", SARV_VOICE_DEBUG="").CONFIG
        self.assertFalse(config_module.CONFIG.debug)
        # Restore the module to the pre-test environment state.
        os.environ.clear()
        os.environ.update({k: v for k, v in restored.items() if v is not None})
        config_module.CONFIG = reload_with(
            **{k: v for k, v in restored.items() if v is not None}
        ).CONFIG

    def test_returns_stats_when_requested(self):
        monitor = _FakeMonitor([_LOUD] * 5 + [_SILENT] * 12)
        audio, stats = audio_capture.capture_command(monitor, config=_cfg(), stats=True)
        self.assertIsNotNone(audio)
        for key in ("first_speech_after", "duration", "final_silence", "samples"):
            self.assertIn(key, stats)
        self.assertEqual(stats["samples"], 14 * _FRAME_SAMPLES)
        self.assertAlmostEqual(stats["final_silence"], 0.72, places=2)

    def test_no_speech_returns_none(self):
        monitor = _FakeMonitor([_SILENT] * 5)  # quiet, then mic starves
        result = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNone(result)

    def test_pre_roll_prepends_wake_tail(self):
        # The monitor's rolling memory (the wake phrase tail) is prepended so a
        # command in the same breath as the wake word is never clipped.  With a
        # pre-roll present, capture starts in wake context: the command follows
        # the wake tail with no pause, so the loud run past the post-wake grace
        # is the command and the trailing silence ends the utterance.
        monitor = _FakeMonitor(
            [_LOUD] * 10 + [_SILENT] * 12, pre_roll=[_LOUD, _LOUD]
        )
        audio = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNotNone(audio)
        # 2 pre-roll + 10 loud (tail->command) + 9 trailing quiet.
        self.assertEqual(audio.shape[0], (2 + 10 + 9) * _FRAME_SAMPLES)

    def test_two_stage_pause_between_wake_and_command(self):
        # "hey jarvis" [1.2s pause] "increase the volume": the silence between
        # the wake tail and the command is the pause, NOT the end of the
        # utterance.  The capture must keep listening and capture the command.
        monitor = _FakeMonitor(
            [_LOUD] * 2            # wake tail
            + [_SILENT] * 15       # ~1.2s pause (>= silence_duration_s)
            + [_LOUD] * 6          # the command
            + [_SILENT] * 12,      # trailing silence
            pre_roll=[_LOUD, _LOUD],
        )
        audio = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNotNone(audio)
        # pre-roll + tail + command + trailing quiet (the pause is skipped).
        self.assertEqual(audio.shape[0], (2 + 2 + 6 + 9) * _FRAME_SAMPLES)

    def test_two_stage_pause_within_pre_speech_timeout(self):
        # A pause close to the 3.0s pre_speech_timeout still captures the
        # command; only exceeding it returns None.
        cfg = _cfg()
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            monitor = _FakeMonitor(
                [_LOUD] * 2 + [_SILENT] * 30 + [_LOUD] * 5 + [_SILENT] * 12,
                pre_roll=[_LOUD],
            )
            audio = audio_capture.capture_command(monitor, config=cfg)
        self.assertIsNotNone(audio)

    def test_pause_longer_than_pre_speech_timeout_returns_none(self):
        # The user never says the command within pre_speech_timeout_s: the
        # capture must give up and report no speech.
        cfg = _cfg()
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            monitor = _FakeMonitor(
                [_LOUD] * 2 + [_SILENT] * 45, pre_roll=[_LOUD]
            )
            audio = audio_capture.capture_command(monitor, config=cfg)
        self.assertIsNone(audio)

    def test_same_utterance_command_after_wake(self):
        # "hey jarvis, increase the volume" with no pause: the loud run flowing
        # past the post-wake grace is the command, and the trailing silence ends
        # it (a long continuous run must not be mistaken for the wake tail).
        cfg = _cfg()
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            monitor = _FakeMonitor(
                [_LOUD] * 20 + [_SILENT] * 12, pre_roll=[_LOUD]
            )
            audio = audio_capture.capture_command(monitor, config=cfg)
        self.assertIsNotNone(audio)

    def test_wake_tail_only_returns_none(self):
        # "hey jarvis" followed by silence and then nothing: no command was
        # said, so capture reports no speech.
        monitor = _FakeMonitor(
            [_LOUD] * 3 + [_SILENT] * 12, pre_roll=[_LOUD, _LOUD]
        )
        result = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNone(result)

    def test_pre_roll_preserved_in_audio(self):
        # The pre-roll wake tail is always part of the returned clip, so a
        # same-breath command is never clipped.
        monitor = _FakeMonitor(
            [_LOUD] * 12 + [_SILENT] * 12, pre_roll=[_LOUD, _LOUD, _LOUD]
        )
        audio = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNotNone(audio)
        self.assertEqual(audio.shape[0], (3 + 12 + 9) * _FRAME_SAMPLES)

    def test_real_machine_command_survives_mid_command_dip(self):
        # Regression: the exact real-machine frame RMS sequence that failed in
        # production.  The command burst is 151..161 (RMS 429 .. 302), but a
        # mid-command dip (seq 152, RMS 164.9, a quiet syllable between words)
        # used to feed the adaptive noise-floor estimator, raising the gate
        # from 300 to ~311 so the command-tail frame (seq 161, RMS 302.7) was
        # misclassified as quiet and the burst never reached the 0.8s
        # post-wake grace -> capture returned None ("no speech after wake").
        # The noise floor must only adapt during pre-speech silence; once the
        # first loud frame is seen the gate is frozen.
        rms_seq = [
            7.6, 6.9, 81.4,               # leading silence
            429.0, 164.9, 367.8,          # burst starts, then the dip
            1321.2, 700.9, 409.7, 855.4,
            1167.5, 716.8, 869.0, 302.7,  # burst tail (was dropped before)
        ] + [8.0] * 26                    # trailing silence
        frames = [np.full(_FRAME_SAMPLES, r, dtype=np.int16) for r in rms_seq]
        monitor = _FakeMonitor(frames, pre_roll=[_SILENT] * 4)
        audio = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNotNone(audio)
        self.assertGreater(audio.shape[0] / 16000, 1.0)  # > 1s of audio

    def test_shutdown_mid_capture_returns_none(self):
        # Ctrl+C / shutdown signal during the wake-command pause ends the
        # capture without a command.
        called = {"n": 0}

        def flaky_stop():
            called["n"] += 1
            return called["n"] >= 8  # stop partway through the pause

        monitor = _FakeMonitor(
            [_LOUD] * 2 + [_SILENT] * 40, pre_roll=[_LOUD]
        )
        result = audio_capture.capture_command(
            monitor, config=_cfg(), stop=flaky_stop
        )
        self.assertIsNone(result)

    def test_max_command_s_truncates_long_utterance(self):
        cfg = _cfg(max_command_s=0.3)  # ~3.75 frames -> truncate early
        monitor = _FakeMonitor([_LOUD] * 10)
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            audio = audio_capture.capture_command(monitor, config=cfg)
        self.assertIsNotNone(audio)
        self.assertLess(audio.shape[0], 10 * _FRAME_SAMPLES)

    def test_stop_ends_capture_immediately(self):
        monitor = _FakeMonitor([_LOUD] * 5)
        result = audio_capture.capture_command(
            monitor, config=_cfg(), stop=lambda: True
        )
        self.assertIsNone(result)


class CommandCaptureSensitivityCase(unittest.TestCase):
    """Command-capture sensitivity regressions.

    These reproduce the real-machine failure shape: normal conversational
    speech (frame RMS far below the old absolute 300 floor), natural pauses
    between the wake word and the command, and intermittent low-energy
    syllables.  Each must now be captured as ONE utterance, while pure
    background noise and short noise bursts must still be rejected.
    """

    def _cfg(self, **overrides):
        # Real defaults: a LOW absolute floor + relative gate.
        overrides.setdefault("silence_threshold_rms", 100.0)
        return _cfg(**overrides)

    def _frame_seq(self, rms_seq):
        return [np.full(_FRAME_SAMPLES, r, dtype=np.int16) for r in rms_seq]

    def test_speech_immediately_after_wake(self):
        # "hey jarvis, increase the volume" same-breath at normal volume.
        monitor = _FakeMonitor(
            [_LOUD] * 12 + [_SILENT] * 12, pre_roll=[_LOUD, _LOUD]
        )
        audio, stats = audio_capture.capture_command(
            monitor, config=self._cfg(), stats=True
        )
        self.assertIsNotNone(audio)
        self.assertEqual(stats["ended_reason"], "silence_after_command")

    def test_speech_after_half_second_pause(self):
        # "hey jarvis" [~0.5s] "increase the volume".
        cfg = self._cfg()
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            monitor = _FakeMonitor(
                [_SILENT] * 6 + [_LOUD] * 12 + [_SILENT] * 12,
                pre_roll=[_LOUD],
            )
            audio, stats = audio_capture.capture_command(
                monitor, config=cfg, stats=True
            )
        self.assertIsNotNone(audio)
        self.assertEqual(stats["ended_reason"], "silence_after_command")

    def test_speech_after_one_second_pause(self):
        # "hey jarvis" [~1s] "increase the volume".
        cfg = self._cfg()
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            monitor = _FakeMonitor(
                [_SILENT] * 12 + [_LOUD] * 12 + [_SILENT] * 12,
                pre_roll=[_LOUD],
            )
            audio, stats = audio_capture.capture_command(
                monitor, config=cfg, stats=True
            )
        self.assertIsNotNone(audio)
        self.assertEqual(stats["ended_reason"], "silence_after_command")

    def test_speech_after_one_and_half_second_pause(self):
        # "hey jarvis" [~1.5s] "increase the volume".
        cfg = self._cfg()
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            monitor = _FakeMonitor(
                [_SILENT] * 18 + [_LOUD] * 12 + [_SILENT] * 12,
                pre_roll=[_LOUD],
            )
            audio, stats = audio_capture.capture_command(
                monitor, config=cfg, stats=True
            )
        self.assertIsNotNone(audio)
        self.assertEqual(stats["ended_reason"], "silence_after_command")

    def test_normal_volume_speech_is_captured(self):
        # Normal conversational speech: frames at 150-400 RMS in a quiet room.
        # The old absolute 300 floor classified most of these as quiet; the
        # relative gate (max(100, ~3x floor)) lets them through.
        rms_seq = [35.0] * 8
        rms_seq += [200.0, 180.0, 340.0, 220.0, 160.0, 300.0,
                    250.0, 380.0, 210.0, 170.0, 330.0, 240.0]
        rms_seq += [35.0] * 12
        monitor = _FakeMonitor(
            self._frame_seq(rms_seq), pre_roll=[_LOUD]
        )
        audio, stats = audio_capture.capture_command(
            monitor, config=self._cfg(), stats=True
        )
        self.assertIsNotNone(audio)
        self.assertEqual(stats["ended_reason"], "silence_after_command")
        self.assertGreater(audio.size / 16000.0, 1.0)

    def test_quiet_speech_is_captured(self):
        # Quieter-than-normal speech in a quiet room (~120-160 RMS frames).
        rms_seq = [30.0] * 8
        rms_seq += [160.0, 120.0, 150.0, 130.0, 155.0, 125.0, 140.0, 145.0]
        rms_seq += [30.0] * 12
        monitor = _FakeMonitor(
            self._frame_seq(rms_seq), pre_roll=[_LOUD]
        )
        audio = audio_capture.capture_command(monitor, config=self._cfg())
        self.assertIsNotNone(audio)

    def test_loud_speech_is_captured(self):
        rms_seq = [40.0] * 4 + [2000.0] * 12 + [40.0] * 12
        monitor = _FakeMonitor(
            self._frame_seq(rms_seq), pre_roll=[_LOUD]
        )
        audio = audio_capture.capture_command(monitor, config=self._cfg())
        self.assertIsNotNone(audio)

    def test_mid_command_quiet_syllable_keeps_utterance_alive(self):
        # Regression (must stay fixed): a quiet syllable between words must
        # NOT terminate the utterance, and once speech has begun the gate must
        # stay frozen so a later tail frame is not dropped.
        rms_seq = [35.0] * 8                       # quiet room -> gate ~127.5
        rms_seq += [400.0, 90.0, 130.0, 500.0, 700.0,
                    130.0, 620.0, 130.0, 300.0]     # burst with a 90 RMS dip
        rms_seq += [35.0] * 12                      # trailing silence
        monitor = _FakeMonitor(
            self._frame_seq(rms_seq), pre_roll=[_LOUD]
        )
        audio, stats = audio_capture.capture_command(
            monitor, config=self._cfg(), stats=True
        )
        self.assertIsNotNone(audio)
        self.assertEqual(stats["ended_reason"], "silence_after_command")
        self.assertGreater(audio.size / 16000.0, 1.0)

    def test_background_noise_without_speech_returns_none(self):
        # Continuous room noise just above a quiet floor must never become a
        # command, even after a natural wake pause.
        cfg = self._cfg()
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            monitor = _FakeMonitor(
                self._frame_seq([55.0] * 45), pre_roll=[_LOUD]
            )
            result = audio_capture.capture_command(monitor, config=cfg)
        self.assertIsNone(result)

    def test_short_noise_burst_without_speech_returns_none(self):
        # A brief loud burst (door slam / cough) right after wake with no real
        # speech following must not become a command.
        rms_seq = [800.0, 600.0, 900.0] + [40.0] * 45
        monitor = _FakeMonitor(
            self._frame_seq(rms_seq), pre_roll=[_LOUD]
        )
        result = audio_capture.capture_command(monitor, config=self._cfg())
        self.assertIsNone(result)

    def test_intermittent_low_energy_onset(self):
        # Same-breath speech whose frames dip below the gate between
        # syllables, with no pause after the wake.  Speech onset must be
        # established by interleaved evidence, not a long uninterrupted run.
        rms_seq = [250.0, 60.0, 300.0, 50.0, 380.0, 70.0,
                   310.0, 55.0, 350.0, 340.0] + [40.0] * 12
        monitor = _FakeMonitor(
            self._frame_seq(rms_seq), pre_roll=[_LOUD]
        )
        audio, stats = audio_capture.capture_command(
            monitor, config=self._cfg(), stats=True
        )
        self.assertIsNotNone(audio)
        self.assertEqual(stats["ended_reason"], "silence_after_command")
        self.assertEqual(stats["command_after"], 0.0)  # same-breath: no pause

    def test_noise_floor_frozen_once_speech_begins(self):
        # Once speech onset is established the noise floor / gate must remain
        # frozen for that utterance.  A quiet frame after onset (a dip) must
        # not feed the estimator and raise the gate above later command-tail
        # frames.  With a live estimator the 90 RMS dip would push the gate
        # from ~127 to ~142 and drop the 130 RMS tail frames.
        rms_seq = [35.0] * 8
        rms_seq += [400.0, 90.0, 130.0, 500.0, 130.0, 700.0, 130.0, 300.0]
        rms_seq += [35.0] * 12
        monitor = _FakeMonitor(
            self._frame_seq(rms_seq), pre_roll=[_LOUD]
        )
        audio, stats = audio_capture.capture_command(
            monitor, config=self._cfg(), stats=True
        )
        self.assertIsNotNone(audio)
        self.assertEqual(stats["ended_reason"], "silence_after_command")
        self.assertGreater(stats["samples"], 0)

    def test_post_speech_silence_still_terminates_capture(self):
        # A real command followed by silence still ends the utterance promptly
        # (bounded, not waiting for max_command_s).
        cfg = self._cfg(max_command_s=8.0)
        monitor = _FakeMonitor(
            [_LOUD] * 12 + [_SILENT] * 30, pre_roll=[_LOUD]
        )
        audio, stats = audio_capture.capture_command(
            monitor, config=cfg, stats=True
        )
        self.assertIsNotNone(audio)
        self.assertEqual(stats["ended_reason"], "silence_after_command")
        self.assertLess(stats["final_silence"], 2.0)

    def test_pre_speech_timeout_still_bounded(self):
        # Never wait forever: pure silence after wake still times out and
        # returns None within the configured window.
        cfg = self._cfg()
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            monitor = _FakeMonitor(
                [_SILENT] * 50, pre_roll=[_LOUD]
            )
            result = audio_capture.capture_command(monitor, config=cfg)
        self.assertIsNone(result)


class RecordUntilSilenceCompatibilityCase(unittest.TestCase):
    def _run(self, frames, report=False):
        monitor = _FakeMonitor(frames)
        buf = io.StringIO()
        with mock.patch.object(
            audio_capture, "MicMonitor", return_value=monitor
        ), redirect_stdout(buf):
            audio = audio_capture.record_until_silence(report=report)
        return audio, buf.getvalue(), monitor

    def test_compat_path_records_one_utterance(self):
        audio, out, monitor = self._run([_LOUD] * 5 + [_SILENT] * 12)
        self.assertEqual(out, "")
        self.assertEqual(audio.shape[0], 14 * _FRAME_SAMPLES)
        self.assertTrue(monitor.closed)

    def test_compat_path_returns_empty_on_no_speech(self):
        audio, out, monitor = self._run([_SILENT] * 5)
        self.assertEqual(audio.shape[0], 0)
        self.assertTrue(monitor.closed)

    def test_compat_path_report_prints_stats(self):
        audio, out, monitor = self._run([_LOUD] * 5 + [_SILENT] * 12, report=True)
        self.assertIn("[record-stats]", out)
        self.assertIn("first_speech_after=", out)
        self.assertIn("duration=", out)
        self.assertIn("final_silence=0.72s", out)
        self.assertIn(f"samples={14 * _FRAME_SAMPLES}", out)

    def test_compat_path_report_prints_no_speech(self):
        audio, out, monitor = self._run([_SILENT] * 5, report=True)
        self.assertIn("[record-stats]", out)
        self.assertIn("no_speech=yes", out)
        self.assertEqual(audio.shape[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)