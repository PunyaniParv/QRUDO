"""Tests for the faster-whisper STT wrapper (voice/stt.py).

The WhisperModel backend is faked, so no model is downloaded/loaded and no
hardware is touched. Focus is the wrapper contract: empty/near-silent audio
returns "" and faster-whisper's empty-VAD crash (ValueError on language
auto-detection) is contained.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice.config import CONFIG
from voice.stt import SpeechToText


class _Segment:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self, segments=(), error=None):
        self.segments = segments
        self.error = error
        self.calls = []

    def transcribe(self, audio, language=None, beam_size=1, vad_filter=True,
                   initial_prompt=None, condition_on_previous_text=True):
        self.calls.append((audio, language, beam_size, vad_filter,
                           initial_prompt, condition_on_previous_text))
        if self.error is not None:
            raise self.error
        return self.segments, {}

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None


def _make_stt(model):
    with mock.patch("voice.stt.WhisperModel", return_value=model):
        return SpeechToText(), model


class SpeechToTextCase(unittest.TestCase):
    def test_empty_audio_returns_empty(self):
        stt, model = _make_stt(_FakeModel())
        self.assertEqual(stt.transcribe(np.array([], dtype=np.float32)), "")
        self.assertEqual(model.calls, [])

    def test_silent_audio_returns_empty_without_touching_model(self):
        stt, model = _make_stt(_FakeModel())
        silent = np.zeros(16000, dtype=np.float32)
        self.assertEqual(stt.transcribe(silent), "")
        self.assertEqual(model.calls, [])

    def test_near_silent_audio_returns_empty(self):
        stt, model = _make_stt(_FakeModel())
        audio = np.full(16000, 1e-6, dtype=np.float32)
        self.assertEqual(stt.transcribe(audio), "")
        self.assertEqual(model.calls, [])

    def test_whisper_empty_vad_crash_returns_empty(self):
        # faster-whisper raises ValueError("max() arg is an empty sequence")
        # when VAD trims everything and language auto-detection runs on nothing.
        stt, model = _make_stt(_FakeModel(error=ValueError("max() arg is an empty sequence")))
        audio = (np.random.default_rng(0).random(16000).astype(np.float32) - 0.5) * 0.2
        self.assertEqual(stt.transcribe(audio), "")

    def test_audio_with_speech_returns_joined_text(self):
        stt, model = _make_stt(_FakeModel(segments=[_Segment("open"), _Segment("Chrome")]))
        audio = (np.random.default_rng(1).random(16000).astype(np.float32) - 0.5) * 0.2
        self.assertEqual(stt.transcribe(audio), "open Chrome")
        self.assertEqual(len(model.calls), 1)

    def test_transcribe_passes_whisper_args(self):
        stt, model = _make_stt(_FakeModel(segments=[_Segment("volume up")]))
        audio = (np.random.default_rng(2).random(16000).astype(np.float32) - 0.5) * 0.2
        stt.transcribe(audio)
        (_audio, language, beam_size, vad_filter,
         initial_prompt, condition_on_previous_text) = model.calls[0]
        self.assertEqual(language, CONFIG.whisper_language)
        self.assertEqual(beam_size, 1)
        self.assertTrue(vad_filter)
        # A decoder bias prompt is off by default (None); each utterance is
        # decoded independently (a hallucination cannot bleed in).
        self.assertIsNone(initial_prompt)
        self.assertFalse(condition_on_previous_text)

    def test_configured_initial_prompt_is_forwarded(self):
        cfg = SimpleNamespace(
            whisper_model_size="base",
            whisper_device="cpu",
            whisper_compute_type="int8",
            whisper_language="en",
            whisper_beam_size=1,
            whisper_initial_prompt="open chrome, close tab.",
            stt_min_peak=0.02,
            stt_min_frames=5,
        )
        model = _FakeModel(segments=[_Segment("open chrome")])
        with mock.patch("voice.stt.WhisperModel", return_value=model):
            stt = SpeechToText(config=cfg)
        audio = (np.random.default_rng(4).random(16000).astype(np.float32) - 0.5) * 0.2
        stt.transcribe(audio)
        self.assertEqual(model.calls[0][4], "open chrome, close tab.")

    def test_isolated_clicks_do_not_reach_the_decoder(self):
        # A clip with a couple of loud single-frame clicks (quiet-room noise
        # pattern) must return "" without touching the model: real speech is
        # sustained over many 10 ms frames.
        stt, model = _make_stt(_FakeModel(segments=[_Segment("volume up")]))
        audio = np.zeros(16000, dtype=np.float32)
        audio[8000:8160] = 0.5      # one loud click
        audio[8160:8320] = 0.0      # silence again
        self.assertEqual(stt.transcribe(audio), "")
        self.assertEqual(model.calls, [])

    def test_sustained_speech_reaches_the_decoder(self):
        stt, model = _make_stt(_FakeModel(segments=[_Segment("volume up")]))
        audio = np.zeros(16000, dtype=np.float32)
        audio[4000:12000] = 0.5     # 0.5 s of real energy
        self.assertEqual(stt.transcribe(audio), "volume up")
        self.assertEqual(len(model.calls), 1)

    def test_transcribe_uses_config_beam_size(self):
        cfg = SimpleNamespace(
            whisper_model_size="base",
            whisper_device="cpu",
            whisper_compute_type="int8",
            whisper_language="en",
            whisper_beam_size=3,
            whisper_initial_prompt=None,
            stt_min_peak=0.02,
        )
        model = _FakeModel(segments=[_Segment("volume up")])
        with mock.patch("voice.stt.WhisperModel", return_value=model):
            stt = SpeechToText(config=cfg)
        audio = (np.random.default_rng(3).random(16000).astype(np.float32) - 0.5) * 0.2
        stt.transcribe(audio)
        self.assertEqual(model.calls[0][2], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)