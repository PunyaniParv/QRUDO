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

    def transcribe(self, audio, language=None, beam_size=1, vad_filter=True):
        self.calls.append((audio, language, beam_size, vad_filter))
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
        _audio, language, beam_size, vad_filter = model.calls[0]
        self.assertEqual(language, CONFIG.whisper_language)
        self.assertEqual(beam_size, 1)
        self.assertTrue(vad_filter)


if __name__ == "__main__":
    unittest.main(verbosity=2)