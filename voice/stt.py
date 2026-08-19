"""
Speech-to-text using faster-whisper.

The model is loaded once and reused — loading it per-utterance would kill
latency. Import and instantiate SpeechToText a single time at startup
(pipeline.py does this for you).
"""

import numpy as np
from faster_whisper import WhisperModel

from voice.config import CONFIG


class SpeechToText:
    def __init__(self):
        self.model_name = CONFIG.whisper_model_size
        self._model = WhisperModel(
            CONFIG.whisper_model_size,
            device=CONFIG.whisper_device,
            compute_type=CONFIG.whisper_compute_type,
        )

    def transcribe(self, audio_float32: np.ndarray) -> str:
        """
        audio_float32: mono float32 samples in [-1, 1] at CONFIG.sample_rate.
        Returns the transcribed text, stripped. Empty string if nothing was heard.
        """
        if audio_float32.size == 0:
            return ""

        # faster-whisper's language auto-detection crashes with
        # `ValueError: max() arg is an empty sequence` when its VAD trims
        # everything (i.e. the user stayed silent). Treat essentially-silent
        # audio as "nothing said" up front, and defend against the same crash
        # on very short/quiet clips that still reach the model.
        peak = float(np.max(np.abs(audio_float32)))
        if peak < 1e-4:
            return ""

        try:
            segments, _info = self._model.transcribe(
                audio_float32,
                language=CONFIG.whisper_language,
                beam_size=1,          # beam_size=1 (greedy) is noticeably faster; raise if accuracy suffers
                vad_filter=True,      # whisper's own VAD trims leading/trailing silence and filler noise
            )
        except ValueError:
            # Blank VAD output again -- nothing audible, nothing to return.
            return ""
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()