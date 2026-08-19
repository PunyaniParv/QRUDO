"""
Speech-to-text using faster-whisper.

The model is loaded once and reused — loading it per-utterance would kill
latency. Import and instantiate SpeechToText a single time at startup
(pipeline.py does this for you).

Configuration is tuned for a local command assistant, not open-ended dictation:

* ``language="en"`` — forced English.  Auto-detection was measured to mis-fire
  on accented (Indian-English) speech and on quiet audio, and it added latency
  plus a crash path (faster-whisper's ``ValueError`` on blank VAD output).
* ``initial_prompt`` — off by default.  Measured on this machine, a decoder
  bias prompt ~quadrupled decode time and broke a clear wake clip
  ("Hey Jarvis." -> "H-J-R-W-S."), so it is not worth it here; the forced
  language and the sustained-energy gate below already fix the measured
  hallucinations.
* ``condition_on_previous_text=False`` — each utterance is decoded on its own,
  so a hallucination in one clip cannot bleed into the next.
* Pre-processing: leading/trailing silence is trimmed and the gain is
  normalised so a quiet microphone still reaches the model at full level.
* Gate: a clip must have real, sustained energy — at least ``stt_min_frames``
  active 10 ms frames at/above ``stt_min_peak``.  Quiet-room noise measured on
  this machine had peak 0.064 but only two isolated click-frames, and feeding
  that to the base model made it invent whole sentences; the gate returns ""
  instead.
"""

import numpy as np
from faster_whisper import WhisperModel

from voice.config import CONFIG


class SpeechToText:
    def __init__(self, config=None):
        cfg = config if config is not None else CONFIG
        self.config = cfg
        self.model_name = cfg.whisper_model_size
        self._model = WhisperModel(
            cfg.whisper_model_size,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
        )

    def transcribe(self, audio_float32: np.ndarray) -> str:
        """
        audio_float32: mono float32 samples in [-1, 1] at CONFIG.sample_rate.
        Returns the transcribed text, stripped. Empty string if nothing usable
        was heard.
        """
        if audio_float32.size == 0:
            return ""

        trimmed = _trim_silence(audio_float32, cfg=self.config)
        peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
        if peak < float(getattr(self.config, "stt_min_peak", 0.02)):
            # Nothing but quiet -- a clip quieter than this is where the base
            # model starts hallucinating whole sentences.
            return ""
        if not _has_sustained_energy(trimmed, cfg=self.config):
            # Isolated noise clicks can exceed the peak gate; real speech is
            # sustained over many 10 ms frames.  Only sustained audio reaches
            # the decoder, so two clicks can never become a hallucinated order.
            return ""

        boosted = _normalise_gain(trimmed)
        prompt = self.config.whisper_initial_prompt or None
        try:
            segments, _info = self._model.transcribe(
                boosted,
                language=self.config.whisper_language,
                beam_size=int(getattr(self.config, "whisper_beam_size", 1)),
                initial_prompt=prompt,
                condition_on_previous_text=False,
                vad_filter=True,      # whisper's own VAD trims filler noise
            )
        except ValueError:
            # Blank VAD output -- nothing audible, nothing to return.
            return ""
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()


def _trim_silence(audio: np.ndarray, *, cfg=None) -> np.ndarray:
    """Cut leading/trailing audio below an energy floor.

    Whisper's VAD does its own trimming, but removing dead leading/trailing
    samples first shortens the clip (less decode time) and removes the quiet
    tails that tempt the model into hallucinating.
    """
    if audio.size == 0:
        return audio
    frame = 160  # 10 ms at 16 kHz
    energies = []
    for start in range(0, audio.size - frame + 1, frame):
        block = audio[start:start + frame]
        energies.append(float(np.sqrt(np.mean(block.astype(np.float64) ** 2))))
    if not energies:
        return audio
    peak = max(energies)
    floor = max(peak * 0.05, 1e-4)
    first = last = None
    for i, e in enumerate(energies):
        if e >= floor:
            if first is None:
                first = i
            last = i
    if first is None:
        return np.array([], dtype=np.float32)
    start = first * frame
    end = min(audio.size, (last + 1) * frame)
    return audio[start:end]


def _normalise_gain(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Scale audio so its peak hits ``target_peak`` (quiet mics are boosted).

    Conservative: only amplifies, never clamps.  A clip that is already at a
    healthy level is left alone; an already-clipped clip is not made worse.
    """
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0 or peak >= target_peak:
        return audio
    gain = target_peak / peak
    return audio * gain


def _has_sustained_energy(audio: np.ndarray, *, cfg=None, frame: int = 160) -> bool:
    """True when the clip has real, sustained speech energy.

    Counts 10 ms frames whose RMS is at/above ``stt_min_peak`` and requires at
    least ``stt_min_frames`` of them.  This is what separates a real utterance
    from isolated noise clicks (measured: quiet-room ambient peaked at 0.064
    but only two frames crossed the floor, while real speech holds hundreds).
    """
    if audio.size < frame:
        return False
    required = int(getattr(cfg, "stt_min_frames", 5)) if cfg is not None else 5
    floor = float(getattr(cfg, "stt_min_peak", 0.02)) if cfg is not None else 0.02
    active = 0
    for start in range(0, audio.size - frame + 1, frame):
        block = audio[start:start + frame]
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        if rms >= floor:
            active += 1
            if active >= required:
                return True
    return False
