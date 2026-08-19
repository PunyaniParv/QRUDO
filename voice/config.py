"""
Central configuration for SARV's voice module.

All tunable values live here so latency/behavior can be adjusted without
touching pipeline logic. Nothing here talks to hardware.
"""

import os
from dataclasses import dataclass


def _env_microphone_device() -> str | int | None:
    """Read SARV_MICROPHONE_DEVICE into the config field representation.

    A plain integer string becomes a device *index*; anything else is kept as
    a *name* substring (case-insensitive), which is exactly what sounddevice's
    ``device`` argument accepts. Empty/unset -> None (use the OS default).
    """
    raw = os.getenv("SARV_MICROPHONE_DEVICE")
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    return raw


@dataclass(frozen=True)
class VoiceConfig:
    # --- Audio capture ---
    sample_rate: int = 16000          # openWakeWord and Whisper both want 16kHz mono
    channels: int = 1

    # Which microphone to use. None = the operating system's default input
    # device. Otherwise, because sounddevice accepts either a device index or
    # a substring of a device name, this field mirrors that: an int is a
    # device index, a str is a (case-insensitive) substring of a device name.
    # Set it via the SARV_MICROPHONE_DEVICE environment variable (e.g.
    # "SARV_MICROPHONE_DEVICE=USB" or "SARV_MICROPHONE_DEVICE=2").
    # If the configured device is unavailable (e.g. a Bluetooth headset that
    # disconnected), SARV falls back to the OS default instead of crashing.
    microphone_device: str | int | None = _env_microphone_device()

    # --- Wake word (local, replaceable engine) ---
    # Which wake-word backend to use. The rest of the pipeline only talks to
    # the WakeWordEngine interface (voice/wake_word.py); this field selects the
    # concrete implementation. Currently only "openwakeword" is supported.
    wake_word_engine: str = os.getenv("SARV_WAKE_WORD_ENGINE", "openwakeword")

    # Which bundled openWakeWord *pretrained* model name to use when
    # wake_word_model_path is unset. Valid names: alexa, hey_jarvis,
    # hey_mycroft, hey_rhasspy, timer, weather.
    # LICENSE CAVEAT: these pretrained models are CC BY-NC-SA 4.0
    # (non-commercial). Fine for local development/manual testing, but MUST NOT
    # be shipped in a commercial QRUDO build. For production, train a custom
    # "Hey QRUDO" model on permissive data and point wake_word_model_path at it.
    wake_word_model_name: str = os.getenv("SARV_WAKE_WORD_MODEL_NAME", "hey_jarvis")

    # Absolute path to a custom wake-word model file (.onnx). When set, this
    # takes precedence over wake_word_model_name. This is where a trained
    # "Hey QRUDO" model goes. Leave None to use wake_word_model_name.
    wake_word_model_path: str | None = (
        os.getenv("SARV_WAKE_WORD_MODEL_PATH") or None
    )

    # Detection threshold: 0.0 (fewest false positives, less sensitive) to 1.0
    # (most sensitive, more false positives). 0.5 is openwakeword's recommended
    # default.
    wake_word_sensitivity: float = 0.5

    # --- Silence-based end-of-utterance detection ---
    # Simple RMS energy check — no extra native dependency (avoids webrtcvad
    # build headaches on Windows). Good enough for command-style utterances.
    silence_threshold_rms: float = 300.0   # tune this against your mic (see calibration note below)
    silence_duration_s: float = 0.9        # how much continuous quiet ends the utterance
    max_recording_s: float = 12.0          # hard cap so a stuck mic can't hang forever
    min_recording_s: float = 0.3           # ignore accidental blips shorter than this

    # --- Voice response (wake word -> spoken reply) ---
    # Pause (seconds) after the spoken "Hey, how are you?" reply and before the
    # command mic stream opens. TTS is synchronous so QRUDO can never record its
    # own reply, but on a Bluetooth headset the reply's speaker audio can echo
    # into the mic for a few hundred ms; this lets that echo decay below the
    # silence RMS gate before command listening starts. Kept minimal on purpose.
    post_tts_settle_s: float = 0.5

    # --- Speech-to-text (faster-whisper) ---
    whisper_model_size: str = "base"       # tiny/base/small/medium — base is the sweet spot on CPU
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"     # int8 = fastest on CPU, small accuracy tradeoff
    # None = auto-detect language per utterance (handles Hinglish code-switching
    # reasonably well). Force "en" if you find it mis-detecting Hindi script often.
    whisper_language: str | None = None


CONFIG = VoiceConfig()