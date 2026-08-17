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
    sample_rate: int = 16000          # Porcupine and Whisper both want 16kHz mono
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

    # --- Wake word (Picovoice Porcupine) ---
    # Get a free AccessKey at https://console.picovoice.ai
    # Set it as an environment variable rather than hardcoding it:
    #   Windows (PowerShell):  setx PICOVOICE_ACCESS_KEY "your-key-here"
    picovoice_access_key: str = os.getenv("PICOVOICE_ACCESS_KEY", "")

    # Path to the custom "Hey Qrudo" .ppn file you download from the
    # Picovoice console (choose Windows as the target platform).
    # Place it in this project and point to it here, e.g.:
    #   voice/models/hey-qrudo_en_windows.ppn
    wake_word_path: str = os.getenv(
        "SARV_WAKE_WORD_PATH",
        os.path.join(os.path.dirname(__file__), "models", "hey-qrudo_en_windows.ppn"),
    )
    wake_word_sensitivity: float = 0.6  # 0.0 (fewer false positives) to 1.0 (more sensitive)

    # --- Silence-based end-of-utterance detection ---
    # Simple RMS energy check — no extra native dependency (avoids webrtcvad
    # build headaches on Windows). Good enough for command-style utterances.
    silence_threshold_rms: float = 300.0   # tune this against your mic (see calibration note below)
    silence_duration_s: float = 0.9        # how much continuous quiet ends the utterance
    max_recording_s: float = 12.0          # hard cap so a stuck mic can't hang forever
    min_recording_s: float = 0.3           # ignore accidental blips shorter than this

    # --- Speech-to-text (faster-whisper) ---
    whisper_model_size: str = "base"       # tiny/base/small/medium — base is the sweet spot on CPU
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"     # int8 = fastest on CPU, small accuracy tradeoff
    # None = auto-detect language per utterance (handles Hinglish code-switching
    # reasonably well). Force "en" if you find it mis-detecting Hindi script often.
    whisper_language: str | None = None


CONFIG = VoiceConfig()