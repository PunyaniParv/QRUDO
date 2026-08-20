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

    # Audio frames are 80 ms at 16 kHz (1280 samples) -- the size the wake
    # model is designed for.  One continuous monitor feeds both wake detection
    # and command capture, so every consumer sees the same frame size.
    frame_ms: int = 80
    frame_samples: int = 1280

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
    # default.  The threshold is the *consecutive/window* bar; a single frame
    # at/above wake_peak_threshold also fires (see voice/detect.py).
    wake_word_sensitivity: float = 0.5

    # A single frame at/above this raw score fires immediately -- catches a
    # real phrase whose peak lasts only one 80 ms frame (measured on a real
    # "hey jarvis" clip: peak 0.683 with just one frame >= 0.5).  To stay
    # false-positive-resistant, a peak must also have at least one frame
    # >= wake_peak_support within wake_support_window of it (the phrase has a
    # time spread; a lone noise spike does not).
    wake_peak_threshold: float = 0.65
    wake_peak_support: float = 0.3
    wake_support_window: int = 6

    # Sliding-window patience: fire when at least wake_window_min of the last
    # wake_window frames are at/above the threshold -- tolerates a one-frame
    # dip mid-phrase that would reset a purely consecutive counter.
    wake_window: int = 4
    wake_window_min: int = 2

    # Minimum gap between two detections (seconds).  A detection fires at
    # "jarvis"; the phrase tail can still score high for a moment, so a
    # cooldown stops an immediate second trigger.
    wake_cooldown_s: float = 1.0

    # --- Command capture (silence-based end-of-utterance detection) ---
    # Simple RMS energy check -- no extra native dependency (avoids webrtcvad
    # build headaches on Windows).  When adaptive_silence is on, the working
    # threshold is the larger of silence_threshold_rms and noise_gate_multiplier
    # times the measured noise floor, so a noisier room still ends the
    # utterance instead of recording the full max_recording_s.
    #
    # silence_threshold_rms is a low ABSOLUTE floor only: it keeps a
    # near-silent room's tiny dither/mic self-noise from being speech, but it
    # is NOT the dominant gate.  Measured on the real machine, a fixed 300 RMS
    # floor forced the user to shout (normal conversational speech produces
    # many frames at 100-300 RMS, all below the old gate) and made those quiet
    # speech frames feed the noise estimator, raising the gate further.  The
    # gate is now relative to the measured noise floor instead.
    silence_threshold_rms: float = 100.0
    adaptive_silence: bool = True
    # The working gate rides this many times above the measured noise floor.
    # Background noise stays rejected (it sits at ~1x the floor) while normal
    # speech (typically 3-10x the floor) crosses it.
    noise_gate_multiplier: float = 3.0
    # Robust noise floor estimation: pre-speech quiet frames adapt the floor
    # fast downward and only very slowly upward, so a quiet syllable that dips
    # below the gate cannot pollute the floor and raise the gate (the
    # mid-command-dip bug).  The floor is also seeded from the monitor's
    # session ambient estimate when one is available.
    noise_floor_init: float = 80.0
    silence_duration_s: float = 0.7        # continuous quiet ends the utterance
    min_recording_s: float = 0.3           # ignore accidental blips shorter than this
    max_recording_s: float = 12.0          # hard cap so a stuck mic can't hang forever
    max_command_s: float = 8.0             # cap on a single wake->command utterance
    pre_speech_timeout_s: float = 3.0      # how long to wait for speech after wake
    command_pre_roll_s: float = 0.3        # audio retained just before wake/command
    pre_roll_frames: int = 4               # ~0.32 s at 80 ms frames
    # Post-wake grace: when a command capture begins right after wake detection,
    # the first loud frames are the wake phrase tail, and a short silence that
    # follows is the natural pause before the command -- NOT the end of the
    # utterance.  Within this grace window a continuous loud run is treated as
    # the command itself (same-utterance "hey jarvis, increase the volume"); a
    # silence gap >= silence_duration_s inside it is treated as the wake-command
    # pause and the capture keeps listening (up to pre_speech_timeout_s).
    post_wake_grace_s: float = 0.8
    # Speech onset after the wake phrase: a pre-speech silence of at least this
    # long proves the wake phrase is over (the phrase itself is never silent),
    # so the next loud frame is the command -- the natural pause the user takes
    # before speaking is not mistaken for the end of the utterance.  This is
    # what lets "hey jarvis" [0.8-2s pause] "increase the volume" work without
    # the user having to time their command precisely.
    wake_pause_min_s: float = 0.3
    # Same-breath / intermittent onset: the command may begin right after the
    # wake tail with frames that dip below the gate between syllables.  Rather
    # than requiring a long uninterrupted loud run, speech onset is also
    # established when at least onset_min_loud of the last onset_window_frames
    # are loud AND at least one frame in that window is quiet (a real speech
    # rhythm; a solid wake-tail run never looks like that, so the tail cannot
    # be mistaken for the command and ended early on the pause that follows it).
    onset_window_frames: int = 6
    onset_min_loud: int = 3
    # How much of the monitor's recent history feeds the ambient-noise estimate
    # that seeds the capture's noise floor (frames; 120 x 80 ms ~ 9.6 s).
    ambient_floor_frames: int = 120
    ambient_floor_percentile: float = 15.0

    # --- Voice response (wake word -> acknowledgement) ---
    # By default there is NO spoken acknowledgement in the command path: a
    # conversational TTS reply ("Hey, how are you?") blocks the mic for
    # seconds and forces the user to repeat their command.  The command is
    # captured straight off the wake phrase instead.  A caller that wants a
    # short ack passes one explicitly; it is spoken on a non-blocking thread.
    wake_response: str = ""
    # Legacy pause kept for callers that still play a spoken ack; the default
    # command path never sleeps here.
    post_tts_settle_s: float = 0.5

    # --- Speech-to-text (faster-whisper) ---
    whisper_model_size: str = "base"       # tiny/base/small/medium — base is the sweet spot on CPU
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"     # int8 = fastest on CPU, small accuracy tradeoff
    # Forced to English.  Auto-detection was measured to mis-fire on accented
    # (Indian-English) speech and on quiet audio, and it adds latency and a
    # crash path (faster-whisper's `ValueError` on blank VAD).  This product is
    # an English command assistant; steer the model with initial_prompt too.
    whisper_language: str = "en"
    whisper_beam_size: int = 1             # greedy is fastest; raise if accuracy suffers
    # A decoder bias prompt is OFF by default: measured on this machine, passing
    # an initial_prompt to faster-whisper ~quadrupled decode time (2.3s -> 9.8s
    # on a short clip) and turned a clear "hey jarvis" into "H-J-R-W-S."  Leave
    # it set to a phrase to re-enable; the forced English + sustained-energy
    # gate below already stops the measured hallucinations.
    whisper_initial_prompt: str = ""
    # A captured clip quieter than this peak is treated as "nothing was said"
    # (real measurement: quiet ambient audio of peak ~0.06 makes the base model
    # hallucinate whole sentences).
    stt_min_peak: float = 0.02
    # Isolated clicks can pass the peak gate (measured: quiet-room noise had
    # peak 0.064 but only 2 active 10 ms frames).  Requiring at least this many
    # active frames means real (sustained) speech reaches the decoder, while a
    # couple of noise clicks return "".  ~50 ms of real audio.
    stt_min_frames: int = 5

    # --- Diagnostics ---
    # USER lifecycle messages print always.  Aggregate per-utterance
    # diagnostics ([wake-stats], [record-stats], [whisper-stats],
    # [voice-timing], [capture-debug], [stt-input-debug], [handoff-debug])
    # print only when debug is on.  Frame-level telemetry ([mic-frame],
    # [mic-pop], [capture-frame], per-second [wake-debug] lines, raw scores,
    # thresholds, counters) prints only when debug AND trace are both on, so
    # DEBUG mode never floods the terminal.  Set QRUDO_VOICE_DEBUG=1 or
    # SARV_VOICE_DEBUG=1 to enable debug, and QRUDO_VOICE_TRACE=1 or
    # SARV_VOICE_TRACE=1 to additionally enable trace.
    debug: bool = (
        os.getenv("QRUDO_VOICE_DEBUG") or os.getenv("SARV_VOICE_DEBUG") or ""
    ).lower() in ("1", "true", "yes", "on")
    trace: bool = (
        os.getenv("QRUDO_VOICE_TRACE") or os.getenv("SARV_VOICE_TRACE") or ""
    ).lower() in ("1", "true", "yes", "on")


CONFIG = VoiceConfig()