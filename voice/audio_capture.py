"""
Records a single utterance from the microphone, stopping automatically
once the user has gone quiet.

Uses a plain RMS energy check rather than webrtcvad to avoid a native
dependency that's occasionally painful to build on Windows. This is a
deliberate tradeoff: less robust to background noise than a real VAD,
but zero extra install friction. If false cutoffs become a problem in
practice, swap this file for a webrtcvad-based version later — nothing
else in the pipeline needs to change.
"""

import time

import numpy as np

from voice.config import CONFIG
from voice.device import MicrophoneStream


def _rms(chunk: np.ndarray) -> float:
    # chunk is int16; compute RMS in that scale
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def record_until_silence(report: bool = False) -> np.ndarray:
    """
    Records from the default mic until CONFIG.silence_duration_s of
    continuous quiet is detected (or max_recording_s is hit).

    Returns mono float32 audio in [-1, 1], the format faster-whisper expects.

    ``report=True`` additionally prints a ``[record-stats]`` block measuring
    this recording session (time to first meaningful audio, total duration,
    final silence, sample count) -- pure measurement, identical recording
    behavior.
    """
    sample_rate = CONFIG.sample_rate
    chunk_ms = 30
    chunk_samples = int(sample_rate * chunk_ms / 1000)

    silence_chunks_needed = int(CONFIG.silence_duration_s * 1000 / chunk_ms)
    min_chunks_needed = int(CONFIG.min_recording_s * 1000 / chunk_ms)
    max_chunks = int(CONFIG.max_recording_s * 1000 / chunk_ms)

    frames: list[np.ndarray] = []
    consecutive_silent_chunks = 0
    t_start = time.monotonic()
    t_first_speech: float | None = None
    chunks_read = 0

    with MicrophoneStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=chunk_samples,
    ) as stream:
        for i in range(max_chunks):
            chunk, _overflowed = stream.read(chunk_samples)
            chunk = chunk.reshape(-1)
            frames.append(chunk.copy())
            chunks_read += 1

            rms = _rms(chunk)
            if rms < CONFIG.silence_threshold_rms:
                consecutive_silent_chunks += 1
            else:
                if t_first_speech is None:
                    t_first_speech = time.monotonic()
                consecutive_silent_chunks = 0

            enough_speech_captured = i >= min_chunks_needed
            gone_quiet = consecutive_silent_chunks >= silence_chunks_needed
            if enough_speech_captured and gone_quiet:
                break

    t_end = time.monotonic()
    if report:
        duration = t_end - t_start
        final_silence_s = consecutive_silent_chunks * chunk_ms / 1000.0
        first_speech_after = (
            (t_first_speech - t_start) if t_first_speech is not None else float("nan")
        )
        print("[record-stats]")
        print(f"first_speech_after={first_speech_after:.2f}s")
        print(f"duration={duration:.2f}s")
        print(f"final_silence={final_silence_s:.2f}s")
        print(f"samples={chunks_read * chunk_samples}")

    audio_int16 = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32