"""
Ends a command utterance once the user has gone quiet, on shared audio.

Two entry points:

* :func:`capture_command` -- the fast path.  Reads frames from the already-
  running :class:`~voice.stream.MicMonitor` (the single owner of the mic), so
  there is no gap between wake detection and command capture and a command
  started right after "jarvis" is never lost.  A small pre-roll from the
  monitor's rolling memory keeps the audio just before the current point.
* :func:`record_until_silence` -- the compatibility path: opens its own mic,
  records one utterance, closes it.  Used by the raw ``run_voice_loop`` and
  kept so nothing outside the command loop changes.

Silence is judged by RMS energy (no webrtcvad native dependency).  The
working threshold is the larger of ``silence_threshold_rms`` and ~3x the
measured noise floor, so a noisy room still ends the utterance instead of
recording the full ``max_command_s``.
"""

from __future__ import annotations

import time

import numpy as np

from voice.config import CONFIG
from voice.stream import MicMonitor


def _rms(chunk: np.ndarray) -> float:
    # chunk is int16; compute RMS in that scale
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def _working_threshold(cfg, noise_floor: float) -> float:
    """Silence threshold that tracks a room's noise floor.

    ``silence_threshold_rms`` stays the absolute floor; once the measured
    noise is loud enough that the floor would gate forever, the threshold
    rides ~3x above the noise instead.
    """
    return max(float(getattr(cfg, "silence_threshold_rms", 300.0)), noise_floor * 3.0)


def _audio_from_frames(frames: list[np.ndarray]) -> np.ndarray:
    audio_int16 = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def capture_command(
    monitor: MicMonitor,
    *,
    config=None,
    stop=None,
    stats: bool = False,
) -> np.ndarray | None:
    """Capture one command utterance from ``monitor``; mono float32 or None.

    Pre-roll: the monitor's rolling memory (audio just before the current
    point, i.e. the tail of the wake phrase) is prepended so a command that
    follows the wake word in the same breath is not clipped.  Leading silence
    after that is skipped, so a pause between wake and command adds no dead
    time to the recording.

    The utterance ends on adaptive silence (``silence_duration_s`` of quiet
    frames, once at least ``min_recording_s`` of speech was captured) or on
    ``max_command_s`` from the first speech frame.  Returns None when no real
    speech appears within ``pre_speech_timeout_s``, or when ``stop()`` turns
    true mid-capture.  ``stats=True`` additionally returns ``(audio, stats)``
    where stats holds first-speech/duration/silence/samples for the caller's
    ``[record-stats]`` block.
    """
    cfg = config if config is not None else CONFIG
    frame_s = cfg.frame_samples / cfg.sample_rate
    silence_chunks = max(1, int(round(float(cfg.silence_duration_s) / frame_s)))
    min_chunks = max(1, int(float(cfg.min_recording_s) / frame_s))
    max_chunks = int(float(cfg.max_command_s) / frame_s)
    pre_speech_timeout = float(cfg.pre_speech_timeout_s)

    frames = list(monitor.pre_roll())
    noise_floor = 120.0
    threshold = _working_threshold(cfg, noise_floor)

    t_start = time.monotonic()
    first_speech_at: float | None = None
    speech_chunks = 0
    consecutive_silent = 0
    frame_starvation = 0

    while True:
        if stop is not None and stop():
            return None
        frame = monitor.next_frame(timeout=0.25, stop=stop)
        if frame is None:
            # Not stopped: the mic vanished mid-capture.  Give it a moment,
            # then end the utterance (None if nothing had been said yet).
            frame_starvation += 1
            if frame_starvation >= 3:
                break
            continue
        frame_starvation = 0
        rms = _rms(frame)
        now = time.monotonic()

        if rms < threshold:
            # Quiet frame: adapt the noise floor and (after speech) count it.
            noise_floor = 0.9 * noise_floor + 0.1 * rms
            threshold = _working_threshold(cfg, noise_floor)
            if first_speech_at is not None:
                frames.append(frame)
                consecutive_silent += 1
            else:
                # Leading quiet: wait for speech (bounded by pre_speech_timeout).
                if now - t_start > pre_speech_timeout:
                    break
        else:
            if first_speech_at is None:
                first_speech_at = now
                speech_chunks = 1
                consecutive_silent = 0
            else:
                speech_chunks += 1
                consecutive_silent = 0
            frames.append(frame)

        if first_speech_at is not None:
            gone_quiet = consecutive_silent >= silence_chunks
            enough_speech = speech_chunks >= min_chunks
            hit_max = now - first_speech_at >= max_chunks * frame_s
            if (enough_speech and gone_quiet) or hit_max:
                break

    if first_speech_at is None:
        return None

    audio = _audio_from_frames(frames)
    if stats:
        final_silence_s = consecutive_silent * frame_s
        return audio, {
            "first_speech_after": first_speech_at - t_start,
            "duration": time.monotonic() - t_start,
            "final_silence": final_silence_s,
            "samples": sum(len(f) for f in frames),
        }
    return audio


def record_until_silence(report: bool = False) -> np.ndarray:
    """Record one utterance from the default mic (compatibility path).

    Opens its own microphone for the duration of the utterance and returns
    mono float32 audio in [-1, 1].  ``report=True`` prints a ``[record-stats]``
    block measuring the session.  Prefer :func:`capture_command` in loops that
    already own a :class:`~voice.stream.MicMonitor`.
    """
    monitor = MicMonitor()
    monitor.open()
    try:
        if report:
            result = capture_command(monitor, stats=True)
            if result is None:
                print("[record-stats]")
                print("no_speech=yes")
                return np.array([], dtype=np.float32)
            audio, stats = result
            print("[record-stats]")
            print(f"first_speech_after={stats['first_speech_after']:.2f}s")
            print(f"duration={stats['duration']:.2f}s")
            print(f"final_silence={stats['final_silence']:.2f}s")
            print(f"samples={stats['samples']}")
            return audio
        audio = capture_command(monitor)
        if audio is None:
            return np.array([], dtype=np.float32)
        return audio
    finally:
        monitor.close()
