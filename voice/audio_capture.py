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
working threshold is the larger of ``silence_threshold_rms`` (a low absolute
floor) and ``noise_gate_multiplier`` x the measured noise floor, so a noisy
room still ends the utterance instead of recording the full ``max_command_s``.
The noise floor is estimated robustly (fast-down/slow-up EMA, seeded from the
monitor's session ambient) so quiet speech can never pollute the gate.

Speech onset is decided in three independent ways so the user never has to
shout or time their command precisely:

1. **Pause confirmed** -- after the wake phrase, a pre-speech silence of at
   least ``wake_pause_min_s`` proves the phrase is over, so the next loud
   frame is the command ("hey jarvis" [0.5-2s pause] "increase the volume").
2. **Sustained run** -- a loud run past ``post_wake_grace_s`` that cannot be
   the wake tail is the command itself (same-utterance "hey jarvis, increase
   the volume").
3. **Interleaved evidence** -- several loud frames in a short window with at
   least one quiet frame in between is speech rhythm (not a solid wake-tail
   run), so intermittent low-energy speech establishes onset quickly.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from voice.config import CONFIG
from voice.log import voice_debug, voice_trace
from voice.stream import MicMonitor


def _rms(chunk: np.ndarray) -> float:
    # chunk is int16; compute RMS in that scale
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def _working_threshold(cfg, noise_floor: float) -> float:
    """Silence threshold that tracks a room's noise floor.

    ``silence_threshold_rms`` is a low absolute safety net that rejects a
    near-silent room's dither/mic self-noise.  Once the measured noise is
    loud enough that the floor would gate forever, the threshold rides
    ``noise_gate_multiplier`` x above the noise instead.  This keeps the gate
    RELATIVE to the microphone's actual noise, so normal conversational volume
    is accepted instead of a fixed 300 RMS floor that forced shouting.
    """
    abs_floor = float(getattr(cfg, "silence_threshold_rms", 100.0))
    mult = float(getattr(cfg, "noise_gate_multiplier", 3.0))
    return max(abs_floor, noise_floor * mult)


def _audio_from_frames(frames: list[np.ndarray]) -> np.ndarray:
    audio_int16 = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def capture_command(
    monitor: MicMonitor,
    *,
    config=None,
    stop=None,
    stats: bool = False,
    debug: bool = False,
) -> np.ndarray | None:
    """Capture one command utterance from ``monitor``; mono float32 or None.

    Pre-roll: the monitor's rolling memory (audio just before the current
    point, i.e. the tail of the wake phrase) is prepended so a command that
    follows the wake word in the same breath is not clipped.  Leading silence
    after that is skipped, so a pause between wake and command adds no dead
    time to the recording.

    Wake-aware ending (when the monitor has a pre-roll, i.e. a wake phrase was
    just heard): the first loud frames are the wake phrase tail, and a silence
    gap right after them is the natural pause before the command -- NOT the end
    of the utterance.  The gap makes the capture keep listening (up to
    ``pre_speech_timeout_s``) for the command that follows, so two-stage
    "hey jarvis" -> [pause] -> "increase the volume" is captured whole.  A
    loud run that simply continues past ``post_wake_grace_s`` is the command
    itself (same-utterance "hey jarvis, increase the volume").  Without a
    pre-roll (standalone ``record_until_silence``) the original behaviour
    applies: the first loud audio is the utterance.

    The utterance ends on adaptive silence (``silence_duration_s`` of quiet
    frames, once at least ``min_recording_s`` of command speech was captured)
    or on ``max_command_s`` from the command start.  Returns None when no real
    command appears within ``pre_speech_timeout_s``, or when ``stop()`` turns
    true mid-capture.  ``stats=True`` additionally returns ``(audio, stats)``
    where stats holds first-speech/duration/silence/samples for the caller's
    ``[record-stats]`` block; ``debug=True`` prints an aggregate
    ``[capture-debug]`` block (frames, queue depth, RMS, threshold, end reason).
    """
    cfg = config if config is not None else CONFIG
    frame_s = cfg.frame_samples / cfg.sample_rate
    silence_chunks = max(1, int(round(float(cfg.silence_duration_s) / frame_s)))
    min_chunks = max(1, int(float(cfg.min_recording_s) / frame_s))
    max_chunks = int(float(cfg.max_command_s) / frame_s)
    pre_speech_timeout = float(cfg.pre_speech_timeout_s)
    grace_s = float(getattr(cfg, "post_wake_grace_s", 0.8))
    grace_chunks = max(1, int(round(grace_s / frame_s)))
    wake_pause_s = float(getattr(cfg, "wake_pause_min_s", 0.3))
    wake_pause_chunks = max(1, int(round(wake_pause_s / frame_s)))
    onset_window = int(getattr(cfg, "onset_window_frames", 6))
    onset_min_loud = int(getattr(cfg, "onset_min_loud", 3))
    noise_init = float(getattr(cfg, "noise_floor_init", 80.0))

    frames = list(monitor.pre_roll())
    pre_roll_count = len(frames)
    wake_context = pre_roll_count > 0

    # Seed the noise floor from the monitor's session ambient estimate when it
    # has one; otherwise fall back to the config default.  Same-breath commands
    # start with no quiet frames of their own, so this seed is what makes the
    # gate relative to the room instead of an arbitrary high constant.
    ambient = float(getattr(monitor, "ambient_floor", lambda: 0.0)())
    noise_floor = ambient if ambient > 0.0 else noise_init
    threshold = _working_threshold(cfg, noise_floor)

    t_start = time.monotonic()
    first_loud_at: float | None = None
    speech_chunks = 0
    consecutive_silent = 0
    frame_starvation = 0
    consecutive_loud = 0
    quiet_after_loud = 0
    pre_speech_quiet = 0
    gap_confirmed = False
    command_started = False
    command_at: float | None = None
    ended_reason = "unknown"
    loud_frames = 0
    quiet_frames = 0
    frames_seen = 0
    recent_loud: deque[int] = deque(maxlen=onset_window)

    while True:
        if stop is not None and stop():
            return None
        frame = monitor.next_frame(timeout=0.25, stop=stop, consumer="capture")
        if frame is None:
            # Not stopped: the mic vanished mid-capture.  Give it a moment,
            # then end the utterance (None if nothing had been said yet).
            frame_starvation += 1
            if frame_starvation >= 3:
                ended_reason = "mic_ended"
                break
            continue
        frame_starvation = 0
        frames_seen += 1
        rms = _rms(frame)
        now = time.monotonic()
        loud = rms >= threshold
        recent_loud.append(1 if loud else 0)

        if not loud:
            # Quiet frame: adapt the noise floor ONLY during the pre-speech
            # silence (before the first loud frame).  A dip inside the
            # utterance -- a quiet syllable between words -- is NOT room
            # noise; feeding it into the estimator raises the gate above the
            # command tail and real speech is dropped.  Measured on the real
            # machine: a 164.9 RMS mid-command dip pushed the gate from 300
            # to ~311, so a 302.7 RMS command-tail frame was classified quiet
            # and the entire burst was discarded as "no speech after wake".
            #
            # The estimator is asymmetric: it tracks a rise in genuine room
            # noise only very slowly, so an occasional quiet speech frame
            # below the gate cannot pollute the floor and push the gate up
            # (the "must shout" failure).  Once speech begins it freezes.
            if first_loud_at is None:
                if rms <= noise_floor:
                    noise_floor = 0.8 * noise_floor + 0.2 * rms
                else:
                    noise_floor = 0.98 * noise_floor + 0.02 * rms
                noise_floor = max(noise_floor, 1.0)
                threshold = _working_threshold(cfg, noise_floor)
            quiet_frames += 1
            consecutive_loud = 0
        else:
            loud_frames += 1
            consecutive_loud += 1

        if debug:
            seq = getattr(monitor, "last_seq", lambda: 0)()
            voice_trace(
                f"[capture-frame] seq={seq} rms={rms:.1f} "
                f"loud={int(loud)} threshold={threshold:.1f} "
                f"noise_floor={noise_floor:.1f} speech_chunks={speech_chunks} "
                f"consecutive_loud={consecutive_loud} "
                f"pre_speech_quiet={pre_speech_quiet} "
                f"quiet_after_loud={quiet_after_loud} "
                f"gap_confirmed={int(gap_confirmed)} "
                f"command_started={int(command_started)}",
                enabled=debug,
            )

        if not wake_context:
            # ---- Standalone: first loud audio IS the utterance. ----------
            if loud:
                if first_loud_at is None:
                    first_loud_at = now
                    speech_chunks = 1
                    consecutive_silent = 0
                else:
                    speech_chunks += 1
                    consecutive_silent = 0
                frames.append(frame)
            else:
                if first_loud_at is not None:
                    frames.append(frame)
                    consecutive_silent += 1
                elif now - t_start > pre_speech_timeout:
                    ended_reason = "pre_speech_timeout"
                    break
            if first_loud_at is not None:
                gone_quiet = consecutive_silent >= silence_chunks
                enough_speech = speech_chunks >= min_chunks
                hit_max = now - first_loud_at >= max_chunks * frame_s
                if (enough_speech and gone_quiet) or hit_max:
                    ended_reason = "max_command" if hit_max else "silence_after_command"
                    break
        else:
            # ---- Post-wake: the first loud audio is the wake tail. -------
            if command_started:
                # The command is being captured: keep everything, and end on
                # the silence that follows it.
                if loud:
                    speech_chunks += 1
                    consecutive_silent = 0
                else:
                    consecutive_silent += 1
                frames.append(frame)
                gone_quiet = consecutive_silent >= silence_chunks
                enough_speech = speech_chunks >= min_chunks
                hit_max = now - command_at >= max_chunks * frame_s
                too_long_quiet = consecutive_silent >= 2 * silence_chunks
                if (enough_speech and gone_quiet) or hit_max or too_long_quiet:
                    ended_reason = (
                        "max_command" if hit_max else "silence_after_command"
                    )
                    break
            else:
                if loud:
                    if first_loud_at is None:
                        first_loud_at = now
                    loud_in_window = sum(recent_loud)
                    # Three independent speech-onset paths:
                    #  1. Pause confirmed: the user paused >= wake_pause_min_s
                    #     after the wake phrase, so this loud frame is the
                    #     command ("hey jarvis" [pause] "increase the volume").
                    #  2. Sustained run: consecutive loud past the grace window
                    #     cannot be the wake tail, so it is the command.
                    #  3. Interleaved evidence: several loud frames in a short
                    #     window with at least one quiet frame between them is
                    #     speech rhythm, not a solid wake-tail run -- catches
                    #     intermittent low-energy speech without a long run.
                    onset_by_pause = (
                        gap_confirmed or pre_speech_quiet >= wake_pause_chunks
                    )
                    onset_by_run = consecutive_loud >= grace_chunks
                    onset_by_evidence = (
                        loud_in_window >= onset_min_loud
                        and loud_in_window < len(recent_loud)
                    )
                    if onset_by_pause or onset_by_run or onset_by_evidence:
                        # Speech after the wake/command pause, a loud run long
                        # enough to be the command itself, or interleaved
                        # speech evidence.  The command has begun.
                        command_started = True
                        command_at = now
                        speech_chunks += 1
                        consecutive_silent = 0
                    else:
                        # Still the wake tail run (or too little evidence yet).
                        speech_chunks += 1
                        consecutive_silent = 0
                        quiet_after_loud = 0
                    frames.append(frame)
                else:
                    consecutive_loud = 0
                    if speech_chunks > 0:
                        quiet_after_loud += 1
                        if quiet_after_loud >= silence_chunks:
                            # The wake phrase ended: this quiet is the pause.
                            # Keep listening for the command that follows.
                            gap_confirmed = True
                            speech_chunks = 0
                            quiet_after_loud = 0
                    else:
                        # Pre-speech silence before any loud frame: count it
                        # so a natural pause after the wake phrase confirms the
                        # phrase is over and the next loud frame is the command.
                        pre_speech_quiet += 1
                    if now - t_start > pre_speech_timeout:
                        ended_reason = "pre_speech_timeout"
                        break

    if not (command_started if wake_context else (first_loud_at is not None)):
        if debug:
            _print_capture_debug(
                frames_seen, frames, pre_roll_count,
                pending=-1, rms=0.0, noise_floor=noise_floor,
                threshold=threshold, loud_frames=loud_frames,
                quiet_frames=quiet_frames, gap_confirmed=gap_confirmed,
                grace_s=grace_s, command_started=False,
                first_loud_at=first_loud_at, ended_reason=ended_reason,
                t_start=t_start, consecutive_loud=consecutive_loud,
                pre_speech_quiet=pre_speech_quiet, enabled=debug,
            )
        return None

    audio = _audio_from_frames(frames)
    if debug:
        _print_capture_debug(
            frames_seen, frames, pre_roll_count,
            pending=monitor.pending_count(), rms=rms,
            noise_floor=noise_floor, threshold=threshold,
            loud_frames=loud_frames, quiet_frames=quiet_frames,
            gap_confirmed=gap_confirmed, grace_s=grace_s,
            command_started=command_started, first_loud_at=first_loud_at,
            ended_reason=ended_reason, t_start=t_start,
            consecutive_loud=consecutive_loud,
            pre_speech_quiet=pre_speech_quiet, enabled=debug,
        )
        _print_capture_audio_debug(audio, len(frames), frame_s, enabled=debug)
    if stats:
        final_silence_s = consecutive_silent * frame_s
        return audio, {
            "first_speech_after": (
                (first_loud_at - t_start)
                if first_loud_at is not None
                else ((command_at - t_start) if command_at is not None else 0.0)
            ),
            "command_after": (
                (command_at - t_start) if command_at is not None else 0.0
            ),
            "duration": time.monotonic() - t_start,
            "final_silence": final_silence_s,
            "samples": sum(len(f) for f in frames),
            "gap_confirmed": gap_confirmed,
            "ended_reason": ended_reason,
        }
    return audio


def _print_capture_debug(
    frames_seen: int,
    frames: list[np.ndarray],
    pre_roll_count: int,
    *,
    pending: int,
    rms: float,
    noise_floor: float,
    threshold: float,
    loud_frames: int,
    quiet_frames: int,
    gap_confirmed: bool,
    grace_s: float,
    command_started: bool,
    first_loud_at,
    ended_reason: str,
    t_start: float,
    consecutive_loud: int = 0,
    pre_speech_quiet: int = 0,
    enabled: bool = False,
) -> None:
    """Aggregate one-line capture telemetry (gated by ``debug=True``)."""
    voice_debug("[capture-debug]", enabled=enabled)
    voice_debug(f"frames_seen={frames_seen}", enabled=enabled)
    voice_debug(f"frames_appended={len(frames)}", enabled=enabled)
    voice_debug(f"pre_roll_frames={pre_roll_count}", enabled=enabled)
    voice_debug(f"pending_at_start={pending}", enabled=enabled)
    voice_debug(f"rms_last={rms:.1f}", enabled=enabled)
    voice_debug(f"noise_floor={noise_floor:.1f}", enabled=enabled)
    voice_debug(f"threshold={threshold:.1f}", enabled=enabled)
    voice_debug(f"loud_frames={loud_frames}", enabled=enabled)
    voice_debug(f"quiet_frames={quiet_frames}", enabled=enabled)
    voice_debug(f"gap_confirmed={int(gap_confirmed)}", enabled=enabled)
    voice_debug(f"post_wake_grace_s={grace_s}", enabled=enabled)
    voice_debug(f"command_started={int(command_started)}", enabled=enabled)
    voice_debug(f"consecutive_loud={consecutive_loud}", enabled=enabled)
    voice_debug(f"pre_speech_quiet={pre_speech_quiet}", enabled=enabled)
    voice_debug(
        f"first_loud_after={0.0 if first_loud_at is None else (first_loud_at - t_start):.2f}s",
        enabled=enabled,
    )
    voice_debug(f"ended_reason={ended_reason}", enabled=enabled)


def _print_capture_audio_debug(
    audio: np.ndarray,
    frame_count: int,
    frame_s: float,
    enabled: bool = False,
) -> None:
    """Inspect the exact float32 buffer capture_command is returning."""
    samples = int(audio.size)
    voice_debug("[capture-audio-debug]", enabled=enabled)
    voice_debug(f"frame_count={frame_count}", enabled=enabled)
    voice_debug(
        f"samples_per_frame={int(frame_s * 16000.0):d}", enabled=enabled
    )
    voice_debug(f"total_samples={samples}", enabled=enabled)
    voice_debug(f"dtype={audio.dtype}", enabled=enabled)
    voice_debug(f"duration={samples / 16000.0:.3f}s", enabled=enabled)
    if samples:
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        voice_debug(f"rms={rms:.4f}", enabled=enabled)
    else:
        voice_debug("rms=0.0", enabled=enabled)


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
                voice_debug("[record-stats]", enabled=report)
                voice_debug("no_speech=yes", enabled=report)
                return np.array([], dtype=np.float32)
            audio, stats = result
            voice_debug("[record-stats]", enabled=report)
            voice_debug(
                f"first_speech_after={stats['first_speech_after']:.2f}s",
                enabled=report,
            )
            voice_debug(f"duration={stats['duration']:.2f}s", enabled=report)
            voice_debug(
                f"final_silence={stats['final_silence']:.2f}s", enabled=report
            )
            voice_debug(f"samples={stats['samples']}", enabled=report)
            return audio
        audio = capture_command(monitor)
        if audio is None:
            return np.array([], dtype=np.float32)
        return audio
    finally:
        monitor.close()


