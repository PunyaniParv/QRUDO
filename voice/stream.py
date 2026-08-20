"""One continuous microphone monitor that owns every audio frame.

The voice loop needs wake detection and command capture to see the *same*
uninterrupted audio: if each stage opens its own stream there is a gap the
size of the wake phrase tail, dropped frames while a stream re-opens, and no
way to keep the command the user started saying right after "jarvis".  This
module fixes that by giving the loop a single owner of the microphone.

``MicMonitor`` wraps one callback ``sounddevice.InputStream`` that runs for
the whole voice session.  PortAudio's callback thread only ever copies the
incoming 80 ms frame onto two thread-safe structures:

* ``pending`` -- frames no consumer has read yet.  The active consumer (wake
  detector, then command capture) drains these with ``next_frame()``.
* ``recent``  -- a never-consumed rolling window (the pre-roll memory) so the
  command capture can prepend the audio that arrived just before the current
  moment and nothing said between "hey" and "jarvis" is clipped.

Because the callback never blocks on the consumer, the loop is free to take
as long as it needs (STT, routing, execution) without ever missing a frame
that arrives meanwhile -- frames that pile up in ``pending`` are fed to the
next wake session, so a wake phrase said during STT is still heard.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from voice.config import CONFIG
from voice.device import _default_input, _resolve
from voice.log import voice_trace


def _rms_int16(chunk: np.ndarray) -> float:
    """RMS of an int16 1-D chunk, in the int16 scale (0-32768)."""
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


class MicMonitor:
    """A continuously-running microphone feeding 80 ms frames to consumers.

    One instance is shared by the whole voice loop.  ``next_frame()`` hands
    the next unread frame to the active consumer; ``pre_roll()`` returns the
    rolling memory for the current moment.  ``close()`` releases the mic and
    wakes any waiting reader, so shutdown never hangs on a blocking read.
    """

    def __init__(
        self,
        *,
        samplerate: int | None = None,
        frame_samples: int | None = None,
        pre_roll_frames: int | None = None,
        max_pending_frames: int = 120,   # ~9.6 s of 80 ms frames
    ) -> None:
        self.samplerate = samplerate if samplerate is not None else CONFIG.sample_rate
        self.frame_samples = (
            frame_samples if frame_samples is not None else CONFIG.frame_samples
        )
        self.pre_roll_frames = (
            pre_roll_frames if pre_roll_frames is not None else CONFIG.pre_roll_frames
)
        self.device_spec: int | None = None
        self.device_name: str = ""
        self._pending: deque[np.ndarray] = deque(maxlen=max_pending_frames)
        self._recent: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self._rms_history: deque[float] = deque(
            maxlen=int(getattr(CONFIG, "ambient_floor_frames", 120))
        )
        self._cv = threading.Condition()
        self._stream = None
        self._closed = False
        self._seq = 0

    # -- lifecycle --------------------------------------------------------
    def open(self) -> None:
        """Open the microphone.  Raises on failure (no usable mic)."""
        import sounddevice as sd

        from voice.device import _log_device

        configured = CONFIG.microphone_device
        spec, name = _resolve(configured)
        try:
            self._open(sd, spec)
        except Exception as exc:
            spec, name = _default_input()
            if spec is None:
                raise RuntimeError(f"No usable microphone found: {exc}") from exc
            self._open(sd, spec)
        self.device_spec, self.device_name = spec, name
        _log_device(
            _device_info(sd, spec),
            reason="voice monitor",
            channels=1,
            samplerate=self.samplerate,
        )

    def _open(self, sd, spec) -> None:
        self._stream = sd.InputStream(
            device=spec,
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        """Release the microphone and wake any blocked reader.  Idempotent."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

# -- callback side (runs on PortAudio's thread) -----------------------
    def _callback(self, indata, frames, time_info, status):
        try:
            # sounddevice hands us a view into PortAudio's reusable input
            # buffer -- the SAME memory every callback.  It is overwritten on
            # the next block, so any frame we keep must be a copy; otherwise a
            # consumer that concatenates queued frames later reads one buffer's
            # final contents for every frame (captured speech becomes silence).
            frame = np.asarray(indata, dtype=np.int16).reshape(-1).copy()
            rms = _rms_int16(frame)
            with self._cv:
                self._seq += 1
                self._rms_history.append(rms)
                voice_trace(
                    f"[mic-frame] seq={self._seq} "
                    f"rms={rms:.1f}"
                )
                self._pending.append(frame)
                self._recent.append(frame)
                self._cv.notify_all()
        except Exception:
            pass  # never let a callback error take the audio thread down

    # -- consumer side ----------------------------------------------------
    def next_frame(self, timeout: float = 0.25, stop=None, consumer: str = ""):
        """Return the next unread frame, or None on timeout/stop/close.

        ``stop`` (a zero-arg callable) is polled between waits, so a consumer
        honors a shutdown signal within about ``timeout`` seconds instead of
        blocking forever on the mic.
        """
        deadline = time.monotonic() + timeout
        with self._cv:
            while not self._pending:
                if self._closed:
                    return None
                if stop is not None and stop():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            frame = self._pending.popleft()
            voice_trace(
                f"[mic-pop] seq={self._seq - len(self._pending)} "
                f"rms={_rms_int16(frame):.1f} consumer={consumer}"
            )
            return frame

    def last_seq(self) -> int:
        """The sequence number of the most recently consumed frame (0 if none).

        Debug/telemetry only: lets the command capture report the exact
        monitor frame sequence numbers it is consuming, so a trace can be
        matched 1:1 against ``[mic-frame]``/``[mic-pop]``.
        """
        with self._cv:
            return max(0, self._seq - len(self._pending))

    def pre_roll(self) -> list[np.ndarray]:
        """The last ``pre_roll_frames`` frames before the current moment."""
        with self._cv:
            return list(self._recent)

    def ambient_floor(self) -> float:
        """Robust estimate of the room's ambient noise RMS.

        The low percentile of the RMS history is used (not the mean), so the
        loud wake phrase and brief speech that run through the monitor barely
        move it -- the capture layer seeds its noise floor from this, which is
        what makes same-breath "hey jarvis, increase the volume" work even
        though capture starts with no quiet frames of its own to estimate from.
        Returns 0.0 until enough history has been seen.
        """
        with self._cv:
            hist = list(self._rms_history)
        if len(hist) < 10:
            return 0.0
        pct = float(getattr(CONFIG, "ambient_floor_percentile", 15.0))
        return float(np.percentile(hist, pct))

    def pending_count(self) -> int:
        """How many unread frames are queued (debug/telemetry)."""
        with self._cv:
            return len(self._pending)

    def drain(self) -> None:
        """Discard all unread frames (e.g. after an audible ack echoes)."""
        with self._cv:
            self._pending.clear()
            self._recent.clear()

    def frame_ms(self) -> float:
        return self.frame_samples / self.samplerate * 1000.0


def _device_info(sd, spec):
    try:
        return sd.query_devices(spec)
    except Exception:
        return None

