"""Pure wake-word detection state machine.

Measured on real audio (``--record-test`` WAVs from the failing laptop), a
genuine "hey jarvis" scores a raw peak of 0.683 with only *one* 80 ms frame at
or above the 0.5 threshold -- the old 3-consecutive-frames rule missed it.
This module is the replacement detection criterion, kept deliberately free of
audio/hardware code so it can be unit-tested with plain score sequences and
measured against scored WAVs.

A ``WakeDetector`` is fed one raw model score per frame and returns True when
the wake phrase is considered heard.  Any of these fires it:

* **Peak trigger** -- a single frame at/above ``peak_threshold`` (0.65) that
  also has at least two frames at/above ``peak_support`` (0.3) within the last
  ``support_window`` frames.  The support requirement means a real phrase (a
  sustained ridge of 0.26-0.49 around the peak) fires, but an isolated noise
  spike does not.
* **Window patience** -- at least ``window_min`` (2) of the last ``window``
  (4) frames at/above ``threshold``.  Tolerates a one-frame dip mid-phrase
  that would reset a purely consecutive counter.

A cooldown after each detection stops the phrase tail from re-triggering, and
``reset()`` clears the rolling window between wake sessions.
"""

from __future__ import annotations

import time
from collections import deque


class WakeDetector:
    """State machine over per-frame raw model scores.  Pure and testable."""

    def __init__(
        self,
        threshold: float = 0.5,
        window: int = 4,
        window_min: int = 2,
        peak_threshold: float = 0.65,
        peak_support: float = 0.3,
        support_window: int = 6,
        cooldown_s: float = 1.0,
        debug: bool = False,
    ) -> None:
        self.threshold = threshold
        self.window = window
        self.window_min = window_min
        self.peak_threshold = peak_threshold
        self.peak_support = peak_support
        self.support_window = max(support_window, window)
        self.cooldown_s = cooldown_s
        self.debug = debug

        self._window = deque(maxlen=self.support_window)
        self._run = 0
        self._cooldown_until = 0.0

        # Diagnostic counters (reset on reset()).
        self.frames = 0
        self.max_score = 0.0
        self.frames_above = 0
        self.longest_run = 0
        self.detections = 0

    def update(self, score: float, t: float | None = None) -> bool:
        """Feed one frame's raw model score; True when the wake word fires.

        ``t`` is a monotonic clock (defaults to ``time.monotonic()``) used
        only for the cooldown.
        """
        if t is None:
            t = time.monotonic()

        self.frames += 1
        self.max_score = max(self.max_score, score)
        if score >= self.threshold:
            self.frames_above += 1
            self._run += 1
            self.longest_run = max(self.longest_run, self._run)
        else:
            self._run = 0

        if t < self._cooldown_until:
            return False

        self._window.append(score)
        fired = False

        if score >= self.peak_threshold:
            support = sum(1 for s in self._window if s >= self.peak_support)
            if support >= 2:
                fired = True

        if not fired and len(self._window) >= self.window:
            above = sum(1 for s in self._window if s >= self.threshold)
            if above >= self.window_min:
                fired = True

        if fired:
            self.detections += 1
            self._cooldown_until = t + self.cooldown_s
            self._window.clear()
            self._run = 0

        return fired

    def reset(self) -> None:
        """Clear rolling state between wake sessions (keeps diagnostics)."""
        self._window.clear()
        self._run = 0

    def stats(self) -> dict:
        """Current diagnostics: frames, peak, above-threshold count, runs."""
        return {
            "frames": self.frames,
            "max_score": self.max_score,
            "frames_above_threshold": self.frames_above,
            "longest_consecutive_high": self.longest_run,
            "window": self.window,
            "window_min": self.window_min,
            "peak_threshold": self.peak_threshold,
            "patience_fired": self.detections > 0,
        }
