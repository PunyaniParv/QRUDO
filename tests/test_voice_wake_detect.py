"""Tests for the pure wake-word detection state machine (voice/detect.py).

The criteria are calibrated against real audio measured on the failing laptop
(``--record-test`` WAVs): a genuine "hey jarvis" peaks at raw score 0.683 with
only ONE frame at/above 0.5, so a 3-consecutive-frames rule missed it.  The
detector here fires on either a supported peak or a sliding-window majority.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.detect import WakeDetector


def _detector(**overrides):
    defaults = dict(
        threshold=0.5,
        window=4,
        window_min=2,
        peak_threshold=0.65,
        peak_support=0.3,
        support_window=6,
        cooldown_s=1.0,
    )
    defaults.update(overrides)
    return WakeDetector(**defaults)


def _feed(det, scores, step=1.0):
    """Feed one frame per score at increasing monotonic times; return the
    list of (frame_index, fired) results."""
    results = []
    for i, score in enumerate(scores):
        fired = det.update(score, t=i * step)
        results.append((i, fired))
    return results


class LowScoresNeverFire(unittest.TestCase):
    def test_quiet_room_scores_never_fire(self):
        # Measured ambient: raw model scores 0.000-0.002 over 13 s of quiet.
        det = _detector()
        fired = _feed(det, [0.002] * 40)
        self.assertFalse(any(f for _, f in fired))

    def test_wav1_never_fires(self):
        # Measured quiet clip: max raw score 0.039 -> must NOT detect.
        det = _detector()
        fired = _feed(det, [0.039] * 20)
        self.assertFalse(any(f for _, f in fired))


class PeakTriggerFiresOnSinglePeakWithSupport(unittest.TestCase):
    def test_real_phrase_single_peak_with_support_fires(self):
        # Measured "hey jarvis": raw peak 0.683 with a single >=0.5 frame,
        # surrounded by a 0.26-0.49 ridge.  The old consecutive rule missed it;
        # the peak trigger must fire.
        det = _detector()
        scores = [0.10, 0.26, 0.42, 0.55, 0.683, 0.48, 0.31, 0.20, 0.12]
        fired = _feed(det, scores)
        hit = [i for i, f in fired if f]
        self.assertTrue(hit, "single-peak phrase must fire")
        self.assertEqual(hit[0], 4)  # fires on the peak frame itself

    def test_isolated_peak_without_support_does_not_fire(self):
        # One lone 0.7 frame surrounded by silence: peak support needs >= 2
        # frames >= 0.3 nearby, window patience needs 2 of the last 4 >= 0.5.
        det = _detector()
        scores = [0.0] * 5 + [0.7] + [0.0] * 10
        fired = _feed(det, scores)
        self.assertFalse(any(f for _, f in fired))

    def test_peak_fires_on_the_frame_it_is_reached(self):
        det = _detector()
        scores = [0.4, 0.45, 0.7, 0.4]
        fired = _feed(det, scores)
        self.assertTrue(fired[2][1])  # third frame reaches 0.7 with support


class WindowPatienceToleratesMidPhraseDip(unittest.TestCase):
    def test_three_of_four_above_threshold_fires(self):
        # A one-frame dip that would reset a purely consecutive counter.
        det = _detector()
        scores = [0.55, 0.55, 0.1, 0.55]
        fired = _feed(det, scores)
        self.assertTrue(fired[3][1])  # fires on the 4th frame

    def test_only_two_of_four_above_threshold_does_not_fire(self):
        # With window_min=3, a 2-of-4 score spread must not fire.
        det = _detector(window_min=3)
        scores = [0.55, 0.1, 0.55, 0.1]
        fired = _feed(det, scores)
        self.assertFalse(any(f for _, f in fired))


class CooldownAndReset(unittest.TestCase):
    def test_cooldown_blocks_re_firing(self):
        det = _detector()
        fired = _feed(det, [0.9, 0.9], step=0.05)  # both inside cooldown window
        self.assertTrue(fired[1][1])
        # Immediately after firing, more highs must NOT re-fire.
        self.assertFalse(det.update(0.95, t=0.10))
        self.assertFalse(det.update(0.95, t=0.50))

    def test_after_cooldown_expires_highs_fire_again(self):
        det = _detector()
        fired = _feed(det, [0.9, 0.9], step=0.05)
        self.assertTrue(fired[1][1])
        # Past the 1.0 s cooldown: a lone high still needs a support neighbour
        # (the window was cleared on fire), so the second high fires.
        self.assertFalse(det.update(0.95, t=2.0))
        self.assertTrue(det.update(0.95, t=2.1))

    def test_reset_clears_window_between_sessions(self):
        det = _detector()
        _feed(det, [0.9, 0.9], step=0.05)  # fires at t=0.05
        det.reset()
        # Cooldown from the fired frame still guards the immediate tail.
        self.assertFalse(det.update(0.95, t=0.10))
        # After cooldown, the window is clean: a lone high needs support.
        self.assertFalse(det.update(0.95, t=2.0))
        self.assertTrue(det.update(0.95, t=2.1))

    def test_stats_reflect_diagnostics(self):
        det = _detector()
        _feed(det, [0.1, 0.9, 0.9])
        stats = det.stats()
        self.assertEqual(stats["frames"], 3)
        self.assertAlmostEqual(stats["max_score"], 0.9)
        self.assertEqual(stats["frames_above_threshold"], 2)
        self.assertEqual(stats["longest_consecutive_high"], 2)
        self.assertTrue(stats["patience_fired"])


if __name__ == "__main__":
    unittest.main(verbosity=2)