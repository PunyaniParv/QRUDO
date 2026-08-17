"""The scanner: where the tracker looks, so far hands can be found at all.

The palm detector sees its input at about two hundred pixels across,
so a hand at two and a half metres -- forty pixels of a 1600-wide
frame -- is a handful of pixels to it, and it never fires.  The
scanner alternates the full frame with enlarged views while searching
and follows a found hand through a window around it.  These tests pin
the policy and the coordinate unwrapping; MediaPipe is not involved,
which is what lets them run without a camera or the model.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.hand_tracker import Scanner, unwrap


class StubPoint:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class TestTheSweep(unittest.TestCase):
    def test_every_other_look_is_the_whole_frame(self):
        """A near hand must never wait more than one frame to be found.

        Whatever the sweep does for range, the price it may not charge
        is close-up latency -- so the full frame is every second view.
        """

        scanner = Scanner()

        for i in range(20):
            view = scanner.view()
            if i % 2 == 0:
                self.assertEqual(view, Scanner.FULL)

    def test_the_zoomed_views_cover_the_frame(self):
        """A far hand can be anywhere, so somewhere has to zoom on it.

        Every point of the frame must fall inside at least one view
        that is smaller than the whole -- a seam with no enlarged
        coverage is a chair the app cannot be used from.
        """

        zoomed = [view for view in Scanner.SWEEP if view != Scanner.FULL]

        for gx in range(11):
            for gy in range(11):
                x, y = gx / 10, gy / 10
                inside = any(x0 <= x <= x0 + w and y0 <= y <= y0 + h
                             for x0, y0, w, h in zoomed)
                self.assertTrue(inside, f"({x}, {y}) is in no zoomed view")

    def test_one_look_per_frame(self):
        """view() hands out exactly one box per call: the sweep must
        never make a frame cost two detector passes."""

        scanner = Scanner()
        first, second = scanner.view(), scanner.view()

        self.assertNotEqual(first, second)


class TestFollowing(unittest.TestCase):
    def test_a_found_hand_gets_a_window_around_it(self):
        scanner = Scanner()
        scanner.found(0.5, 0.5, span=0.06)

        x0, y0, w, h = scanner.view()

        self.assertLess(w, 1.0, "the window must zoom")
        self.assertAlmostEqual(x0 + w / 2, 0.5, places=2)
        self.assertAlmostEqual(y0 + h / 2, 0.5, places=2)

    def test_the_window_never_starves_the_detector(self):
        """A tiny far hand still gets a workable window, not a keyhole."""

        scanner = Scanner()
        scanner.found(0.5, 0.5, span=0.01)

        _, _, w, _ = scanner.view()

        self.assertGreaterEqual(w, Scanner.WINDOW_FLOOR)

    def test_a_near_hand_gets_most_of_the_frame(self):
        scanner = Scanner()
        scanner.found(0.5, 0.5, span=0.3)

        _, _, w, _ = scanner.view()

        self.assertEqual(w, 1.0)

    def test_the_window_stays_inside_the_frame(self):
        """A hand at the edge is followed through a window that clamps
        rather than spills."""

        scanner = Scanner()
        scanner.found(0.02, 0.98, span=0.06)

        x0, y0, w, h = scanner.view()

        self.assertGreaterEqual(x0, 0.0)
        self.assertLessEqual(y0 + h, 1.0)

    def test_misses_widen_the_window_before_giving_up(self):
        """The blur of a fast gesture is when the hand is hardest to
        find, and hardest to find is not gone."""

        scanner = Scanner()
        scanner.found(0.5, 0.5, span=0.06)
        before = scanner.view()[2]

        scanner.missed()

        self.assertGreater(scanner.view()[2], before)

    def test_enough_misses_return_to_the_sweep(self):
        scanner = Scanner()
        scanner.found(0.5, 0.5, span=0.06)

        for _ in range(Scanner.MISSES_TO_SWEEP):
            scanner.missed()

        self.assertEqual(scanner.view(), Scanner.FULL)

    def test_a_miss_with_no_window_is_quiet(self):
        """The sweep misses constantly by nature; only the window cares."""

        scanner = Scanner()
        scanner.missed()

        self.assertEqual(scanner.view(), Scanner.FULL)


class TestUnwrap(unittest.TestCase):
    def test_the_full_frame_is_untouched(self):
        point = unwrap([StubPoint(0.3, 0.7, 0.1)], (0.0, 0.0, 1.0, 1.0))[0]

        self.assertAlmostEqual(point.x, 0.3)
        self.assertAlmostEqual(point.y, 0.7)
        self.assertAlmostEqual(point.z, 0.1)

    def test_a_crop_lands_where_the_crop_was(self):
        """The centre of the centre crop is the centre of the frame."""

        point = unwrap([StubPoint(0.5, 0.5)], (0.25, 0.25, 0.5, 0.5))[0]

        self.assertAlmostEqual(point.x, 0.5)
        self.assertAlmostEqual(point.y, 0.5)

    def test_a_corner_crop_offsets(self):
        point = unwrap([StubPoint(0.0, 1.0)], (0.4, 0.4, 0.6, 0.6))[0]

        self.assertAlmostEqual(point.x, 0.4)
        self.assertAlmostEqual(point.y, 1.0)

    def test_depth_scales_with_the_crop(self):
        """z is in units of the image the model saw, like x -- so it
        shrinks by the crop's width, or a followed hand would seem to
        deepen as the window tightened."""

        point = unwrap([StubPoint(0.5, 0.5, 0.2)], (0.25, 0.25, 0.5, 0.5))[0]

        self.assertAlmostEqual(point.z, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
