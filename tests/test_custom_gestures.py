"""Gestures a user teaches: representation, storage, and matching.

The isolation from the built-ins -- the thing that protects the
detection floor -- is proven separately in test_custom_isolation.py.
This file checks that a taught gesture is stored faithfully, loaded
back, and recognised at runtime once the built-ins have passed.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import vision
from vision import custom, gestures
from vision.custom import CustomError, CustomGesture

from test_gestures import CURLED, EXTENDED, HAND, make_hand


def a_gesture(name="THREE", **over):
    """A valid three-fingers-up custom pose, overridable."""

    base = dict(
        name=name,
        signature={"index": 0.95, "middle": 0.95, "ring": 0.95,
                   "pinky": 0.40},
        tolerance=0.12,
        binding_type="action",
        command="VOLUME_UP",
    )
    base.update(over)
    return CustomGesture(**base)


class TestRepresentation(unittest.TestCase):
    def test_a_valid_gesture_builds(self):
        g = a_gesture()

        self.assertEqual(g.name, "THREE")
        self.assertEqual(set(g.signature), {"index", "middle", "ring",
                                            "pinky"})

    def test_the_name_is_normalised(self):
        self.assertEqual(a_gesture(name=" three ").name, "THREE")

    def test_a_built_in_name_is_refused(self):
        for reserved in ("FIST", "open_palm", "SWIPE_LEFT"):
            with self.subTest(name=reserved):
                with self.assertRaises(CustomError):
                    a_gesture(name=reserved)

    def test_tolerance_is_clamped(self):
        self.assertGreaterEqual(a_gesture(tolerance=0.0).tolerance,
                                custom.MIN_TOLERANCE)
        self.assertLessEqual(a_gesture(tolerance=9.0).tolerance,
                             custom.MAX_TOLERANCE)

    def test_a_move_needs_a_direction(self):
        with self.assertRaises(CustomError):
            a_gesture(kind="move", direction="")

        moved = a_gesture(kind="move", direction="left")
        self.assertEqual(moved.direction, "left")

    def test_a_keystroke_needs_a_combo(self):
        with self.assertRaises(CustomError):
            a_gesture(binding_type="keystroke", combo="", command="")

        keyed = a_gesture(binding_type="keystroke", combo="cmd+shift+n",
                          command="")
        self.assertEqual(keyed.combo, "cmd+shift+n")

    def test_distance_is_zero_at_the_signature(self):
        g = a_gesture()
        self.assertAlmostEqual(g.distance(g.signature), 0.0)

    def test_distance_grows_with_difference(self):
        g = a_gesture()
        near = dict(g.signature, pinky=0.45)
        far = dict(g.signature, pinky=0.95)

        self.assertLess(g.distance(near), g.distance(far))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "qrudo_gestures.json"

    def test_round_trip(self):
        custom.save_all([a_gesture()], self.path)
        back = custom.load_all(self.path)

        self.assertEqual(len(back), 1)
        self.assertEqual(back[0].name, "THREE")
        self.assertEqual(back[0].command, "VOLUME_UP")

    def test_a_missing_file_is_an_empty_list(self):
        self.assertEqual(custom.load_all(self.path), [])

    def test_a_corrupt_entry_is_dropped_not_fatal(self):
        self.path.write_text('[{"name": "OK", "signature": '
                             '{"index":0.9,"middle":0.9,"ring":0.9,'
                             '"pinky":0.4}, "tolerance":0.1, '
                             '"command":"VOLUME_UP"}, {"broken": true}]')

        back = custom.load_all(self.path)

        self.assertEqual([g.name for g in back], ["OK"])

    def test_add_replaces_the_same_name(self):
        custom.save_all([a_gesture(command="VOLUME_UP")], self.path)
        custom.add(a_gesture(command="VOLUME_DOWN"), self.path)

        back = custom.load_all(self.path)
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0].command, "VOLUME_DOWN")

    def test_remove(self):
        custom.save_all([a_gesture("THREE"), a_gesture("FOUR")], self.path)
        custom.remove("THREE", self.path)

        self.assertEqual([g.name for g in custom.load_all(self.path)],
                         ["FOUR"])

    def test_two_of_a_name_is_refused(self):
        with self.assertRaises(CustomError):
            custom.save_all([a_gesture("X"), a_gesture("X")], self.path)


class TestRuntimeMatching(unittest.TestCase):
    """A taught shape is recognised, once the built-ins have passed.

    THREE fingers up (index, middle, ring out; pinky down) is UNKNOWN to
    the built-in classifier -- no shipped pose is three fingers -- so it
    is exactly the kind of shape a user would teach.
    """

    THREE = (EXTENDED, EXTENDED, EXTENDED, CURLED)

    def setUp(self):
        vision.reset_state()
        # A registry loaded straight from an object, no file.
        custom._active = []

    def tearDown(self):
        custom._active = []
        vision.reset_state()

    def register_three(self):
        import vision.hand_state as hs
        spans = hs.finger_span(make_hand(*self.THREE))
        custom._active = [CustomGesture(
            name="THREE", signature=spans, tolerance=0.15,
            command="VOLUME_UP")]

    def test_without_registration_three_is_unknown(self):
        settled = None
        for _ in range(6):
            settled = gestures.detect_gesture(make_hand(*self.THREE), HAND)

        self.assertEqual(settled, "UNKNOWN")

    def test_with_registration_three_is_recognised(self):
        self.register_three()

        settled = None
        for _ in range(6):
            settled = gestures.detect_gesture(make_hand(*self.THREE), HAND)

        self.assertEqual(settled, "THREE")

    def test_a_registered_custom_never_overrides_a_built_in(self):
        """Even with a custom gesture loaded, a fist is still a fist.

        The isolation, seen from the behaviour side: the built-in wins
        because custom matching only runs on the UNKNOWN branch.
        """

        self.register_three()

        settled = None
        for _ in range(6):
            settled = gestures.detect_gesture(
                make_hand(CURLED, CURLED, CURLED, CURLED), HAND)

        self.assertEqual(settled, "FIST")


if __name__ == "__main__":
    unittest.main(verbosity=2)
