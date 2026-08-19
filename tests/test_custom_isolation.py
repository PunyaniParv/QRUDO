"""The guarantee: taught gestures cannot degrade the built-in ones.

This is the enforcement of the detection-floor rule for the custom
system.  It does two things a reviewer would otherwise have to trust:

1. Runs the real capability-floor accuracy grid with a full set of
   custom gestures loaded, and asserts every built-in cell still meets
   its floor.  If custom matching ever leaks into a built-in verdict,
   a cell drops and this goes red.

2. Scans vision/custom.py and asserts it never assigns to a built-in
   threshold (hand_state.* or motion.SWIPE*).  The isolation is an
   ordering rule today; this makes the "no shared threshold surface"
   half mechanical, so an later optimisation that reached for a shared
   knob fails here rather than in the field.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import vision
from vision import custom, hand_state
from vision.custom import CustomGesture

from test_capability_floor import ACCURACY_FLOOR, accuracy
from test_gestures import CURLED, EXTENDED, make_hand


def a_full_registry():
    """Several plausible taught gestures, spanning the finger space.

    Deliberately including shapes that sit *near* built-ins (a loose
    three-fingers near the open palm, a one-finger-ish near point) so
    that if isolation were broken, the floor grid would feel it.
    """

    def sig(index, middle, ring, pinky):
        return {"index": index, "middle": middle, "ring": ring,
                "pinky": pinky}

    return [
        CustomGesture(name="THREE", signature=sig(0.95, 0.95, 0.95, 0.40),
                      tolerance=0.15, command="VOLUME_UP"),
        CustomGesture(name="PINKY_OUT", signature=sig(0.40, 0.40, 0.40, 0.95),
                      tolerance=0.15, command="VOLUME_DOWN"),
        CustomGesture(name="ROCK", signature=sig(0.95, 0.40, 0.40, 0.95),
                      tolerance=0.15, binding_type="keystroke",
                      combo="cmd+shift+n"),
        CustomGesture(name="RING_DOWN",
                      signature=sig(0.95, 0.95, 0.40, 0.95),
                      tolerance=0.15, command="FORWARD"),
    ]


class TestTheFloorHoldsWithCustomsLoaded(unittest.TestCase):
    """The whole point: loading taught gestures changes no built-in cell."""

    def setUp(self):
        custom._active = a_full_registry()

    def tearDown(self):
        custom._active = []
        vision.reset_state()

    def test_every_built_in_cell_still_holds(self):
        for pose, floors in ACCURACY_FLOOR.items():
            for scale, floor in floors.items():
                with self.subTest(pose=pose, scale=scale):
                    got = accuracy(pose, scale)
                    self.assertGreaterEqual(
                        got, floor,
                        f"{pose} at {scale} fell to {got:.3f} with custom "
                        f"gestures loaded (floor {floor})")


class TestCustomTouchesNoBuiltInThreshold(unittest.TestCase):
    """A source-level guarantee, not just a behavioural one.

    The isolation is an ordering rule at one call site.  This makes the
    other half -- that custom code shares no threshold surface with the
    built-ins -- impossible to break silently: an assignment to
    hand_state.* or motion.SWIPE* inside custom.py fails the test.
    """

    GUARDED_MODULES = ("hand_state", "motion")

    def test_no_assignment_to_a_built_in_threshold(self):
        source = (ROOT / "vision" / "custom.py").read_text()
        tree = ast.parse(source)

        offenders = []

        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]

            for target in targets:
                if isinstance(target, ast.Attribute) and \
                        isinstance(target.value, ast.Name) and \
                        target.value.id in self.GUARDED_MODULES:
                    offenders.append(
                        f"{target.value.id}.{target.attr} at line "
                        f"{target.lineno}")

        self.assertEqual(
            offenders, [],
            "custom.py must never write a built-in threshold: "
            + ", ".join(offenders))

    def test_custom_does_not_import_the_classifier_internals(self):
        """It may read finger_span (a measurement) but must not reach into
        the classifier to change what it decides."""

        source = (ROOT / "vision" / "custom.py").read_text()

        self.assertNotIn("from .gestures import", source)
        self.assertNotIn("import gestures", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
