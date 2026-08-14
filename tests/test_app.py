"""The join between the two halves: gesture names in, commands out.

GestureRouter is deliberately the only place the vision side and the
control side meet, and it holds no camera state, so it can be tested by
handing it a sequence of gesture names.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import Command
from integration.bridge import POSE_COMMANDS, SWIPE_COMMANDS, GestureRouter


class TestSwipes(unittest.TestCase):
    def setUp(self):
        self.router = GestureRouter()

    def test_left_rewinds(self):
        self.assertIs(self.router.update(swipe="SWIPE_LEFT"), Command.REWIND)

    def test_right_forwards(self):
        self.assertIs(self.router.update(swipe="SWIPE_RIGHT"), Command.FORWARD)

    def test_swipes_repeat(self):
        """Two swipes in a row are two seeks; the cooldown lives lower down."""

        self.assertIsNotNone(self.router.update(swipe="SWIPE_LEFT"))
        self.assertIsNotNone(self.router.update(swipe="SWIPE_LEFT"))


class TestHeldPoses(unittest.TestCase):
    def setUp(self):
        self.router = GestureRouter()

    def test_fist_plays_or_pauses(self):
        self.assertIs(self.router.update("FIST"), Command.PLAY_PAUSE)

    def test_holding_a_pose_fires_once(self):
        """The vision side reports a fist on every frame it can see one.

        Without this, holding a fist for a second would toggle play/pause
        thirty times.
        """

        self.assertIsNotNone(self.router.update("FIST"))

        for _ in range(30):
            self.assertIsNone(self.router.update("FIST"))

    def test_making_the_pose_again_fires_again(self):
        """Re-arming takes a real gesture in between, not an unsure frame.

        This used to accept "UNKNOWN" as the gap, which is what made a fist
        fire twice on a machine where the picture flickers.
        """

        self.assertIsNotNone(self.router.update("FIST"))
        self.router.update("OPEN_PALM")
        self.assertIsNotNone(self.router.update("FIST"))

    def test_unmapped_poses_do_nothing(self):
        for gesture in ("OPEN_PALM", "POINT", "TWO_FINGER", "UNKNOWN"):
            with self.subTest(gesture=gesture):
                self.assertIsNone(GestureRouter().update(gesture))

    def test_swipe_pose_is_not_bound(self):
        """TWO_FINGER is how you get ready to swipe.

        Binding it would fire a command every time you raised your hand to
        swipe, before the swipe even happened.
        """

        self.assertNotIn("TWO_FINGER", POSE_COMMANDS)

    def test_hand_leaving_resets_the_pose(self):
        self.assertIsNotNone(self.router.update("FIST"))
        self.router.forget()
        self.assertIsNotNone(self.router.update("FIST"))


class TestSwipeAndPoseTogether(unittest.TestCase):
    def test_pose_after_a_swipe_has_to_be_made_again(self):
        """A swipe ends with the hand in some pose; that must not fire.

        Swiping with two fingers and then closing them is one gesture to a
        person, and it should not become two commands.
        """

        router = GestureRouter()
        router.update("TWO_FINGER")
        self.assertIsNotNone(router.update("TWO_FINGER", "SWIPE_LEFT"))
        self.assertIsNone(router.update("UNKNOWN"))

    def test_swipe_wins_over_a_pose_in_the_same_frame(self):
        router = GestureRouter()
        self.assertIs(router.update("FIST", "SWIPE_RIGHT"), Command.FORWARD)


class TestMapping(unittest.TestCase):
    def test_every_mapped_command_is_real(self):
        for name, command in {**POSE_COMMANDS, **SWIPE_COMMANDS}.items():
            with self.subTest(gesture=name):
                self.assertIsInstance(command, Command)

    def test_mapping_can_be_replaced(self):
        """The demo may want a different set; nothing else should change."""

        router = GestureRouter(poses={"OPEN_PALM": Command.VOLUME_UP},
                               swipes={})
        self.assertIs(router.update("OPEN_PALM"), Command.VOLUME_UP)
        self.assertIsNone(router.update(swipe="SWIPE_LEFT"))


class TestFlickerDoesNotDoubleFire(unittest.TestCase):
    """Reported from a slower machine: a fist played and paused at once.

    "UNKNOWN" is the vision side saying it is unsure, not a gesture.  It
    happens for a frame or two whenever the picture is poor, and treating
    it as a change re-arms the pose -- so a fist that flickers out and back
    fires twice, and the video ends up exactly where it started.
    """

    def test_flicker_through_unknown_fires_once(self):
        router = GestureRouter()
        self.assertIsNotNone(router.update("FIST"))

        for _ in range(5):
            self.assertIsNone(router.update("UNKNOWN"))
            self.assertIsNone(router.update("FIST"))

    def test_a_frame_with_no_hand_does_not_re_arm(self):
        router = GestureRouter()
        self.assertIsNotNone(router.update("FIST"))
        self.assertIsNone(router.update(None))
        self.assertIsNone(router.update("FIST"))

    def test_another_real_gesture_still_re_arms(self):
        """Fist, open the hand, fist again is two deliberate gestures."""

        router = GestureRouter()
        self.assertIsNotNone(router.update("FIST"))
        router.update("OPEN_PALM")
        self.assertIsNotNone(router.update("FIST"))

    def test_the_hand_leaving_re_arms(self):
        router = GestureRouter()
        self.assertIsNotNone(router.update("FIST"))
        router.forget()
        self.assertIsNotNone(router.update("FIST"))


class TestArmingSwitch(unittest.TestCase):
    """Holding an open hand turns SARV on and off.

    The problem it answers: talking with your hands should not pause your
    video, and no amount of threshold tuning reliably tells a gesture from
    a gesticulation.  This hands the decision over instead.
    """

    def setUp(self):
        from integration.bridge import ArmingSwitch
        self.switch = ArmingSwitch()
        self.now = 1000.0

    def hold(self, gesture, seconds, step=0.05):
        """Show a gesture for a while; returns how often it switched."""

        switches = 0
        for _ in range(int(seconds / step)):
            switches += bool(self.switch.update(gesture, self.now))
            self.now += step
        return switches

    def test_starts_listening(self):
        self.assertTrue(self.switch.armed)

    def test_holding_an_open_hand_pauses_it(self):
        self.hold("OPEN_PALM", 2.0)
        self.assertFalse(self.switch.armed)

    def test_holding_it_again_starts_it_listening(self):
        self.hold("OPEN_PALM", 2.0)
        self.hold("UNKNOWN", 0.3)
        self.hold("OPEN_PALM", 2.0)
        self.assertTrue(self.switch.armed)

    def test_a_glimpse_of_an_open_hand_does_nothing(self):
        """An open hand is what your hand passes through on the way to
        every other gesture."""

        self.hold("OPEN_PALM", 0.5)
        self.assertTrue(self.switch.armed)

    def test_holding_on_does_not_flip_it_back(self):
        """Keep holding after it switches and it must stay switched."""

        self.assertEqual(self.hold("OPEN_PALM", 6.0), 1)
        self.assertFalse(self.switch.armed)

    def test_the_hand_has_to_leave_before_it_counts_again(self):
        self.hold("OPEN_PALM", 6.0)
        self.hold("FIST", 0.2)
        self.hold("OPEN_PALM", 2.0)
        self.assertTrue(self.switch.armed)

    def test_progress_is_reported_while_holding(self):
        """Shown as a filling bar, so a held hand does not look ignored."""

        self.hold("OPEN_PALM", 0.75)
        self.assertGreater(self.switch.holding(self.now), 0.4)
        self.assertLess(self.switch.holding(self.now), 0.6)

    def test_progress_is_nothing_when_not_holding(self):
        self.assertEqual(self.switch.holding(self.now), 0.0)
        self.hold("FIST", 1.0)
        self.assertEqual(self.switch.holding(self.now), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
