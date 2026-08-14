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

from sarv import Command
from sarv_app import POSE_COMMANDS, SWIPE_COMMANDS, GestureRouter


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
        self.assertIsNotNone(self.router.update("FIST"))
        self.router.update("UNKNOWN")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
