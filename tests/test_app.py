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
        """Two swipes in a row are two seeks, once the cooldown has passed.

        It used to live lower down, in the detector, where the four
        directions share one.  It is here as well now, so that a gesture
        added to the table inherits it rather than depending on which
        route it happened to arrive by.
        """

        self.assertIsNotNone(self.router.update(swipe="SWIPE_LEFT",
                                               now=1000.0))
        self.assertIsNone(self.router.update(swipe="SWIPE_LEFT", now=1000.2),
                          "the second came too soon after the first")
        self.assertIsNotNone(self.router.update(swipe="SWIPE_LEFT",
                                                now=1001.0))

    def test_one_direction_holds_off_the_others(self):
        """Which is what "global" means: a hand settling after a raise
        passes through angles that read as a turn."""

        self.assertIsNotNone(self.router.update(swipe="SWIPE_UP", now=1000.0))

        for swipe in ("SWIPE_LEFT", "SWIPE_DOWN", "PALM_UP"):
            with self.subTest(then=swipe):
                self.assertIsNone(
                    self.router.update(swipe=swipe, now=1000.2))

    def test_a_pose_is_held_off_too(self):
        self.assertIsNotNone(self.router.update(swipe="SWIPE_UP", now=1000.0))

        self.assertIsNone(self.router.update("FIST", now=1000.2))

    def test_and_a_swipe_after_a_pose(self):
        self.assertIsNotNone(self.router.update("FIST", now=1000.0))

        self.assertIsNone(self.router.update(swipe="SWIPE_UP", now=1000.2))

    def test_the_hand_leaving_clears_it(self):
        """A hand that left and came back is not one movement."""

        self.assertIsNotNone(self.router.update(swipe="SWIPE_UP", now=1000.0))
        self.router.forget()

        self.assertIsNotNone(self.router.update(swipe="SWIPE_UP", now=1000.1))


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
        fire twice on a machine where the picture flickers.  A moment has
        to pass as well now, since a flicker to another real gesture and
        back does the same thing in under a second.
        """

        self.assertIsNotNone(self.router.update("FIST", now=1000.0))
        self.router.update("OPEN_PALM", now=1000.5)
        self.assertIsNotNone(self.router.update("FIST", now=1002.0))

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
        """Fist, open the hand, fist again is two deliberate gestures.

        Given a moment between them: done inside a second it is a
        misreading flickering, which is what this guards.
        """

        router = GestureRouter()
        self.assertIsNotNone(router.update("FIST", now=1000.0))
        router.update("OPEN_PALM", now=1000.5)
        self.assertIsNotNone(router.update("FIST", now=1002.0))

    def test_the_hand_leaving_re_arms(self):
        router = GestureRouter()
        self.assertIsNotNone(router.update("FIST"))
        router.forget()
        self.assertIsNotNone(router.update("FIST"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestFlickerBetweenTwoRealGestures(unittest.TestCase):
    """A misreading that is not "unknown" must not fire a pose twice.

    Reported twice, from two different flickers.  The first went through
    "unknown" and was fixed by ignoring that; this one went fist to pinch
    and back, which are both real gestures, so the pose re-armed and fired
    on the way in and again on the way back -- playing and pausing at once,
    leaving the video where it started.
    """

    def setUp(self):
        self.router = GestureRouter()
        self.now = 1000.0

    def fire(self, gesture, after=0.05):
        self.now += after
        return self.router.update(gesture, now=self.now)

    def test_a_flicker_to_another_gesture_and_back_fires_once(self):
        self.assertIsNotNone(self.fire("FIST"))
        self.fire("PINCH")
        self.assertIsNone(self.fire("FIST"))

    def test_a_deliberate_repeat_after_a_pause_still_fires(self):
        self.assertIsNotNone(self.fire("FIST"))
        self.fire("OPEN_PALM")
        self.assertIsNotNone(self.fire("FIST", after=1.5))

    def test_the_hand_leaving_counts_as_deliberate(self):
        """Fist, drop your hand, fist again is two gestures however fast."""

        self.assertIsNotNone(self.fire("FIST"))
        self.router.forget()
        self.assertIsNotNone(self.fire("FIST"))

    def test_different_poses_are_unaffected_by_each_other(self):
        """Each keeps its own repeat guard.  They still wait for the
        cooldown between them, as everything does -- two different poses
        a tenth of a second apart is a flicker, not two gestures."""

        router = GestureRouter(poses={"FIST": Command.PLAY_PAUSE,
                                      "POINT": Command.VOLUME_UP})
        self.assertIsNotNone(router.update("FIST", now=1000.0))
        self.assertIsNotNone(router.update("POINT", now=1001.0))


class TestTheOverlayIsWhole(unittest.TestCase):
    """Everything the app draws with has to be there.

    Deleting the arming indicator took the calibration prompt with it,
    because they sat next to each other -- and nothing noticed, since the
    tests never import the drawing.  The app failed to start.
    """

    def test_every_drawing_function_the_app_uses_exists(self):
        import ui

        for name in ("draw_gesture", "draw_hint", "draw_legend",
                     "draw_prompt", "draw_result", "draw_tuning"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(ui, name, None)))

    def test_the_app_and_the_setup_both_import(self):
        """Neither is covered by a test that runs them, so at least this."""

        import importlib

        for module in ("integration.runner", "integration.calibrate", "main"):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(module))
