"""Which app the targeted commands land in, and how that is chosen.

The resolver is fed fake probes throughout, so nothing here asks this
machine what is actually running -- the platform probes are the one
part these tests cannot cover, deliberately.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import Command, ControlConfig, log
from control.targets import TargetResolver

# Scratch logs, not the live ones -- see test_control for why.
import tempfile

log.setup(tempfile.mkdtemp(), console=False)


def seeing(frontmost=None, running=(), playing=()):
    return lambda: {"frontmost": frontmost, "running": list(running),
                    "playing": list(playing)}


class TestAutoResolution(unittest.TestCase):
    def resolver(self, probe, target_app=""):
        config = ControlConfig(target_app=target_app, cooldown_seconds=0)
        made = TargetResolver(config, probe=probe)
        made.refresh(force=True)
        made.apply()
        return made, config

    def test_the_one_playing_app_wins(self):
        made, config = self.resolver(
            seeing(running=["Google Chrome", "Spotify"], playing=["Spotify"]))

        self.assertEqual(config.target_app, "Spotify")

    def test_the_focused_candidate_beats_a_playing_one(self):
        """You are looking at it; the gesture is for it."""

        made, config = self.resolver(
            seeing(frontmost="Google Chrome",
                   running=["Google Chrome", "Spotify"], playing=["Spotify"]))

        self.assertEqual(config.target_app, "Google Chrome")

    def test_nothing_around_falls_back_to_the_preference(self):
        made, config = self.resolver(seeing(), target_app="auto")
        self.assertEqual(config.target_app, "")

    def test_a_configured_name_pins_and_auto_never_moves_it(self):
        made, config = self.resolver(
            seeing(running=["Spotify"], playing=["Spotify"]),
            target_app="Google Chrome")

        self.assertEqual(config.target_app, "Google Chrome",
                         "a pinned target must not be second-guessed")

    def test_a_failing_probe_is_the_old_behaviour(self):
        def broken():
            raise RuntimeError("no OS today")

        made, config = self.resolver(broken, target_app="Google Chrome")

        self.assertEqual(config.target_app, "Google Chrome")


class TestCycling(unittest.TestCase):
    def test_cycles_through_auto_and_every_candidate(self):
        config = ControlConfig(cooldown_seconds=0)
        made = TargetResolver(
            config, probe=seeing(running=["Google Chrome", "Spotify"]))

        first = made.cycle(+1)
        second = made.cycle(+1)
        third = made.cycle(+1)

        self.assertIn("Google Chrome", first)
        self.assertIn("Spotify", second)
        self.assertIn("auto", third)

    def test_cycling_back_returns_the_same_way(self):
        config = ControlConfig(cooldown_seconds=0)
        made = TargetResolver(
            config, probe=seeing(running=["Google Chrome", "Spotify"]))

        made.cycle(+1)
        detail = made.cycle(-1)

        self.assertIn("auto", detail)

    def test_a_cycle_pins_until_the_next_cycle(self):
        config = ControlConfig(cooldown_seconds=0)
        made = TargetResolver(
            config,
            probe=seeing(running=["Google Chrome", "Spotify"],
                         playing=["Spotify"]))

        made.cycle(+1)   # -> the frontmost-less first candidate

        self.assertEqual(config.target_app, made.choice)
        picked = made.choice

        made.refresh(force=True)
        made.apply()

        self.assertEqual(config.target_app, picked,
                         "a refresh must not move a pinned choice")


class TestThroughTheEngine(unittest.TestCase):
    def test_the_commands_switch_and_report(self):
        from control import ControlEngine
        from control.backends.null import NullController

        config = ControlConfig(cooldown_seconds=0)
        engine = ControlEngine(controller=NullController(config),
                               config=config)
        engine.targets.probe = seeing(running=["Spotify"])

        result = engine.execute(Command.TARGET_NEXT)

        self.assertEqual(result.status, "OK")
        self.assertIn("Spotify", result.detail)
        self.assertEqual(config.target_app, "Spotify")

        back = engine.execute(Command.TARGET_PREV)

        self.assertIn("auto", back.detail)


class TestThePointingGesture(unittest.TestCase):
    def test_point_cycles_the_target_once_it_is_held(self):
        from integration.bridge import GestureRouter

        router = GestureRouter()

        self.assertIsNone(router.update("POINT", now=1000.0),
                          "a glimpse of pointing is not a switch")
        self.assertIs(router.update("POINT", now=1000.6),
                      Command.TARGET_NEXT)
        self.assertIsNone(router.update("POINT", now=1000.7),
                          "held pointing is one switch, not many")

    def test_the_way_into_the_swipe_pose_is_not_a_point(self):
        """The index leads into the two-finger pose, so its first frames
        read as an honest POINT.  Those frames must not switch anything
        -- and must not spend the cooldown the swipe itself needs a
        moment later, which is how binding POINT silently broke every
        swipe until the dwell existed.
        """

        from integration.bridge import GestureRouter

        router = GestureRouter()

        self.assertIsNone(router.update("POINT", now=1000.0))
        self.assertIsNone(router.update("POINT", now=1000.1))
        self.assertIsNone(router.update("TWO_FINGER", now=1000.2))

        self.assertIs(router.update(swipe="SWIPE_LEFT", now=1000.5),
                      Command.REWIND,
                      "the transition must not have spent the cooldown")


class TestTabTargets(unittest.TestCase):
    """Two videos in ONE browser are two targets.

    Switching apps cannot tell them apart -- that was 'switch target is
    not working' with two Chrome tabs open -- so media tabs join the
    cycle, and picking one ACTIVATES it, because keys only ever land in
    the tab a browser is showing."""

    def resolver(self):
        from control.targets import TabTarget

        tabs = [TabTarget("Google Chrome", 1, 1, "lofi - YouTube"),
                TabTarget("Google Chrome", 1, 2, "talk - YouTube")]

        probe = lambda: {"frontmost": "Google Chrome",
                         "running": ["Google Chrome"],
                         "playing": [],
                         "tabs": tabs}

        config = ControlConfig(target_app="", cooldown_seconds=0)
        self.activated = []
        made = TargetResolver(config, probe=probe,
                              activate=self.activated.append)
        made.refresh(force=True)
        return made, config, tabs

    def test_tabs_cycle_as_numbered_chromes(self):
        """Two tabs are 'Google Chrome 1' and 'Google Chrome 2' -- the
        way a person counts them -- and the bare 'Google Chrome' entry
        steps aside, since it says nothing the numbers do not."""

        made, config, tabs = self.resolver()

        said = [made.cycle(+1) for _ in range(3)]

        self.assertEqual(said[0],
                         'target -> Google Chrome 1 ("lofi - YouTube")')
        self.assertEqual(said[1],
                         'target -> Google Chrome 2 ("talk - YouTube")')
        self.assertTrue(said[2].startswith("target -> auto"))

    def test_picking_a_tab_activates_it(self):
        made, config, tabs = self.resolver()

        made.cycle(+1)          # Google Chrome 1

        self.assertEqual(self.activated, [tabs[0]])

    def test_a_tab_choice_keeps_the_browser_in_the_config(self):
        """target_app must stay an APP name -- every keystroke path
        looks the target up as a process."""

        made, config, tabs = self.resolver()

        made.cycle(+1)

        self.assertEqual(config.target_app, "Google Chrome")

    def test_a_pinned_tab_is_written_for_the_play_routing(self):
        """The media key follows whatever was ALREADY playing, so a
        pinned tab must announce itself: config.target_tab set while a
        tab is chosen, empty the moment the choice moves on."""

        made, config, tabs = self.resolver()

        made.cycle(+1)                      # Google Chrome 1
        self.assertEqual(config.target_tab, "lofi - YouTube")

        made.cycle(+1)                      # Google Chrome 2
        self.assertEqual(config.target_tab, "talk - YouTube")

        made.cycle(+1)                      # back to auto
        self.assertEqual(config.target_tab, "")

    def test_other_media_apps_come_before_the_numbered_tabs(self):
        """Apps first, then the browser's numbered tabs: Spotify ->
        Google Chrome 1 -> Google Chrome 2."""

        from control.targets import TabTarget

        tabs = [TabTarget("Google Chrome", 1, 1, "lofi - YouTube"),
                TabTarget("Google Chrome", 1, 2, "talk - YouTube")]
        probe = lambda: {"frontmost": "Spotify",
                         "running": ["Spotify", "Google Chrome"],
                         "playing": ["Spotify"],
                         "tabs": tabs}
        config = ControlConfig(target_app="", cooldown_seconds=0)
        made = TargetResolver(config, probe=probe, activate=lambda t: None)
        made.refresh(force=True)

        said = [made.cycle(+1) for _ in range(3)]

        self.assertEqual(said[0], "target -> Spotify")
        self.assertEqual(said[1],
                         'target -> Google Chrome 1 ("lofi - YouTube")')
        self.assertEqual(said[2],
                         'target -> Google Chrome 2 ("talk - YouTube")')

    def test_unmatched_tabs_stay_out_of_the_cycle(self):
        from control.targets import TabTarget

        probe = lambda: {"frontmost": None, "running": ["Google Chrome"],
                         "playing": [],
                         "tabs": [TabTarget("Google Chrome", 1, 1,
                                            "New Tab")]}
        config = ControlConfig(target_app="", cooldown_seconds=0)
        made = TargetResolver(config, probe=probe, activate=lambda t: None)
        made.refresh(force=True)

        self.assertEqual(made.tabs, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
