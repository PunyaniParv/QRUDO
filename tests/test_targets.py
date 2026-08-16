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
    def test_point_cycles_the_target_once(self):
        from integration.bridge import GestureRouter

        router = GestureRouter()

        self.assertIs(router.update("POINT", now=1000.0),
                      Command.TARGET_NEXT)
        self.assertIsNone(router.update("POINT", now=1000.1),
                          "held pointing is one switch, not many")


if __name__ == "__main__":
    unittest.main(verbosity=2)
