"""The application shell, tested where a headless suite can reach.

The window itself needs a display; what must not regress without one
is the settings plumbing -- typed values landing on the live config
safely -- and the promise that the shell rides the same runner loop
via its two hooks rather than owning a loop of its own.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.config import ControlConfig
from ui.app import apply_settings


class TestApplySettings(unittest.TestCase):
    def test_numbers_land_typed(self):
        config = ControlConfig()

        changed = apply_settings(config, {"volume_step": "15",
                                          "cooldown_seconds": "0.8"})

        self.assertEqual(config.volume_step, 15)
        self.assertEqual(config.cooldown_seconds, 0.8)
        self.assertEqual(sorted(changed),
                         ["cooldown_seconds", "volume_step"])

    def test_garbage_leaves_the_old_value_standing(self):
        """A save must never crash or corrupt over a typo."""

        config = ControlConfig()
        before = config.volume_step

        changed = apply_settings(config, {"volume_step": "loud"})

        self.assertEqual(config.volume_step, before)
        self.assertEqual(changed, [])

    def test_unchanged_values_do_not_report_as_changes(self):
        config = ControlConfig()

        changed = apply_settings(config,
                                 {"volume_step": str(config.volume_step)})

        self.assertEqual(changed, [])

    def test_unknown_names_are_ignored(self):
        """The page and the config may drift; drift must be harmless."""

        config = ControlConfig()

        self.assertEqual(apply_settings(config, {"warp_speed": "9"}), [])

    def test_strings_are_stripped(self):
        config = ControlConfig()

        apply_settings(config, {"target_app": "  Spotify  "})

        self.assertEqual(config.target_app, "Spotify")


class TestTheShellRidesTheRunner(unittest.TestCase):
    def test_the_runner_offers_the_two_hooks(self):
        """on_frame and should_stop are the whole contract between the
        window and the loop; losing either quietly forks the app into
        two camera loops with two behaviours."""

        from integration import runner

        parameters = inspect.signature(runner.run).parameters

        self.assertIn("on_frame", parameters)
        self.assertIn("should_stop", parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
