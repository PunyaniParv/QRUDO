"""The named-job catalog: the connection, without an AI to guess it.

"Next track for YouTube Music" resolves to Shift+N because that is a
fact in a table, not a thing a model works out.  These pin the resolves
a person will rely on, and that an unknown job fails gracefully so the
form can offer to take a typed shortcut instead.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import catalog
from control.actions import validate


class TestResolve(unittest.TestCase):
    def test_next_track_for_youtube_music_is_shift_n(self):
        """The user's exact example."""

        action = catalog.resolve("Next track", "youtube_music")

        self.assertEqual(action, {"type": "keystroke", "combo": "shift+n"})

    def test_next_track_differs_by_app(self):
        self.assertEqual(catalog.resolve("Next track", "spotify")["combo"],
                         "cmd+right")
        self.assertEqual(catalog.resolve("Next track", "youtube")["combo"],
                         "shift+n")

    def test_a_builtin_job_is_app_independent(self):
        action = catalog.resolve("Play / pause", "spotify")
        self.assertEqual(action, {"type": "builtin", "command": "PLAY_PAUSE"})

    def test_an_unknown_app_falls_back_to_any(self):
        action = catalog.resolve("Next track", "some_obscure_app")
        self.assertEqual(action["combo"], "shift+n")   # the "any" key

    def test_an_unknown_job_is_none(self):
        self.assertIsNone(catalog.resolve("Teleport", "any"))

    def test_every_resolved_action_is_a_valid_action(self):
        """The catalog can only ever produce actions the executor accepts."""

        for job in catalog.job_names():
            for app in catalog.apps_for(job):
                with self.subTest(job=job, app=app):
                    action = catalog.resolve(job, app)
                    self.assertIsNotNone(action)
                    validate(action)   # raises if malformed

    def test_every_builtin_job_names_a_real_command(self):
        from control.commands import Command

        for job in catalog.job_names():
            action = catalog.resolve(job, "any")
            if action["type"] == "builtin":
                with self.subTest(job=job):
                    Command(action["command"])   # raises if not real


class TestForm(unittest.TestCase):
    def test_apps_for_a_keystroke_job_lists_its_apps(self):
        apps = catalog.apps_for("Next track")
        self.assertIn("spotify", apps)
        self.assertIn("youtube_music", apps)

    def test_apps_for_a_builtin_job_is_just_any(self):
        self.assertEqual(catalog.apps_for("Volume up"), ["any"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
