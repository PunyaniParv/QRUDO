"""Which app a play/pause reaches on macOS.

Twice now, pausing has opened Apple Music instead.  The media key looks
like the obvious way to play or pause anything, and it is a message to the
system rather than to a player: with nothing currently playing, macOS
answers it by opening Music.  So it is not used, and these pin that.

The real backend cannot be built here without a Mac's frameworks, so the
parts that touch them are replaced and the routing is what is tested.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import ControlConfig, log
from control.backends.macos import MacOSController
from control.executor import UnsupportedCommand

log.setup(console=False)


class FakeMac(MacOSController):
    """The routing, with everything that needs macOS replaced."""

    def __init__(self, running=(), target="", play_key="k"):
        self.config = ControlConfig(target_app=target,
                                    browser_play_key=play_key)
        self.log = log.get_logger("test")
        self.running = set(running)
        self.told = []          # apps spoken to by AppleScript
        self.keyed = []         # apps sent a keystroke
        self.media_keys = []    # system-wide media keys, which must stay empty
        self._quartz = object()
        self._workspace = None

    def _running_apps(self):
        return self.running

    def _osascript(self, script):
        self.told.append(script)
        return ""

    def _target_pid(self, name=None):
        name = name if name is not None else self.config.app
        return 4242 if name in self.running else None

    def _post_key(self, key_code, *, to_pid=None):
        self.keyed.append(to_pid)

    def _post_media_key(self, key):
        self.media_keys.append(key)


class TestNothingPlaying(unittest.TestCase):
    def test_with_no_player_open_it_refuses(self):
        """Rather than pressing a key macOS answers by opening Music."""

        controller = FakeMac(running={"Finder", "Terminal"})

        with self.assertRaises(UnsupportedCommand):
            controller.play_pause()

        self.assertEqual(controller.media_keys, [])

    def test_a_browser_open_but_idle_does_not_get_the_media_key(self):
        """The case that reopened Music: Chrome was running, not playing.

        A running browser is not a playing one, and the media key goes
        wherever the system thinks the music is -- which is Music.
        """

        controller = FakeMac(running={"Google Chrome"})
        controller.play_pause()

        self.assertEqual(controller.media_keys, [])
        self.assertEqual(controller.keyed, [4242])


class TestWhichKeyTheBrowserGets(unittest.TestCase):
    """Which key plays and pauses depends on the site, not the browser.

    Reported: a fist started skipping tracks instead of pausing.  It was
    sending k, which is YouTube's, to whatever was playing -- and on
    another player that key means something else.
    """

    def test_the_default_is_youtubes(self):
        controller = FakeMac(running={"Google Chrome"})

        self.assertIn("(k)", controller.play_pause())

    def test_the_spacebar_for_players_that_use_it(self):
        controller = FakeMac(running={"Google Chrome"}, play_key="space")

        self.assertIn("(space)", controller.play_pause())
        self.assertEqual(controller.media_keys, [])

    def test_the_media_key_for_anyone_who_wants_it(self):
        """Works wherever something really is playing, and opens Music if
        nothing is -- so it is asked for rather than assumed."""

        controller = FakeMac(running={"Google Chrome"}, play_key="media")
        controller.play_pause()

        self.assertEqual(len(controller.media_keys), 1)

    def test_a_scriptable_player_ignores_the_setting(self):
        """Spotify is told in words; no key is involved."""

        controller = FakeMac(running={"Spotify"}, play_key="space")
        controller.play_pause()

        self.assertEqual(controller.keyed, [])
        self.assertEqual(controller.media_keys, [])


class TestWhichAppItReaches(unittest.TestCase):
    def test_a_scriptable_player_is_told_directly(self):
        controller = FakeMac(running={"Spotify", "Google Chrome"})
        detail = controller.play_pause()

        self.assertIn("Spotify", detail)
        self.assertTrue(any("Spotify" in said for said in controller.told))
        self.assertEqual(controller.keyed, [])

    def test_a_browser_gets_the_players_own_shortcut(self):
        controller = FakeMac(running={"Google Chrome"})
        detail = controller.play_pause()

        self.assertIn("Chrome", detail)
        self.assertEqual(controller.keyed, [4242])

    def test_a_named_app_wins_over_whatever_else_is_open(self):
        controller = FakeMac(running={"Spotify", "Google Chrome"},
                             target="Google Chrome")
        detail = controller.play_pause()

        self.assertIn("Chrome", detail)
        self.assertEqual(controller.told, [])

    def test_a_player_that_is_shut_is_never_spoken_to(self):
        """Naming an app in AppleScript is enough to launch it.

        The obvious "if it is running then tell it" starts the very thing
        it was written to avoid starting, so whether it is running has to
        be settled before saying anything at all.
        """

        controller = FakeMac(running={"Google Chrome"})
        controller.play_pause()

        self.assertFalse([said for said in controller.told if "Music" in said])


if __name__ == "__main__":
    unittest.main(verbosity=2)
