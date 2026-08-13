"""Windows backend logic, tested from any machine.

The Windows backend was written on a Mac, so the parts that can be checked
without Windows are checked here: key-press counts, output parsing, and the
PowerShell script.  What these cannot prove is that Windows itself responds to
the keys -- that is what ``python main.py --selftest`` on a real Windows box is
for.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sarv import ControlConfig, log
from sarv.backends import windows
from sarv.backends.windows import (
    VK_LEFT,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PLAY_PAUSE,
    VK_RIGHT,
    VK_VOLUME_DOWN,
    VK_VOLUME_UP,
    WindowsController,
)
from sarv.controller import UnsupportedCommand

log.setup(console=False)


class FakeWindowsController(WindowsController):
    """WindowsController with user32 and PowerShell replaced by recorders."""

    def __init__(self, config=None, reply="40|48|ok"):
        self.config = config or ControlConfig()
        self.log = log.get_logger("test")
        self.pressed: list[tuple[int, bool]] = []
        self.scripts: list[str] = []
        self.reply = reply

    def _press(self, key, *, extended=False):
        self.pressed.append((key, extended))

    def _powershell(self, script):
        self.scripts.append(script)
        return self.reply


class TestVolume(unittest.TestCase):
    def test_step_becomes_whole_key_presses(self):
        """Windows moves 2% per press, so 5% rounds to 2 presses."""
        controller = FakeWindowsController()
        detail = controller.volume_up(5)
        self.assertEqual(controller.pressed, [(VK_VOLUME_UP, False)] * 2)
        self.assertIn("4%", detail)  # honest about what actually happened

    def test_direction(self):
        controller = FakeWindowsController()
        controller.volume_down(10)
        self.assertEqual(controller.pressed, [(VK_VOLUME_DOWN, False)] * 5)

    def test_tiny_step_still_presses_once(self):
        controller = FakeWindowsController()
        controller.volume_up(1)
        self.assertEqual(len(controller.pressed), 1)


class TestBrightness(unittest.TestCase):
    def test_reports_before_and_after(self):
        controller = FakeWindowsController(reply="40|48|ok")
        self.assertEqual(controller.brightness_up(8), "brightness 40% -> 48%")

    def test_at_limit_is_not_an_error(self):
        controller = FakeWindowsController(reply="100|100|ok")
        self.assertIn("already at maximum", controller.brightness_up(8))

    def test_unsupported_display_raises_unsupported_not_error(self):
        """Desktops and external monitors have no WMI brightness -- that is a
        limitation to report, not a crash."""
        controller = FakeWindowsController(reply="error|Not supported")
        with self.assertRaises(UnsupportedCommand) as caught:
            controller.brightness_down(8)
        self.assertIn("external monitors", str(caught.exception))

    def test_garbled_output_is_a_control_error(self):
        controller = FakeWindowsController(reply="something unexpected")
        with self.assertRaises(Exception):
            controller.brightness_up(8)

    def test_delta_reaches_the_script(self):
        controller = FakeWindowsController()
        controller.brightness_down(8)
        self.assertIn("$cur + (-8)", controller.scripts[0])

    def test_powershell_braces_survive_formatting(self):
        """A .format() brace bug here would only show up on Windows."""
        script = windows._BRIGHTNESS_SCRIPT.format(delta=5)
        self.assertIn("@{Brightness=[byte]$target; Timeout=[uint32]1}", script)
        self.assertIn("if ($target -gt 100) { $target = 100 }", script)


class TestMedia(unittest.TestCase):
    def test_play_pause_uses_the_media_key(self):
        controller = FakeWindowsController()
        controller.play_pause()
        self.assertEqual(controller.pressed, [(VK_MEDIA_PLAY_PAUSE, False)])

    def test_seek_repeats_arrows_and_marks_them_extended(self):
        controller = FakeWindowsController(ControlConfig(seek_seconds=10, seek_step_seconds=5))
        controller.forward(10)
        self.assertEqual(controller.pressed, [(VK_RIGHT, True)] * 2)

    def test_rewind_goes_left(self):
        controller = FakeWindowsController()
        controller.rewind(10)
        self.assertTrue(all(key == VK_LEFT for key, _ in controller.pressed))

    def test_track_mode_uses_track_keys(self):
        controller = FakeWindowsController(ControlConfig(seek_mode="track"))
        controller.forward(10)
        self.assertEqual(controller.pressed, [(VK_MEDIA_NEXT_TRACK, False)])


class TestPreflight(unittest.TestCase):
    def test_reports_the_volume_rounding(self):
        controller = FakeWindowsController(ControlConfig(volume_step=5))
        self.assertTrue(any("becomes 4%" in w for w in controller.preflight()))

    def test_reports_missing_brightness_support(self):
        controller = FakeWindowsController(reply="error|Not supported")
        self.assertTrue(any("brightness control unavailable" in w
                            for w in controller.preflight()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
