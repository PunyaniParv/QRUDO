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

from control import ControlConfig, log
from control.backends import windows
from control.backends.windows import (
    VK_LEFT,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PLAY_PAUSE,
    VK_RIGHT,
    VK_VOLUME_DOWN,
    VK_VOLUME_UP,
    WindowsController,
)
from control.executor import UnsupportedCommand

log.setup(console=False)


class FakeWindowsController(WindowsController):
    """WindowsController with user32 and PowerShell replaced by recorders."""

    def __init__(self, config=None, reply="40|48|ok", read_reply="40|ok"):
        self.config = config or ControlConfig()
        self.log = log.get_logger("test")
        self.pressed: list[tuple[int, bool]] = []
        self.scripts: list[str] = []
        self.reply = reply            # what the setting script returns
        self.read_reply = read_reply  # what the read-only script returns
        # Typed as the real thing, because some tests put a stand-in here.
        self._worker: windows._PowerShellWorker | None = None

    def _press(self, key, *, extended=False):
        self.pressed.append((key, extended))

    def _powershell(self, script):
        self.scripts.append(script)
        # The two scripts return different shapes: "old|new|ok" vs "level|ok".
        return self.reply if "WmiSetBrightness" in script else self.read_reply


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


class TestSeekTargeting(unittest.TestCase):
    """Aiming seek keys at a named window instead of the focused one."""

    def test_no_target_uses_ordinary_key_presses(self):
        controller = FakeWindowsController(ControlConfig(seek_target_app=""))
        detail = controller.forward(10)
        self.assertEqual(controller.pressed, [(VK_RIGHT, True)] * 2)
        self.assertNotIn(" to ", detail)

    def test_missing_window_falls_back_to_key_presses(self):
        """A configured app that is not open must not break seeking."""
        controller = FakeWindowsController(ControlConfig(seek_target_app="YouTube"))
        controller._target_window = lambda title=None: None
        controller.forward(10)
        self.assertEqual(controller.pressed, [(VK_RIGHT, True)] * 2)

    def test_found_window_receives_posted_keys(self):
        controller = FakeWindowsController(ControlConfig(seek_target_app="YouTube"))
        controller._target_window = lambda title=None: 4242
        posted = []
        controller._post_key_to_window = lambda hwnd, key: posted.append((hwnd, key))
        detail = controller.forward(10)
        self.assertEqual(posted, [(4242, VK_RIGHT)] * 2)
        self.assertEqual(controller.pressed, [])  # nothing sent to the focused window
        self.assertIn("to YouTube", detail)


class TestState(unittest.TestCase):
    def test_reports_brightness(self):
        controller = FakeWindowsController(read_reply="37|ok")
        self.assertEqual(controller.read_state(), {"brightness": 0.37})

    def test_unreadable_display_is_blank_not_an_error(self):
        controller = FakeWindowsController(read_reply="error|Not supported")
        self.assertEqual(controller.read_state(), {})

    def test_restore_asks_for_the_exact_delta(self):
        """Windows can only move brightness relatively, so restoring 40% from
        37% has to become "+3"."""
        controller = FakeWindowsController(read_reply="37|ok", reply="37|40|ok")
        controller.restore_state({"brightness": 0.40})
        self.assertIn("$cur + (3)", controller.scripts[-1])

    def test_restore_does_nothing_when_already_correct(self):
        controller = FakeWindowsController(read_reply="37|ok")
        controller.restore_state({"brightness": 0.37})
        self.assertEqual(len(controller.scripts), 1)  # the read only, no write


class TestPersistentPowerShell(unittest.TestCase):
    """The resident helper must never be able to break brightness control."""

    # Stand-ins for the resident helper.  They subclass the real thing so
    # that they still fit where it fits, and so a change to its interface
    # shows up here rather than passing silently.

    class DeadWorker(windows._PowerShellWorker):
        def __init__(self):
            pass  # no process, no thread pool, nothing to clean up

        def exchange(self, line, timeout=6.0):
            return None  # unavailable, however it failed

    class LiveWorker(windows._PowerShellWorker):
        def __init__(self):
            self.lines = []

        def exchange(self, line, timeout=6.0):
            self.lines.append(line)
            return "40|48|ok" if line != "read" else "40|ok"

    def test_falls_back_to_launching_powershell(self):
        controller = FakeWindowsController()
        controller._worker = self.DeadWorker()
        self.assertEqual(controller.brightness_up(8), "brightness 40% -> 48%")
        self.assertEqual(len(controller.scripts), 1)  # the one-shot path ran

    def test_uses_the_worker_when_it_answers(self):
        worker = self.LiveWorker()
        controller = FakeWindowsController()
        controller._worker = worker
        self.assertEqual(controller.brightness_up(8), "brightness 40% -> 48%")
        self.assertEqual(controller.scripts, [])      # no process was launched
        self.assertEqual(worker.lines, ["8"])

    def test_reads_go_through_the_worker_too(self):
        worker = self.LiveWorker()
        controller = FakeWindowsController()
        controller._worker = worker
        self.assertEqual(controller.read_state(), {"brightness": 0.40})
        self.assertEqual(worker.lines, ["read"])


class TestResidentLoopScript(unittest.TestCase):
    """The loop runs only on Windows, so its text is checked here instead."""

    script = windows._PowerShellWorker._LOOP

    def test_braces_survive_formatting(self):
        self.assertIn("@{Brightness=[byte]$target; Timeout=[uint32]1}", self.script)
        self.assertIn("while ($true) {", self.script)

    def test_methods_object_is_fetched_outside_the_loop(self):
        """Re-querying it per command was one wasted WMI call every time."""
        self.assertLess(self.script.index("$methods = $null"),
                        self.script.index("while ($true)"))

    def test_cache_window_is_substituted(self):
        self.assertIn(str(windows._PowerShellWorker._CACHE_SECONDS), self.script)

    def test_reads_always_go_to_the_hardware(self):
        """A cached value is fine for a burst of set commands, but a read is
        used to restore the machine, so it must be true."""
        self.assertIn("$line -eq 'read' -or $null -eq $last -or $stale", self.script)

    def test_failure_clears_the_cache(self):
        after_catch = self.script.split("catch")[1]
        self.assertIn("$methods = $null", after_catch)
        self.assertIn("$last = $null", after_catch)


class TestPreflight(unittest.TestCase):
    def test_reports_the_volume_rounding(self):
        controller = FakeWindowsController(ControlConfig(volume_step=5))
        self.assertTrue(any("becomes 4%" in w for w in controller.preflight()))

    def test_reports_missing_brightness_support(self):
        controller = FakeWindowsController(read_reply="error|Not supported")
        self.assertTrue(any("brightness control unavailable" in w
                            for w in controller.preflight()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
