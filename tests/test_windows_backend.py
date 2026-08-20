"""Windows backend logic, tested from any machine.

The Windows backend was written on a Mac, so the parts that can be checked
without Windows are checked here: key-press counts, output parsing, and the
PowerShell script.  What these cannot prove is that Windows itself responds to
the keys -- that is what ``python main.py --selftest`` on a real Windows box is
for.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control import Command, ControlConfig, log

# Scratch logs, not the live ones -- see test_control for why.
import tempfile

log.setup(tempfile.mkdtemp(), console=False)
from control import actions
from control.backends import windows
from control.backends.windows import (
    VK_J,
    VK_L,
    VK_LEFT,
    VK_LWIN,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PLAY_PAUSE,
    VK_RIGHT,
    VK_VOLUME_DOWN,
    VK_VOLUME_UP,
    WindowsController,
)
from control.executor import ControlEngine, ControlError, UnsupportedCommand

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


class TestWhichKeySeeks(unittest.TestCase):
    """Reported on macOS, but the cause is the site, not the platform.

    Arrow keys seek on YouTube and not on YouTube Music, and both are the
    same browser -- so which key to send has to be a setting on both.
    """

    def test_arrows_by_default(self):
        controller = FakeWindowsController()
        detail = controller.forward(10)

        self.assertEqual(controller.pressed, [(VK_RIGHT, True)] * 2)
        self.assertIn("2x arrow", detail)

    def test_the_ten_second_keys_for_players_that_use_them(self):
        controller = FakeWindowsController(
            ControlConfig(browser_seek_keys="jl"))

        self.assertIn("1x j/l", controller.forward(10))
        self.assertEqual(controller.pressed, [(VK_L, False)])

    def test_back_is_the_other_one(self):
        controller = FakeWindowsController(
            ControlConfig(browser_seek_keys="jl"))
        controller.rewind(20)

        self.assertEqual(controller.pressed, [(VK_J, False)] * 2)


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


class FakeCustomController(WindowsController):
    """WindowsController with the OS seams custom actions touch replaced."""

    def __init__(self):
        self.config = ControlConfig()
        self.log = log.get_logger("test")
        self.pressed = []
        self.chords = []
        self.posted = []
        self.launched = []
        self.runs = []
        self.window = None
        self._worker = None

    def _press(self, key, *, extended=False):
        self.pressed.append(key)

    def _press_chord(self, vks):
        self.chords.append(vks)

    def _post_chord(self, hwnd, vks):
        self.posted.append((hwnd, vks))

    def _target_window(self, title=None):
        return self.window

    def _launch(self, exe):
        self.launched.append(exe)

    def _run(self, argv):
        self.runs.append(argv)
        return "ran " + " ".join(argv)


class TestOpenArgv(unittest.TestCase):
    """The ActionRunner's argv convention, turned into Windows launches."""

    def setUp(self):
        self.controller = FakeCustomController()

    def test_open_app_resolves_then_launches(self):
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        self.controller._resolve_app = lambda app: chrome

        detail = self.controller.open_argv(["open", "-a", "Google Chrome"])

        self.assertEqual(self.controller.launched, [chrome])
        self.assertEqual(detail, "launched Google Chrome")

    def test_open_app_with_an_unknown_app_refuses(self):
        self.controller._resolve_app = lambda app: None

        with self.assertRaises(UnsupportedCommand) as caught:
            self.controller.open_argv(["open", "-a", "Televator"])
        self.assertIn("Televator", str(caught.exception))
        self.assertEqual(self.controller.launched, [])

    def test_open_path_and_url_use_startfile(self):
        # create=True: os.startfile only exists on Windows, and this
        # logic must stay testable on the Mac half of CI too.
        with mock.patch("control.backends.windows.os.startfile",
                        create=True) as startfile:
            self.assertEqual(
                self.controller.open_argv(["open", "C:/temp/notes.txt"]),
                "opened C:/temp/notes.txt")
            self.assertEqual(
                self.controller.open_argv(["open", "https://example.com"]),
                "opened https://example.com")
        self.assertEqual(
            [call.args[0] for call in startfile.call_args_list],
            ["C:/temp/notes.txt", "https://example.com"])

    def test_a_confirmed_command_argv_runs_as_a_program(self):
        detail = self.controller.open_argv(["echo", "hi"])
        self.assertEqual(self.controller.runs, [["echo", "hi"]])
        self.assertEqual(detail, "ran echo hi")

    def test_an_empty_argv_is_an_error(self):
        with self.assertRaises(ControlError):
            self.controller.open_argv([])


class TestSendCombo(unittest.TestCase):
    """Taught keystrokes press through the same user32 calls as the media keys."""

    def setUp(self):
        self.controller = FakeCustomController()

    def test_combo_holds_the_modifiers_while_tapping_the_key(self):
        detail = self.controller.send_combo("cmd+shift+n")
        self.assertEqual(self.controller.chords, [[VK_LWIN, 0x10, 0x4E]])
        self.assertEqual(detail, "sent cmd+shift+n")

    def test_a_bare_key_needs_no_modifiers(self):
        self.controller.send_combo("space")
        self.assertEqual(self.controller.chords, [[0x20]])

    def test_alt_and_ctrl_map_to_their_windows_keys(self):
        # Modifiers come out of the parser alphabetically: alt before ctrl.
        self.controller.send_combo("ctrl+alt+n")
        self.assertEqual(self.controller.chords, [[0x12, 0x11, 0x4E]])

    def test_target_app_posts_the_chord_to_that_window(self):
        self.controller.window = 4242
        detail = self.controller.send_combo("cmd+n", target_app="YouTube")
        self.assertEqual(self.controller.posted, [(4242, [VK_LWIN, 0x4E])])
        self.assertEqual(self.controller.chords, [])
        self.assertIn("to YouTube", detail)

    def test_target_app_not_running_refuses(self):
        with self.assertRaises(UnsupportedCommand) as caught:
            self.controller.send_combo("cmd+n", target_app="YouTube")
        self.assertIn("not running", str(caught.exception))

    def test_an_unparsable_combo_is_unsupported(self):
        with self.assertRaises(UnsupportedCommand):
            self.controller.send_combo("cmd+flurb")


class TestKeynameToVk(unittest.TestCase):
    def test_letters_digits_and_named_keys(self):
        self.assertEqual(windows._keyname_vk("n"), 0x4E)
        self.assertEqual(windows._keyname_vk("7"), 0x37)
        self.assertEqual(windows._keyname_vk("space"), 0x20)

    def test_an_unknown_key_is_none(self):
        self.assertIsNone(windows._keyname_vk("zzz"))


class _FakeKey:
    """A registry key that answers only the values it was given."""

    def __init__(self, values):
        self._values = values

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def QueryValueEx(self, name):
        if name in self._values:
            return self._values[name], 1
        raise OSError("missing value")


class _FakeWinreg:
    """A winreg stand-in whose OpenKey answers a fixed table of subkeys."""

    HKEY_LOCAL_MACHINE = "HKLM"
    HKEY_CURRENT_USER = "HKCU"

    def __init__(self, entries):
        self._entries = entries  # subkey suffix -> default value
        self.opened = []

    def OpenKey(self, root, name):
        self.opened.append((root, name))
        for suffix, value in self._entries.items():
            if name.endswith(suffix):
                return _FakeKey({"": value})
        raise OSError("missing key")

    def QueryValueEx(self, key, name):
        return key.QueryValueEx(name)


class TestAppResolution(unittest.TestCase):
    """The registry/PATH lookup behind open_app, mocked so it runs anywhere."""

    def setUp(self):
        self.controller = FakeCustomController()
        self.dir = tempfile.mkdtemp()

    def _exe(self, name):
        path = os.path.join(self.dir, name)
        Path(path).write_text("")
        return path

    def test_a_literal_path_wins(self):
        exe = self._exe("custom.exe")
        self.assertEqual(self.controller._resolve_app(exe), exe)

    def test_path_before_registry(self):
        exe = self._exe("thing.exe")
        with mock.patch.object(windows.shutil, "which", return_value=exe), \
             mock.patch.object(windows.os.path, "isfile", return_value=False):
            self.assertEqual(self.controller._resolve_app("thing"), exe)

    def test_known_display_name_resolves_through_app_paths(self):
        exe = self._exe("chrome.exe")
        fake = _FakeWinreg({r"App Paths\chrome.exe": exe})
        isfile = mock.patch.object(
            windows.os.path, "isfile", side_effect=lambda p: p == exe)
        with mock.patch.object(windows, "winreg", fake), \
             mock.patch.object(windows.shutil, "which", return_value=None), \
             isfile:
            self.assertEqual(self.controller._resolve_app("Google Chrome"), exe)

    def test_app_paths_missing_key_is_none(self):
        with mock.patch.object(windows, "winreg", _FakeWinreg({})):
            self.assertIsNone(self.controller._app_paths("chrome"))

    def test_uninstall_display_icon_is_the_executable(self):
        exe = self._exe("App.exe")
        key = _FakeKey({"DisplayIcon": exe + ",0"})
        with mock.patch.object(windows, "winreg", _FakeWinreg({})):
            self.assertEqual(self.controller._uninstall_exe(key, "App"), exe)

    def test_uninstall_install_location_probes_for_one_exe(self):
        loc = tempfile.mkdtemp()
        exe = os.path.join(loc, "word.exe")
        Path(exe).write_text("")
        key = _FakeKey({"InstallLocation": loc, "DisplayIcon": ""})
        with mock.patch.object(windows, "winreg", _FakeWinreg({})):
            self.assertEqual(
                self.controller._uninstall_exe(key, "Microsoft Word"), exe)


class TestCustomActionsThroughTheEngine(unittest.TestCase):
    """CUSTOM rides the pipe to the Windows backend end to end."""

    def engine(self, **cfg):
        controller = FakeCustomController()
        config = ControlConfig(cooldown_seconds=0.0, **cfg)
        return ControlEngine(controller=controller, config=config), controller

    def test_open_app_custom_reaches_the_windows_backend(self):
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        engine, controller = self.engine()
        controller._resolve_app = lambda app: chrome

        result = engine.execute(
            Command.CUSTOM,
            payload=actions.serialize(
                {"type": "open_app", "app": "Google Chrome"}))

        self.assertTrue(result.ok)
        self.assertEqual(controller.launched, [chrome])

    def test_a_keystroke_custom_reaches_the_windows_backend(self):
        engine, controller = self.engine()

        result = engine.execute(Command.CUSTOM, payload="cmd+shift+n")

        self.assertTrue(result.ok)
        self.assertEqual(controller.chords, [[VK_LWIN, 0x10, 0x4E]])

    def test_an_unresolvable_app_is_unsupported_not_a_crash(self):
        engine, controller = self.engine()
        controller._resolve_app = lambda app: None

        result = engine.execute(
            Command.CUSTOM,
            payload=actions.serialize(
                {"type": "open_app", "app": "Televator"}))

        self.assertEqual(result.status, "UNSUPPORTED")
        self.assertIn("could not find", result.error)
        self.assertEqual(controller.launched, [])

    def test_an_empty_payload_is_unsupported(self):
        engine, _ = self.engine()
        result = engine.execute(Command.CUSTOM, payload="")
        self.assertEqual(result.status, "UNSUPPORTED")

    def test_volume_commands_are_unaffected(self):
        engine, controller = self.engine(volume_step=1)

        result = engine.execute(Command.VOLUME_UP)

        self.assertTrue(result.ok)
        self.assertEqual(controller.pressed, [VK_VOLUME_UP])


if __name__ == "__main__":
    unittest.main(verbosity=2)
