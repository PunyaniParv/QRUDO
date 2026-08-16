"""Global hotkey wiring, tested without touching the keyboard.

The event tap and the Windows hook can only be exercised on a real desktop
session (macOS was verified by synthesising ctrl+alt+u and watching the volume
move).  What is checked here is the part that silently rots: the keycode
tables, which must stay in step with the simulator's letters.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control import ACTIONABLE_COMMANDS, Command, ControlConfig, ControlEngine, log

# Scratch logs, not the live ones -- see test_control for why.
import tempfile

log.setup(tempfile.mkdtemp(), console=False)
from control import hotkeys
from control.backends.null import NullController
from control.simulator import KEY_MAP

log.setup(console=False)


class TestKeyTables(unittest.TestCase):
    def test_macos_codes_cover_every_command(self):
        self.assertEqual(set(hotkeys._MAC_KEYCODES.values()), set(KEY_MAP))

    def test_windows_codes_cover_every_command(self):
        self.assertEqual(set(hotkeys._WIN_KEYCODES.values()), set(KEY_MAP))

    def test_windows_codes_are_uppercase_ascii(self):
        """Windows virtual key codes for letters are the uppercase ASCII values."""
        self.assertEqual(hotkeys._WIN_KEYCODES[ord("U")], "u")

    def test_every_command_is_reachable(self):
        reachable = {KEY_MAP[letter] for letter in hotkeys._MAC_KEYCODES.values()}
        self.assertEqual(reachable, set(ACTIONABLE_COMMANDS))


class TestFiring(unittest.TestCase):
    def _engine(self):
        config = ControlConfig(cooldown_seconds=0.0)
        controller = NullController(config)
        return ControlEngine(controller=controller, config=config), controller

    def test_fire_runs_the_mapped_command(self):
        engine, controller = self._engine()
        with engine:
            hotkeys._fire(engine, "u")
            deadline = time.monotonic() + 2
            while not controller.calls and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertEqual(controller.calls,
                         [f"volume +{ControlConfig().volume_step}%"])

    def test_fire_is_non_blocking(self):
        """The OS gives the keyboard callback a deadline; macOS disables a tap
        that overruns it.  Firing must not wait for a slow command."""
        engine, _ = self._engine()

        class Slow(NullController):
            def brightness_up(self, step):
                time.sleep(0.5)
                return "slow"

        engine.controller = Slow(engine.config)
        with engine:
            started = time.perf_counter()
            hotkeys._fire(engine, "b")
            elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.1, "hotkey callback waited for the command")

    def test_unmapped_letter_is_ignored(self):
        engine, controller = self._engine()
        with engine:
            hotkeys._fire(engine, "z")
            time.sleep(0.1)
        self.assertEqual(controller.calls, [])


class TestBanner(unittest.TestCase):
    def test_lists_every_chord(self):
        config = ControlConfig()
        engine = ControlEngine(controller=NullController(config), config=config)
        text = hotkeys.banner(engine)
        for letter, command in KEY_MAP.items():
            self.assertIn(f"ctrl+alt+{letter}", text)
            self.assertIn(command.value, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
