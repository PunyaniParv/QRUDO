"""Parsing a user's shortcut, and carrying it through the command pipe.

A custom keystroke is how a taught gesture reaches an action QRUDO has
no handler for.  The parsing is platform-neutral and tested here; the
pipe test proves CUSTOM rides through execute() with its payload,
throttles per-keystroke, and still logs and serialises as the closed
enum value "CUSTOM".
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import ControlConfig, log
from control.commands import Command
from control.keystroke import (FLAG_CMD, FLAG_SHIFT, Combo, ComboError,
                               is_valid, parse)

log.setup(tempfile.mkdtemp(), console=False)
from control.executor import ControlEngine, Controller


class TestParse(unittest.TestCase):
    def test_a_plain_combo(self):
        combo = parse("cmd+shift+n")

        self.assertEqual(combo.key_name, "n")
        self.assertTrue(combo.flags & FLAG_CMD)
        self.assertTrue(combo.flags & FLAG_SHIFT)

    def test_order_and_spacing_do_not_matter(self):
        a = parse("cmd+shift+n")
        b = parse("shift + cmd + n")

        self.assertEqual(a.key_code, b.key_code)
        self.assertEqual(a.flags, b.flags)

    def test_aliases_fold_together(self):
        self.assertEqual(parse("win+n").flags, parse("cmd+n").flags)
        self.assertEqual(parse("option+n").flags, parse("alt+n").flags)

    def test_a_bare_key_needs_no_modifier(self):
        self.assertEqual(parse("space").key_name, "space")

    def test_no_base_key_is_refused(self):
        with self.assertRaises(ComboError):
            parse("cmd+shift")

    def test_two_base_keys_are_refused(self):
        with self.assertRaises(ComboError):
            parse("cmd+n+m")

    def test_an_unknown_token_is_refused(self):
        with self.assertRaises(ComboError):
            parse("cmd+flurb")

    def test_empty_is_refused(self):
        for bad in ("", "   ", "+"):
            with self.subTest(combo=bad):
                with self.assertRaises(ComboError):
                    parse(bad)

    def test_is_valid_matches_parse(self):
        self.assertTrue(is_valid("cmd+shift+n"))
        self.assertFalse(is_valid("cmd+flurb"))

    def test_describe_round_trips_readably(self):
        self.assertEqual(parse("shift+cmd+n").describe(), "cmd+shift+n")


class RecordingController(Controller):
    """A backend that records the combos it is asked to send."""

    name = "recording"

    def __init__(self):
        self.sent = []

    def send_combo(self, combo):
        self.sent.append(combo)
        return f"sent {combo}"

    # unused OS actions
    def volume_up(self, step): return ""
    def volume_down(self, step): return ""
    def play_pause(self): return ""
    def rewind(self, seconds): return ""
    def forward(self, seconds): return ""
    def brightness_up(self, step): return ""
    def brightness_down(self, step): return ""


class TestCustomThroughThePipe(unittest.TestCase):
    def engine(self, **cfg):
        return ControlEngine(controller=RecordingController(),
                             config=ControlConfig(cooldown_seconds=0.6,
                                                  **cfg))

    def test_a_custom_keystroke_reaches_the_backend(self):
        engine = self.engine()

        result = engine.execute(Command.CUSTOM, payload="cmd+shift+n")

        self.assertTrue(result.ok)
        self.assertEqual(engine.controller.sent, ["cmd+shift+n"])

    def test_it_still_serialises_as_the_closed_enum(self):
        """The whole reason CUSTOM carries a payload instead of many enum
        members: the vocabulary stays closed for the log and the wire."""

        self.assertEqual(json.dumps(Command.CUSTOM), '"CUSTOM"')

        engine = self.engine()
        result = engine.execute(Command.CUSTOM, payload="cmd+n")

        self.assertEqual(result.command, "CUSTOM")

    def test_two_different_keystrokes_do_not_throttle_each_other(self):
        engine = self.engine()

        first = engine.execute(Command.CUSTOM, payload="cmd+n")
        other = engine.execute(Command.CUSTOM, payload="cmd+m")

        self.assertTrue(first.ok)
        self.assertTrue(other.ok, "a different keystroke must not be throttled")
        self.assertEqual(len(engine.controller.sent), 2)

    def test_the_same_keystroke_throttles(self):
        engine = self.engine()

        engine.execute(Command.CUSTOM, payload="cmd+n")
        again = engine.execute(Command.CUSTOM, payload="cmd+n")

        self.assertEqual(again.status, "THROTTLED")

    def test_dry_run_touches_nothing(self):
        engine = self.engine(dry_run=True)

        result = engine.execute(Command.CUSTOM, payload="cmd+n")

        self.assertTrue(result.ok)
        self.assertEqual(engine.controller.sent, [])

    def test_an_empty_payload_is_unsupported_not_a_crash(self):
        engine = self.engine()

        result = engine.execute(Command.CUSTOM, payload="")

        self.assertEqual(result.status, "UNSUPPORTED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
