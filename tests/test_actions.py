"""What a gesture does when it fires: the action model and its safety.

The model is pure data -- open a thing, launch an app, run a chain --
and it rides the command pipe as one serialised string.  The safety
that matters most is here and tested hardest: a shell command runs only
when a person confirmed it, and never when it matches a destructive
pattern, and both checks hold at run time, not only at save time.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import actions
from control.actions import ActionError, ActionRunner


class TestValidate(unittest.TestCase):
    def test_each_type_needs_its_field(self):
        good = [
            {"type": "open_path", "path": "/tmp"},
            {"type": "open_app", "app": "Spotify"},
            {"type": "open_url", "url": "https://x.com"},
            {"type": "keystroke", "combo": "cmd+n"},
            {"type": "builtin", "command": "VOLUME_UP"},
        ]
        for action in good:
            with self.subTest(action=action):
                self.assertEqual(actions.validate(action)["type"],
                                 action["type"])

    def test_a_missing_field_is_refused(self):
        with self.assertRaises(ActionError):
            actions.validate({"type": "open_app"})

    def test_an_unknown_type_is_refused(self):
        with self.assertRaises(ActionError):
            actions.validate({"type": "self_destruct", "x": "y"})

    def test_unknown_keys_are_dropped(self):
        clean = actions.validate(
            {"type": "open_app", "app": "Spotify", "sneaky": "rm -rf /"})
        self.assertNotIn("sneaky", clean)


class TestSafety(unittest.TestCase):
    def test_a_destructive_command_is_refused_at_save(self):
        for bad in ("rm -rf /", "sudo rm x", "dd if=/dev/zero of=/dev/sda",
                    ":(){ :|:& };:", "mkfs.ext4 /dev/sda"):
            with self.subTest(command=bad):
                with self.assertRaises(ActionError):
                    actions.validate({"type": "run_command", "command": bad,
                                     "confirmed": True})

    def test_an_ordinary_command_validates(self):
        clean = actions.validate(
            {"type": "run_command", "command": "echo hi", "confirmed": True})
        self.assertTrue(clean["confirmed"])

    def test_confirmed_defaults_false(self):
        clean = actions.validate(
            {"type": "run_command", "command": "echo hi"})
        self.assertFalse(clean["confirmed"])


class TestSerialize(unittest.TestCase):
    def test_round_trip(self):
        chain = [{"type": "open_app", "app": "Spotify"},
                 {"type": "open_url", "url": "https://gmail.com"}]
        back = actions.parse(actions.serialize(chain))

        self.assertEqual([a["type"] for a in back],
                         ["open_app", "open_url"])

    def test_a_single_action_is_a_one_element_chain(self):
        s = actions.serialize({"type": "open_app", "app": "Spotify"})
        self.assertEqual(len(actions.parse(s)), 1)

    def test_a_legacy_bare_combo_still_parses(self):
        """The CUSTOM payload was a raw combo before chains; it must
        still mean a keystroke."""

        parsed = actions.parse("cmd+shift+n")
        self.assertEqual(parsed, [{"type": "keystroke", "combo": "cmd+shift+n"}])

    def test_describe_reads_in_words(self):
        text = actions.describe([{"type": "open_app", "app": "Spotify"},
                                 {"type": "open_url", "url": "gmail.com"}])
        self.assertIn("launch Spotify", text)
        self.assertIn("then", text)


class RecordingRunner(ActionRunner):
    """An ActionRunner whose OS pieces just record what they were asked."""

    def __init__(self):
        self.opened = []
        self.keyed = []
        self.builtins = []
        self.quits = []
        self.hidden = []
        super().__init__(
            opener=lambda argv: self.opened.append(argv),
            keystroke=lambda combo, target="": self.keyed.append(
                (combo, target)) or f"sent {combo}",
            builtin=lambda cmd: self.builtins.append(cmd) or f"did {cmd}",
            quitter=lambda app: self.quits.append(app) or f"quit {app}",
            hider=lambda app: self.hidden.append(app) or f"hid {app}")


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.runner = RecordingRunner()

    def test_open_app_shells_out_with_dash_a(self):
        self.runner.run(actions.serialize(
            {"type": "open_app", "app": "Spotify"}))
        self.assertEqual(self.runner.opened, [["open", "-a", "Spotify"]])

    def test_open_url_and_path(self):
        self.runner.run(actions.serialize(
            [{"type": "open_url", "url": "https://x.com"},
             {"type": "open_path", "path": "/tmp"}]))
        self.assertEqual(self.runner.opened,
                         [["open", "https://x.com"], ["open", "/tmp"]])

    def test_a_chain_runs_in_order(self):
        self.runner.run(actions.serialize(
            [{"type": "open_app", "app": "A"},
             {"type": "keystroke", "combo": "cmd+n"},
             {"type": "builtin", "command": "VOLUME_UP"}]))
        self.assertEqual(self.runner.opened, [["open", "-a", "A"]])
        self.assertEqual(self.runner.keyed, [("cmd+n", "")])
        self.assertEqual(self.runner.builtins, ["VOLUME_UP"])

    def test_keystrokes_are_recognisable_for_the_gesture_gate(self):
        """The gesture form refuses keystroke actions until that
        feature exists on purpose; the check must see through chains
        and miss nothing else."""

        self.assertTrue(actions.contains_keystroke(
            {"type": "keystroke", "combo": "shift+n"}))
        self.assertTrue(actions.contains_keystroke(
            [{"type": "open_app", "app": "Spotify"},
             {"type": "keystroke", "combo": "cmd+k"}]))
        self.assertFalse(actions.contains_keystroke(
            [{"type": "open_path", "path": "~/Downloads"},
             {"type": "quit_app", "app": "all"}]))

    def test_quit_app_reaches_the_quitter(self):
        self.runner.run(actions.serialize(
            {"type": "quit_app", "app": "Spotify"}))
        self.assertEqual(self.runner.quits, ["Spotify"])

    def test_quit_all_carries_the_keyword(self):
        self.runner.run(actions.serialize(
            {"type": "quit_app", "app": "all"}))
        self.assertEqual(self.runner.quits, ["all"])

    def test_hide_app_reaches_the_hider(self):
        self.runner.run(actions.serialize(
            {"type": "hide_app", "app": "Spotify"}))
        self.assertEqual(self.runner.hidden, ["Spotify"])

    def test_hide_all_carries_the_keyword(self):
        self.runner.run(actions.serialize(
            {"type": "hide_app", "app": "all"}))
        self.assertEqual(self.runner.hidden, ["all"])

    def test_quitting_without_a_quitter_refuses_clearly(self):
        """A platform with no quit support must say so, not crash."""

        bare = ActionRunner(opener=lambda argv: None,
                            keystroke=lambda c, t="": "",
                            builtin=lambda c: "")
        with self.assertRaises(ActionError):
            bare.run(actions.serialize(
                {"type": "quit_app", "app": "Spotify"}))

    def test_an_unconfirmed_command_refuses_at_run_time(self):
        payload = actions.serialize(
            {"type": "run_command", "command": "echo hi", "confirmed": False})
        with self.assertRaises(ActionError):
            self.runner.run(payload)
        self.assertEqual(self.runner.opened, [])

    def test_a_confirmed_ordinary_command_runs(self):
        payload = actions.serialize(
            {"type": "run_command", "command": "echo hi", "confirmed": True})
        self.runner.run(payload)
        self.assertEqual(self.runner.opened, [["echo", "hi"]])

    def test_a_destructive_command_refuses_at_run_time_too(self):
        """Even a payload hand-built to set confirmed on a dangerous
        command is refused when it runs -- defence in depth."""

        # Bypass validate() by building the payload string directly.
        import json
        payload = json.dumps([{"type": "run_command",
                               "command": "rm -rf /tmp/x",
                               "confirmed": True}])
        with self.assertRaises(ActionError):
            self.runner.run(payload)
        self.assertEqual(self.runner.opened, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
