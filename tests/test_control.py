"""Control-layer tests that touch no hardware.

Run with:  python -m unittest discover tests

These use NullController, so they pass on any machine and in CI -- the OS side
is covered separately by ``python main.py --selftest``.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control import ACTIONABLE_COMMANDS, Command, ControlConfig, ControlEngine, Status, log, parse_command
from control.commands import TARGET_COMMANDS

# Into a scratch directory before any engine runs a command: the log
# module configures itself on first use and keeps that choice, and left
# to default it wrote every test's commands into the live
# logs/commands.jsonl -- where an accuracy analysis read them as the
# user's.
import tempfile

log.setup(tempfile.mkdtemp(), console=False)
from control.commands import NO_CHANGE
from control.backends.null import NullController
from control.executor import ControlError
from control.simulator import KEY_MAP

log.setup(console=False)  # keep test output readable; the file log still fills


def make_engine(**overrides) -> tuple[ControlEngine, NullController]:
    overrides.setdefault("cooldown_seconds", 0.0)
    config = ControlConfig(**overrides)
    controller = NullController(config)
    engine = ControlEngine(controller=controller, config=config)
    # Hermetic: the target resolver must not ask this machine's OS what
    # is running mid-test.
    engine.targets.probe = lambda: {}
    return engine, controller


class TestCommands(unittest.TestCase):
    def test_commands_are_plain_strings(self):
        """The vision engine may send commands over JSON or a socket."""
        self.assertEqual(Command.VOLUME_UP, "VOLUME_UP")

    def test_parse_is_case_insensitive(self):
        self.assertIs(parse_command("volume_up"), Command.VOLUME_UP)
        self.assertIs(parse_command("  Play_Pause "), Command.PLAY_PAUSE)

    def test_parse_rejects_unknown(self):
        with self.assertRaises(ValueError):
            parse_command("EJECT_CD")

    def test_simulator_covers_every_command(self):
        self.assertEqual(set(KEY_MAP.values()), set(ACTIONABLE_COMMANDS))
        self.assertEqual(len(KEY_MAP), len(ACTIONABLE_COMMANDS))


class TestExecution(unittest.TestCase):
    def test_every_command_reaches_the_backend(self):
        """Spec F: verify every command independently.

        The target commands are engine-level by design -- they move
        where the others go -- so the backend sees every command except
        those, and those still answer OK.
        """
        engine, controller = make_engine()
        backend_commands = [command for command in ACTIONABLE_COMMANDS
                            if command not in TARGET_COMMANDS]
        for command in ACTIONABLE_COMMANDS:
            with self.subTest(command=command):
                result = engine.execute(command)
                self.assertEqual(result.status, Status.OK, result.error)
        self.assertEqual(len(controller.calls), len(backend_commands))

    def test_none_is_a_noop(self):
        engine, controller = make_engine()
        self.assertEqual(engine.execute(Command.NONE).status, Status.NOOP)
        self.assertEqual(controller.calls, [])

    def test_string_commands_accepted(self):
        engine, _ = make_engine()
        self.assertEqual(engine.execute("VOLUME_UP").status, Status.OK)

    def test_unknown_command_does_not_raise(self):
        engine, _ = make_engine()
        result = engine.execute("SELF_DESTRUCT")
        self.assertEqual(result.status, Status.ERROR)
        self.assertFalse(result.ok)

    def test_config_steps_are_passed_through(self):
        engine, controller = make_engine(volume_step=12, brightness_step=3, seek_seconds=30)
        engine.execute(Command.VOLUME_UP)
        engine.execute(Command.BRIGHTNESS_DOWN)
        engine.execute(Command.FORWARD)
        self.assertEqual(controller.calls, ["volume +12%", "brightness -3%", "forward 30s"])

    def test_dry_run_never_touches_the_backend(self):
        engine, controller = make_engine(dry_run=True)
        for command in ACTIONABLE_COMMANDS:
            self.assertTrue(engine.execute(command).ok)
        self.assertEqual(controller.calls, [])


class TestFailureHandling(unittest.TestCase):
    """Spec F: a failed OS action must not crash SARV."""

    class Exploding(NullController):
        def volume_up(self, step): raise ControlError("audio device went away")
        def play_pause(self): raise RuntimeError("unexpected backend bug")

    def test_control_error_becomes_a_result(self):
        config = ControlConfig(cooldown_seconds=0.0)
        engine = ControlEngine(controller=self.Exploding(config), config=config)
        result = engine.execute(Command.VOLUME_UP)
        self.assertEqual(result.status, Status.ERROR)
        self.assertIsNotNone(result.error)
        self.assertIn("audio device went away", result.error or "")

    def test_unexpected_exception_is_contained(self):
        config = ControlConfig(cooldown_seconds=0.0)
        engine = ControlEngine(controller=self.Exploding(config), config=config)
        result = engine.execute(Command.PLAY_PAUSE)
        self.assertEqual(result.status, Status.ERROR)
        self.assertIsNotNone(result.error)
        self.assertIn("RuntimeError", result.error or "")
        # the engine still works afterwards
        self.assertTrue(engine.execute(Command.VOLUME_DOWN).ok)


class TestCooldown(unittest.TestCase):
    def test_repeats_are_throttled(self):
        """A held gesture at 30 fps must not fire 30 volume steps."""
        engine, controller = make_engine(cooldown_seconds=10.0)
        for _ in range(30):
            engine.execute(Command.VOLUME_UP)
        self.assertEqual(len(controller.calls), 1)

    def test_throttle_is_per_command(self):
        engine, controller = make_engine(cooldown_seconds=10.0)
        engine.execute(Command.VOLUME_UP)
        engine.execute(Command.BRIGHTNESS_UP)
        self.assertEqual(len(controller.calls), 2)

    def test_force_bypasses_cooldown(self):
        engine, controller = make_engine(cooldown_seconds=10.0)
        engine.execute(Command.VOLUME_UP)
        engine.execute(Command.VOLUME_UP, force=True)
        self.assertEqual(len(controller.calls), 2)

    def test_cooldown_expires(self):
        engine, controller = make_engine(cooldown_seconds=0.05)
        engine.execute(Command.VOLUME_UP)
        time.sleep(0.1)
        engine.execute(Command.VOLUME_UP)
        self.assertEqual(len(controller.calls), 2)


class TestAsyncSubmit(unittest.TestCase):
    def test_submit_runs_every_command_and_reports(self):
        """Paced submissions -- the realistic case -- must all get through."""
        engine, controller = make_engine()
        seen = []
        engine.on_result = seen.append
        with engine:
            for command in ACTIONABLE_COMMANDS:
                engine.submit(command)
                time.sleep(0.02)  # gestures arrive far slower than this
            deadline = time.monotonic() + 5
            while len(seen) < len(ACTIONABLE_COMMANDS) and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertEqual([r.command for r in seen], [c.value for c in ACTIONABLE_COMMANDS])

    def test_burst_drops_rather_than_lagging(self):
        """A flood must not build an ever-growing backlog of stale commands.

        Whatever survives is still executed in submission order.
        """
        engine, _ = make_engine()
        seen = []
        engine.on_result = seen.append
        with engine:
            for _ in range(200):
                engine.submit(Command.VOLUME_UP)
            time.sleep(0.3)
        self.assertGreater(len(seen), 0)
        self.assertLess(len(seen), 200)
        self.assertTrue(all(r.command == "VOLUME_UP" for r in seen))

    def test_submit_does_not_raise_on_bad_command(self):
        engine, _ = make_engine()
        seen = []
        engine.on_result = seen.append
        with engine:
            engine.submit("NOT_REAL")
            time.sleep(0.2)
        self.assertEqual([r.status for r in seen], [Status.ERROR])

    def test_close_without_submit_is_safe(self):
        engine, _ = make_engine()
        engine.close()


class TestSelfTestRestores(unittest.TestCase):
    """The self-test must leave the machine exactly as it found it."""

    class AtMaximum(NullController):
        """A machine whose brightness is already at 100%."""

        def brightness_up(self, step):
            self.calls.append("brightness up (no-op)")
            return f"brightness {NO_CHANGE} maximum (100%)"

    def test_no_op_command_is_not_undone(self):
        """Undoing a command that did nothing would dim the screen.

        Regression: BRIGHTNESS_UP at 100% did nothing, but the paired
        BRIGHTNESS_DOWN still fired, so running the self-test left the display
        one step darker every time.
        """
        from control import selftest

        config = ControlConfig(cooldown_seconds=0.0)
        controller = self.AtMaximum(config)
        engine = ControlEngine(controller=controller, config=config)

        original_pause = selftest.PAUSE_BETWEEN
        selftest.PAUSE_BETWEEN = 0.0
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                selftest.run(engine)
        finally:
            selftest.PAUSE_BETWEEN = original_pause

        ups = controller.calls.count("brightness up (no-op)")
        downs = controller.calls.count(
            f"brightness -{ControlConfig().brightness_step}%")
        # BRIGHTNESS_UP runs twice: once as its own test, once undoing the
        # BRIGHTNESS_DOWN test.  Both are no-ops on this machine.
        self.assertEqual(ups, 2)
        # BRIGHTNESS_DOWN should run exactly once -- as its own test.  With the
        # bug it ran twice, the second time "undoing" a no-op, and that second
        # one is what dimmed the screen on every run.
        self.assertEqual(downs, 1, f"no-op was undone anyway: {controller.calls}")


class TestTheGestureCooldownIsASetting(unittest.TestCase):
    """One movement crosses many frames, and each of them is a chance to
    fire the same command again.

    Time is the weakest of the guards against that and the only one worth
    making adjustable: the others -- a raise having to come back where it
    started, a turn having to go quiet, a held pose having to be dropped
    and made again -- do not depend on how long the movement took.
    """

    def test_it_defaults_to_one_second(self):
        self.assertAlmostEqual(ControlConfig().gesture_cooldown_seconds, 1.0)

    def test_it_can_be_set(self):
        import json
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "sarv_config.json"
        path.write_text(json.dumps({"gesture_cooldown_seconds": 0.6}))

        self.assertAlmostEqual(
            ControlConfig.load(path).gesture_cooldown_seconds, 0.6)


class TestSeekConfig(unittest.TestCase):
    def test_press_count_matches_player_granularity(self):
        self.assertEqual(ControlConfig(seek_seconds=10, seek_step_seconds=5).seek_presses, 2)
        self.assertEqual(ControlConfig(seek_seconds=10, seek_step_seconds=10).seek_presses, 1)
        self.assertEqual(ControlConfig(seek_seconds=30, seek_step_seconds=5).seek_presses, 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
