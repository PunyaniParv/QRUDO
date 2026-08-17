"""A fresh machine becomes a running QRUDO without anyone's help.

ensure() is the first thing a new device hits, so these tests pin its
promises: silent when the environment is complete, one handoff and
never a loop, and a re-install when requirements.txt changes rather
than a crash halfway into the camera loop.  Everything that would touch
the real machine -- venv creation, pip, the exec -- is recorded here
instead of run.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bootstrap


class BootstrapCase(unittest.TestCase):
    """A temp dir standing in for the repo, with every side effect taped."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.requirements = self.tmp / "requirements.txt"
        self.requirements.write_text("mediapipe==1.0.0\n")
        self.venv = self.tmp / ".venv"

        self.ran = []
        self.handoffs = []

        patches = {
            "VENV": self.venv,
            "REQUIREMENTS": self.requirements,
            "STAMP": self.venv / "requirements.sha256",
            "_run": self.record_run,
            "_handoff": self.record_handoff,
        }
        for name, value in patches.items():
            patcher = mock.patch.object(bootstrap, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        env = mock.patch.dict("os.environ", {}, clear=False)
        env.start()
        self.addCleanup(env.stop)

        # The first-run notice is for a person watching a real install,
        # not for the test log.
        quiet = mock.patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)
        import os
        os.environ.pop(bootstrap.GUARD, None)

    def record_run(self, cmd, what):
        self.ran.append(cmd)
        # Creating the venv must leave a python behind, or the install
        # step that follows would address an interpreter that is not
        # there.  The recorder keeps that much of the real behaviour.
        if cmd[1:3] == ["-m", "venv"]:
            py = bootstrap.venv_python()
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")

    def record_handoff(self, py):
        self.handoffs.append(py)

    def make_venv(self, stamped=True):
        py = bootstrap.venv_python()
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("")
        if stamped:
            bootstrap.STAMP.write_text(bootstrap.fingerprint())
        return py


class TestCompleteEnvironment(BootstrapCase):
    def test_nothing_happens(self):
        """An interpreter that has everything is left alone.

        This is every start after the first, and also anyone running
        their own environment -- the old workflow stays legal.
        """

        with mock.patch.object(bootstrap, "missing", lambda: []):
            bootstrap.ensure()

        self.assertEqual(self.ran, [])
        self.assertEqual(self.handoffs, [])


class TestFreshMachine(BootstrapCase):
    def test_builds_installs_and_hands_off(self):
        with mock.patch.object(bootstrap, "missing", lambda: ["cv2"]):
            bootstrap.ensure()

        self.assertEqual(len(self.ran), 2, "one venv creation, one pip")
        self.assertIn("venv", self.ran[0])
        self.assertIn("install", self.ran[1])
        self.assertEqual(self.handoffs, [bootstrap.venv_python()])

    def test_stamp_written_after_install(self):
        """The fingerprint is recorded only once pip has succeeded.

        _run raises on failure, so a broken install leaves no stamp and
        the next start tries again instead of trusting a half-filled
        venv.
        """

        with mock.patch.object(bootstrap, "missing", lambda: ["cv2"]):
            bootstrap.ensure()

        self.assertEqual(bootstrap.STAMP.read_text(),
                         bootstrap.fingerprint())


class TestExistingVenv(BootstrapCase):
    def test_good_venv_is_a_bare_handoff(self):
        """The everyday wrong-python start: hand off, install nothing.

        This is the path a stray `python main.py` takes on a machine
        that is already set up, so it has to stay instant.
        """

        self.make_venv(stamped=True)

        with mock.patch.object(bootstrap, "missing", lambda: ["cv2"]):
            bootstrap.ensure()

        self.assertEqual(self.ran, [])
        self.assertEqual(len(self.handoffs), 1)

    def test_changed_requirements_reinstall(self):
        """Pulling a version with new dependencies repairs itself."""

        self.make_venv(stamped=True)
        self.requirements.write_text("mediapipe==2.0.0\n")

        with mock.patch.object(bootstrap, "missing", lambda: ["cv2"]):
            bootstrap.ensure()

        self.assertEqual(len(self.ran), 1)
        self.assertIn("install", self.ran[0])
        self.assertEqual(len(self.handoffs), 1)


class TestBrokenInstall(BootstrapCase):
    def test_second_pass_explains_instead_of_looping(self):
        """After one handoff the guard is up: fail with words, not a loop.

        The child that still cannot import cv2 must never build another
        venv and exec again -- that spiral would pin the CPU and say
        nothing.
        """

        import os
        os.environ[bootstrap.GUARD] = "1"

        with mock.patch.object(bootstrap, "missing", lambda: ["cv2"]):
            with self.assertRaises(SystemExit) as caught:
                bootstrap.ensure()

        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(self.ran, [])
        self.assertEqual(self.handoffs, [])


if __name__ == "__main__":
    unittest.main()
