"""The crash supervisor's judgement, pinned.

The supervisor exists because native code dies natively: a segfault
deep in MediaPipe takes the window, the camera and the log with it,
and no Python except can catch it.  What must never regress is its
judgement -- which launches it wraps, and which exits it treats as a
crash -- because a wrong answer either loops a deliberate exit
forever or leaves a user staring at a dead app.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import died_by_signal, should_supervise


class TestWhoGetsAWatcher(unittest.TestCase):
    def test_the_double_clicked_app_is_supervised(self):
        self.assertTrue(should_supervise(True, [], {}))

    def test_the_supervised_child_never_supervises_itself(self):
        """Or every launch would nest another watcher, forever."""

        self.assertFalse(should_supervise(True, [],
                                          {"QRUDO_SUPERVISED": "1"}))

    def test_cli_runs_keep_their_exit_codes(self):
        self.assertFalse(should_supervise(True, ["--check"], {}))

    def test_a_checkout_run_is_never_wrapped(self):
        self.assertFalse(should_supervise(False, [], {}))


class TestWhatCountsAsACrash(unittest.TestCase):
    def test_a_posix_signal_death_is_a_crash(self):
        self.assertTrue(died_by_signal(-11))    # SIGSEGV
        self.assertTrue(died_by_signal(-6))     # SIGABRT

    def test_a_person_quitting_is_final(self):
        """SIGTERM and SIGINT are somebody saying quit -- Activity
        Monitor, ctrl-C, logout -- and obeying them beats resurrecting
        the app, which read as 'it relaunches whenever I quit it'."""

        self.assertFalse(died_by_signal(-15))   # SIGTERM
        self.assertFalse(died_by_signal(-2))    # SIGINT
        self.assertFalse(died_by_signal(-1))    # SIGHUP

    def test_a_windows_fatal_exception_is_a_crash(self):
        self.assertTrue(died_by_signal(0xC0000005))

    def test_a_clean_quit_is_final(self):
        self.assertFalse(died_by_signal(0))

    def test_a_deliberate_error_exit_is_final(self):
        """A config error relaunched is a crash-loop of our own
        making: small positive codes are the app MEANING it."""

        self.assertFalse(died_by_signal(1))
        self.assertFalse(died_by_signal(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
