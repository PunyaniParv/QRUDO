"""One QRUDO at a time: the guard against 'could not open camera 0'.

Two copies fighting over the one camera was the quiet cause -- the
first holds it, the second cannot, nothing says why.  These pin that a
second acquire is refused while the first is held, and freed the
moment the first lets go, so a crashed QRUDO never blocks the next.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import singleton


@unittest.skipUnless(sys.platform != "win32", "flock is POSIX")
class TestSingleInstance(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        patcher = mock.patch("paths.data_dir", lambda: self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_first_acquire_succeeds(self):
        lock = singleton.SingleInstance()
        lock.acquire()
        self.addCleanup(lock.release)

        self.assertTrue((self.tmp / "qrudo.lock").exists())

    def test_a_second_acquire_is_refused_while_held(self):
        first = singleton.SingleInstance()
        first.acquire()
        self.addCleanup(first.release)

        second = singleton.SingleInstance()

        with self.assertRaises(singleton.AlreadyRunning):
            second.acquire()

    def test_releasing_frees_the_next(self):
        """A QRUDO that ends -- crash or clean -- must not block the
        next one; the OS drops the flock either way, and release is the
        clean-exit half of that."""

        first = singleton.SingleInstance()
        first.acquire()
        first.release()

        second = singleton.SingleInstance()
        second.acquire()          # must not raise
        self.addCleanup(second.release)

        self.assertIsNotNone(second._handle)


if __name__ == "__main__":
    unittest.main(verbosity=2)
