"""One QRUDO at a time, because there is one camera.

Two copies running is the quiet cause of "could not open camera 0":
the first holds the camera, the second cannot, and nothing on screen
says why.  A double-click while it is already running -- easy to do
when the window is behind others -- is all it takes.

So a launch takes a lock first.  If another live QRUDO holds it, this
one steps aside with a word rather than fighting for a device it will
never get.  The lock is a file whose flock the OS drops automatically
when the process ends however it ends -- a crash included -- so a dead
QRUDO never blocks the next one, which a plain pid file could not
promise.
"""

from __future__ import annotations

from pathlib import Path


class AlreadyRunning(RuntimeError):
    """Another live QRUDO holds the camera lock."""


class SingleInstance:
    """A held file lock that means 'this QRUDO owns the camera'.

    Kept as an attribute for the life of the process; releasing it (or
    exiting) frees the next launch.  On a platform without flock the
    lock is a no-op that always succeeds -- better one camera clash
    than refusing to start at all.
    """

    def __init__(self, name="qrudo.lock"):
        from paths import data_dir

        self.path = data_dir() / name
        self._handle = None

    def acquire(self):
        try:
            import fcntl
        except ImportError:
            return self          # Windows path handled below in main

        self._handle = open(self.path, "w")

        try:
            fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._handle.close()
            self._handle = None
            raise AlreadyRunning(
                "another copy of QRUDO is already running") from exc

        return self

    def release(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc_info):
        self.release()
