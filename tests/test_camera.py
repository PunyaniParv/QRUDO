"""The camera open, and its patience.

"could not open camera 0" was almost always a transient busy device --
a QRUDO just quit, a call closing, a fresh permission settling -- and
the first attempt landed in that moment.  These pin that the open
retries rather than giving up on the first no, and still raises a clear
error when the device is genuinely gone.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.camera import Camera, CameraError


class FakeCapture:
    """A cv2.VideoCapture stand-in that opens after N failed tries."""

    def __init__(self, opens_on_attempt, tracker):
        self._opens_on = opens_on_attempt
        self._tracker = tracker
        self._tracker["attempts"] += 1
        self._me = self._tracker["attempts"]

    def isOpened(self):
        return self._me >= self._opens_on

    def release(self):
        self._tracker["released"] += 1

    def set(self, *a):
        return True

    def get(self, *a):
        return 640.0


def camera_patched(opens_on_attempt):
    tracker = {"attempts": 0, "released": 0}

    fake_cv2 = mock.MagicMock()
    fake_cv2.VideoCapture.side_effect = (
        lambda *a, **k: FakeCapture(opens_on_attempt, tracker))

    return tracker, fake_cv2


class TestOpenRetries(unittest.TestCase):
    def setUp(self):
        # No real waiting between attempts.
        patcher = mock.patch("vision.camera.time.sleep", lambda _s: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_it_opens_on_the_first_try_when_free(self):
        tracker, fake_cv2 = camera_patched(opens_on_attempt=1)

        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            Camera(0).open()

        self.assertEqual(tracker["attempts"], 1)

    def test_it_retries_past_a_busy_moment(self):
        """Busy for the first two tries, free on the third."""

        tracker, fake_cv2 = camera_patched(opens_on_attempt=3)

        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            Camera(0).open()

        self.assertEqual(tracker["attempts"], 3)
        self.assertGreaterEqual(tracker["released"], 2,
                                "each failed handle must be let go")

    def test_a_genuinely_gone_camera_still_raises(self):
        tracker, fake_cv2 = camera_patched(opens_on_attempt=99)

        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            with self.assertRaises(CameraError) as caught:
                Camera(0).open()

        self.assertEqual(tracker["attempts"], Camera.OPEN_ATTEMPTS)
        self.assertIn("another", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
