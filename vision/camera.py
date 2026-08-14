"""The webcam, and nothing else.

Frames come back already mirrored, because everything downstream assumes
it: a hand moving to your right should move right on screen, and the
swipe direction depends on that.
"""

from __future__ import annotations

import sys


class CameraError(RuntimeError):
    """The camera could not be opened or read."""


class Camera:
    """A webcam that yields mirrored frames.

    Usable as a context manager, so the device is released even if the
    loop above it raises -- a camera left open needs the process killed
    before it can be opened again.
    """

    def __init__(self, index=0, mirror=True, width=640, height=480):
        self.index = index
        self.mirror = mirror
        self.width = width
        self.height = height
        self._capture = None
        self._cv2 = None

    def open(self):
        import cv2

        self._cv2 = cv2

        # Windows defaults to the Media Foundation backend, which is slow
        # to open and slow per frame on many webcams.  DirectShow is the
        # one that behaves.
        if sys.platform == "win32":
            self._capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        else:
            self._capture = cv2.VideoCapture(self.index)

        if not self._capture.isOpened():
            raise CameraError(f"could not open camera {self.index}")

        # A webcam left to itself often hands over 1280x720 or more, and
        # every one of those pixels goes through hand detection.  Nothing
        # here needs the resolution: a hand is a hand at 640x480, and the
        # frame rate is what the gestures actually depend on.
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Take the newest frame rather than the oldest queued one -- a
        # backlog is felt directly as lag between moving and reacting.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return self

    def read(self):
        """The next frame, mirrored.  Raises when the camera goes away."""

        ok, frame = self._capture.read()

        if not ok:
            raise CameraError("lost the camera")

        if self.mirror:
            frame = self._cv2.flip(frame, 1)

        return frame

    def frames(self):
        """Every frame until the camera stops."""

        while True:
            yield self.read()

    def release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.release()
