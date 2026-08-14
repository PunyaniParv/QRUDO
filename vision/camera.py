"""The webcam, and nothing else.

Frames come back already mirrored, because everything downstream assumes
it: a hand moving to your right should move right on screen, and the
swipe direction depends on that.
"""

from __future__ import annotations


class CameraError(RuntimeError):
    """The camera could not be opened or read."""


class Camera:
    """A webcam that yields mirrored frames.

    Usable as a context manager, so the device is released even if the
    loop above it raises -- a camera left open needs the process killed
    before it can be opened again.
    """

    def __init__(self, index=0, mirror=True):
        self.index = index
        self.mirror = mirror
        self._capture = None
        self._cv2 = None

    def open(self):
        import cv2

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(self.index)

        if not self._capture.isOpened():
            raise CameraError(f"could not open camera {self.index}")

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
