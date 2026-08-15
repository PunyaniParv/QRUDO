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

        # Range is bought with pixels.  Landmark error is roughly fixed in
        # pixels, so it grows against the hand as the hand shrinks, and a
        # hand three metres off is only about twenty pixels across at
        # 640x480.  Measured on synthetic hands with that error added, the
        # bigger picture roughly doubles the distance every pose survives.
        #
        # It was the other way round for fear of the frame rate, which
        # turns out not to be the trade it looked like: hand detection
        # takes 5.5ms a frame at 640x480 and 5.9ms at 1280x720, because
        # the models resize to their own fixed input either way.  --near
        # is there for a machine that disagrees.
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Take the newest frame rather than the oldest queued one -- a
        # backlog is felt directly as lag between moving and reacting.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return self

    #: A camera drops the odd frame -- just after opening, or under load --
    #: and one bad read is not the camera going away.  Only a run of them
    #: is.
    MISSES_ALLOWED = 30

    def read(self):
        """The next frame, mirrored.  Raises when the camera really goes."""

        if self._capture is None or self._cv2 is None:
            raise CameraError("camera used before it was opened")

        for _ in range(self.MISSES_ALLOWED):
            ok, frame = self._capture.read()

            if ok:
                return self._cv2.flip(frame, 1) if self.mirror else frame

        raise CameraError(
            f"lost the camera: {self.MISSES_ALLOWED} frames in a row came "
            f"back empty. Another program may have taken it, or it may have "
            f"been unplugged.")

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
