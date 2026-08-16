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
        # The frame rate is not the cost it looks like: hand detection
        # takes 5.2ms a frame at 640x480 and 7.1ms at 1760x1328, because
        # the models resize to their own fixed input either way.
        #
        # The shape is the cost, and it is asked for in fours and threes
        # for that reason.  A webcam asked for 1280x720 on this machine
        # answers with the middle band of its 4:3 picture -- measured, the
        # two match at 0.998 once the band is cropped out -- so going
        # widescreen quietly throws away a quarter of the height.  Nobody
        # notices at a distance.  Close up, where a hand fills the frame,
        # raising it takes it out of the picture, and the gestures that
        # travel up and down stop working while the far ones improve.
        #
        # Asked in fours and threes the same camera offers 1760x1328,
        # which matches the 640x480 view at 0.993: the whole picture, with
        # nearly three times the height in pixels.
        self._ask(cv2, self.width, self.height)

        # Some sensors only speak widescreen -- common on Windows
        # laptops -- and answer the 4:3 request with something short,
        # like 640x360.  For them widescreen is not a crop, it is the
        # whole picture, so the right move is to take their taller
        # widescreen mode instead: 1280x720 carries more height than
        # the 480 lines asked for, and more width buys range.  Only
        # when the answer came back short *and* wide; a camera that
        # gave the 4:3 it was asked for is left alone.
        delivered = self.shape

        if delivered:
            width, height = delivered

            if height and height < self.height and width / height > 1.5:
                taller = (1920, 1080) if self.height > 700 else (1280, 720)
                self._ask(cv2, *taller)

        # Take the newest frame rather than the oldest queued one -- a
        # backlog is felt directly as lag between moving and reacting.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return self

    def _ask(self, cv2, width, height):
        """Request one mode, with the codec that fits down the cable.

        DirectShow hands big pictures over as raw YUY2 unless asked
        otherwise, and USB bandwidth then caps the camera at a handful
        of frames a second at exactly the big sizes -- a frame rate no
        quick gesture survives.  MJPG fits the same picture down the
        same cable at full rate; a camera that cannot offer it ignores
        the request and nothing is lost.  Not asked for at the small
        default, where raw fits the bandwidth anyway and skipping the
        JPEG decode is cheaper per frame.
        """

        if sys.platform == "win32" and width * height > 640 * 480:
            self._capture.set(cv2.CAP_PROP_FOURCC,
                              cv2.VideoWriter_fourcc(*"MJPG"))

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def shape(self):
        """What the camera actually hands over, which is not what was
        asked for: it answers with the nearest mode it has."""

        if self._capture is None:
            return None

        return (int(self._capture.get(self._cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT)))

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
