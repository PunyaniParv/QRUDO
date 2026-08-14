"""Where the Vision Engine and the Control Engine meet, and nowhere else."""

from .bridge import POSE_COMMANDS, SWIPE_COMMANDS, GestureRouter

__all__ = ["GestureRouter", "POSE_COMMANDS", "SWIPE_COMMANDS", "run"]


def run(*args, **kwargs):
    """The live loop.  Imported lazily: it needs OpenCV, the rest does not."""

    from .runner import run as _run

    return _run(*args, **kwargs)
