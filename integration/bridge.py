"""Gesture names in, commands out.

This is the entire contract between the two halves.  The vision side
returns strings and knows nothing about the machine; the control side
takes commands and knows nothing about cameras.  Changing what a gesture
does is a change to this file alone.
"""

from __future__ import annotations

from control import Command

#: Held poses.  These fire once when you make them, not repeatedly while
#: you hold them -- see GestureRouter.
POSE_COMMANDS = {
    "FIST": Command.PLAY_PAUSE,
}

#: Movements.  Two fingers facing the camera, wrist rotated left or right.
#: These are already one-off events with their own cooldown.
SWIPE_COMMANDS = {
    "SWIPE_LEFT": Command.REWIND,
    "SWIPE_RIGHT": Command.FORWARD,
}

# Deliberately unmapped for now:
#
#   TWO_FINGER  is the swipe pose.  Binding it would fire a command every
#               time you got ready to swipe.
#   OPEN_PALM   is what a hand looks like on its way to and from every
#               other gesture, so it would fire constantly.
#   POINT       is free, but volume and brightness need four distinct
#               signals and there are not four left.  Vertical swipes are
#               the natural way to find them.


class GestureRouter:
    """Turn a stream of gesture names into commands.

    Held poses fire on the way in and then stay quiet.  Without that, a
    fist held for two seconds would toggle play/pause thirty times,
    because the vision side reports what your hand *is* on every frame,
    not what changed.
    """

    def __init__(self, poses=None, swipes=None):
        self.poses = POSE_COMMANDS if poses is None else poses
        self.swipes = SWIPE_COMMANDS if swipes is None else swipes
        self._held = None

    def update(self, gesture=None, swipe=None):
        """Return the command this frame should run, or None."""

        # A swipe is a movement that already happened, so it always counts.
        if swipe in self.swipes:
            # Whatever pose was being held was part of the swipe; make it
            # ask again rather than firing as the hand settles.
            self._held = None
            return self.swipes[swipe]

        if gesture == self._held:
            return None

        self._held = gesture

        return self.poses.get(gesture)

    def forget(self):
        """Hand left the frame; the next pose is a new one."""

        self._held = None

    def mapping(self):
        """Everything bound, for showing the user what they can do."""

        return {**self.poses, **self.swipes}
