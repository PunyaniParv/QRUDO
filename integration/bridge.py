"""Gesture names in, commands out.

This is the entire contract between the two halves.  The vision side
returns strings and knows nothing about the machine; the control side
takes commands and knows nothing about cameras.  Changing what a gesture
does is a change to this file alone.
"""

from __future__ import annotations

import time

from control import Command

#: How soon the same held pose may fire again.  Making a fist twice in
#: under a second is not something anyone does deliberately, and a
#: misreading that flickers to another gesture and back is: the flicker
#: re-arms the pose, so it fires on the way in and again on the way back,
#: playing and pausing at once.  Only a misreading is refused here.
POSE_REPEAT = 1.0

#: Held poses.  These fire once when you make them, not repeatedly while
#: you hold them -- see GestureRouter.
POSE_COMMANDS = {
    "FIST": Command.PLAY_PAUSE,
}

#: Movements, each on its own pose so that raising a hand mid-seek cannot
#: be read as volume.  These are already one-off events with their own
#: cooldown.
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
#   POINT       is free.
#
# Volume and brightness are on the keyboard -- ctrl+alt+U and D, B and N.
# Volume had a gesture, a pinch raised or lowered, and it was more trouble
# than the other four together: being defined by where the thumb is, it
# was mistaken in turn for a fist, an open hand and two fingers, because
# the thumb is the landmark a camera loses first.  Three gestures that
# cannot be confused with each other beat five that can.


class GestureRouter:
    """Turn a stream of gesture names into commands.

    Held poses fire on the way in and then stay quiet.  Without that, a
    fist held for two seconds would toggle play/pause thirty times,
    because the vision side reports what your hand *is* on every frame,
    not what changed.
    """

    def __init__(self, poses=None, swipes=None, repeat=POSE_REPEAT):
        self.poses = POSE_COMMANDS if poses is None else poses
        self.swipes = SWIPE_COMMANDS if swipes is None else swipes
        self.repeat = repeat
        self._held = None
        self._fired_at = {}

    def update(self, gesture=None, swipe=None, now=None):
        """Return the command this frame should run, or None."""

        now = time.time() if now is None else now

        # A swipe is a movement that already happened, so it always counts.
        if swipe is not None and swipe in self.swipes:
            # Whatever pose was being held was part of the swipe; make it
            # ask again rather than firing as the hand settles.
            self._held = None
            return self.swipes[swipe]

        # "UNKNOWN" is not a gesture, it is the vision side saying it is
        # unsure -- which happens for a frame or two whenever the picture is
        # poor, and constantly on a slow machine.  Treating it as a change
        # re-arms the pose, so a fist flickering out and back fires twice:
        # play, then pause, and nothing appears to have happened.
        if gesture in (None, "UNKNOWN"):
            return None

        if gesture == self._held:
            return None

        self._held = gesture

        command = self.poses.get(gesture)

        if command is None:
            return None

        if now - self._fired_at.get(gesture, -1e9) < self.repeat:
            return None

        self._fired_at[gesture] = now

        return command

    def forget(self):
        """Hand left the frame; the next pose is a new one.

        The hand leaving is deliberate enough to clear the repeat guard as
        well: making a fist, dropping your hand, and making another is two
        gestures however quickly it is done.
        """

        self._held = None
        self._fired_at.clear()

    def mapping(self):
        """Everything bound, for showing the user what they can do."""

        return {**self.poses, **self.swipes}
