"""Gesture names in, commands out.

This is the entire contract between the two halves.  The vision side
returns strings and knows nothing about the machine; the control side
takes commands and knows nothing about cameras.  Changing what a gesture
does is a change to this file alone.
"""

from __future__ import annotations

import time

from control import Command

#: How soon any command may follow any other, whichever gestures they
#: came from.  One movement crosses many frames and a hand settling after
#: one gesture passes through others, so something has to say that a
#: person who just did a thing is not immediately doing another.
#:
#: It lives here, where gestures become commands, rather than in the
#: detector.  The detector has its own -- one shared by the four
#: directions, which is why they have never trodden on each other -- but
#: a gesture added to the table below inherits nothing from it unless it
#: happens to arrive by the same route.  Here, everything inherits it,
#: including whatever is added next.
GLOBAL_COOLDOWN = 1.0

#: How soon the same held pose may fire again.  Making a fist twice in
#: under a second is not something anyone does deliberately, and a
#: misreading that flickers to another gesture and back is: the flicker
#: re-arms the pose, so it fires on the way in and again on the way back,
#: playing and pausing at once.  Only a misreading is refused here.
POSE_REPEAT = 1.0

#: Held poses.  These fire once when you make them, not repeatedly while
#: you hold them -- see GestureRouter.
#:
#: POINT cycles which app the targeted commands land in.  Pointing at
#: the thing you want to control is the gesture's own meaning, and a
#: misread costs routing rather than playback -- the switch is shown on
#: the overlay before any command follows it -- which is why the free
#: pose went to this and not to another command that acts.
POSE_COMMANDS = {
    "FIST": Command.PLAY_PAUSE,
    "POINT": Command.TARGET_NEXT,
}

#: How long a pose must be held before it counts, per pose.  POINT is
#: what a hand looks like on its way into and out of the two-finger
#: pose -- the index leads, the middle follows a beat later -- and the
#: day it was bound, every swipe pose fired a phantom target switch on
#: the way in, whose cooldown then swallowed the swipe itself.  Half a
#: second is several times any transition and nothing to somebody
#: actually pointing.  The fist stays instant: nothing passes through a
#: fist on its way to anything.
POSE_DWELL = {
    "POINT": 0.5,
}

#: Movements, each on its own pose so that raising a hand mid-seek cannot
#: be read as volume.  These are already one-off events with their own
#: cooldown.
SWIPE_COMMANDS = {
    "SWIPE_LEFT": Command.REWIND,
    "SWIPE_RIGHT": Command.FORWARD,
    "SWIPE_UP": Command.VOLUME_UP,
    "SWIPE_DOWN": Command.VOLUME_DOWN,
    "PALM_UP": Command.BRIGHTNESS_UP,
    "PALM_DOWN": Command.BRIGHTNESS_DOWN,
}

# Deliberately unmapped as held poses:
#
#   TWO_FINGER  is the swipe pose.  Binding it would fire a command every
#               time you got ready to swipe.
#   OPEN_PALM   is the pose the brightness lifts are made from, and what
#               a hand looks like on its way to and from every other
#               gesture besides -- so it would fire constantly.
#
# Volume had a gesture before these, a pinch raised or lowered, and it
# was more trouble than the other four together: being defined by where
# the thumb is, it was mistaken in turn for a fist, an open hand and two
# fingers, because the thumb is the landmark a camera loses first.  The
# two fingers raised and lowered above are the same wish granted on
# landmarks the camera keeps.  The keyboard still carries every command
# -- ctrl+alt+U/D/P/L/R/B/N via --hotkeys -- but as demo insurance, not
# as the home of any of them.


class GestureRouter:
    """Turn a stream of gesture names into commands.

    Held poses fire on the way in and then stay quiet.  Without that, a
    fist held for two seconds would toggle play/pause thirty times,
    because the vision side reports what your hand *is* on every frame,
    not what changed.
    """

    def __init__(self, poses=None, swipes=None, repeat=POSE_REPEAT,
                 cooldown=GLOBAL_COOLDOWN, dwell=None):
        self.poses = POSE_COMMANDS if poses is None else poses
        self.swipes = SWIPE_COMMANDS if swipes is None else swipes
        self.repeat = repeat
        self.cooldown = cooldown
        self.dwell = POSE_DWELL if dwell is None else dwell
        self._held = None
        self._held_since = 0.0
        self._fired_this_hold = False
        self._fired_at = {}
        self._last_command_at = -1e9

    def update(self, gesture=None, swipe=None, now=None):
        """Return the command this frame should run, or None."""

        now = time.time() if now is None else now

        if now - self._last_command_at < self.cooldown:
            # Still counting down from the last one.  The pose is dropped
            # as well, so that a hand held through the wait does not fire
            # the moment it ends -- it has to be made again.
            self._held = None
            return None

        # A swipe is a movement that already happened, so it always counts.
        if swipe is not None and swipe in self.swipes:
            # Whatever pose was being held was part of the swipe; make it
            # ask again rather than firing as the hand settles.
            self._held = None
            return self._fire(self.swipes[swipe], now)

        # "UNKNOWN" is not a gesture, it is the vision side saying it is
        # unsure -- which happens for a frame or two whenever the picture is
        # poor, and constantly on a slow machine.  Treating it as a change
        # re-arms the pose, so a fist flickering out and back fires twice:
        # play, then pause, and nothing appears to have happened.
        if gesture in (None, "UNKNOWN"):
            return None

        if gesture != self._held:
            self._held = gesture
            self._held_since = now
            self._fired_this_hold = False

        command = self.poses.get(gesture)

        if command is None or self._fired_this_hold:
            return None

        # Some poses are also what a hand looks like on its way to a
        # different pose: the index leads into the two-finger pose, and
        # for those frames the hand is honestly pointing.  A pose with a
        # dwell only counts once it has been held -- a transition passes
        # through in a fraction of that, and a pose that is meant is
        # held without noticing the wait.
        if now - self._held_since < self.dwell.get(gesture, 0.0):
            return None

        self._fired_this_hold = True

        if now - self._fired_at.get(gesture, -1e9) < self.repeat:
            return None

        self._fired_at[gesture] = now

        return self._fire(command, now)

    def _fire(self, command, now):
        """Note when this went out, so the next one has to wait."""

        self._last_command_at = now

        return command

    def forget(self):
        """Hand left the frame; the next pose is a new one.

        The hand leaving is deliberate enough to clear both guards:
        making a fist, dropping your hand, and making another is two
        gestures however quickly it is done.  The cooldown is there to
        stop one movement counting twice, and a hand that left and came
        back is not one movement.
        """

        self._held = None
        self._fired_this_hold = False
        self._fired_at.clear()
        self._last_command_at = -1e9

    def mapping(self):
        """Everything bound, for showing the user what they can do."""

        return {**self.poses, **self.swipes}
