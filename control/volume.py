"""Volume: which commands exist here, and how they reach the machine.

The three device modules -- this, media.py and brightness.py -- own the
mapping from a command to a call, along with the settings that shape it.
Adding a command to an area means editing that area's file and nothing
else.

How the call is actually made is the backend's business, and that is split
by platform rather than by device on purpose.  Volume on macOS is
CoreAudio through ctypes; on Windows it is a virtual key code. Those have
nothing in common but the name, so grouping them here would mean every
device module carrying its own "which OS is this?" branch.
"""

from __future__ import annotations

from .commands import Command


def handlers(controller, config):
    """Command -> the call that performs it."""

    return {
        Command.VOLUME_UP:
            lambda: controller.volume_up(config.volume_step),
        Command.VOLUME_DOWN:
            lambda: controller.volume_down(config.volume_step),
    }
