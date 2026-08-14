"""Playback: play/pause and seeking.

Seeking has no system-wide key on either platform, so it is sent as arrow
keys -- which is why it is the only area that cares which window is
focused.  ``seek_target_app`` in the config aims it at a named app
instead.
"""

from __future__ import annotations

from .commands import Command


def handlers(controller, config):
    """Command -> the call that performs it."""

    return {
        Command.PLAY_PAUSE:
            controller.play_pause,
        Command.REWIND:
            lambda: controller.rewind(config.seek_seconds),
        Command.FORWARD:
            lambda: controller.forward(config.seek_seconds),
    }
