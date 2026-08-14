"""Screen brightness.

Kept separate because support differs most here: macOS drives the built-in
display through a private framework, Windows goes through WMI and only for
laptop panels, and external monitors on either platform usually cannot be
driven at all.  A backend that cannot do it reports UNSUPPORTED rather
than failing silently.
"""

from __future__ import annotations

from .commands import Command


def handlers(controller, config):
    """Command -> the call that performs it."""

    return {
        Command.BRIGHTNESS_UP:
            lambda: controller.brightness_up(config.brightness_step),
        Command.BRIGHTNESS_DOWN:
            lambda: controller.brightness_down(config.brightness_step),
    }
