"""The menu bar presence: a white Q on the right side, always there.

QRUDO lives as an AGENT on macOS -- no Dock icon, no cmd-tab entry --
because a gesture engine is a background service, not a window you
visit.  What remains visible is one mark in the menu bar: the Q,
drawn as a template image so the system paints it white on a dark
menu bar and dark on a light one, with a small menu behind it (show
the window, quit).

It rides the SAME process and the same main thread as the window:
Tk on macOS pumps the Cocoa run loop, which is what delivers the
status item's clicks, so no second loop and no second thread exist.
Everything here is best-effort -- a machine where AppKit is not
importable just has no menu bar mark, and the window still works.
"""

from __future__ import annotations

import sys
from pathlib import Path


def install(app):
    """Put the Q in the menu bar; returns True when it is there."""

    if sys.platform != "darwin":
        return False

    try:
        import objc
        from AppKit import (NSImage, NSMenu, NSMenuItem, NSObject,
                            NSStatusBar, NSVariableStatusItemLength)
    except Exception:
        return False

    base = getattr(sys, "_MEIPASS", None)
    icon_path = (Path(base) / "assets" / "menubar.png") if base else \
        Path(__file__).resolve().parent.parent / "assets" / "menubar.png"

    class _MenuTarget(NSObject):
        """The menu's callbacks.

        They do the least possible thing: drop a word in a mailbox the
        window's own pulse reads every beat.  Nothing Cocoa-side ever
        touches Tk directly, and even an exotic dispatch context
        cannot lose a plain attribute write.  Each click is also
        logged, so "the buttons do nothing" always has a diagnosis:
        either the click never arrived (no log line) or the pulse
        never collected it (log line, no effect).
        """

        def initWithApp_(self, tk_app):
            self = objc.super(_MenuTarget, self).init()
            if self is None:
                return None
            self._app = tk_app
            return self

        def showWindow_(self, sender):
            from control import log as _log
            _log.get_logger("ui").info("ui: menu bar click -- show")
            self._app._menubar_intent = "show"

        def quitApp_(self, sender):
            from control import log as _log
            _log.get_logger("ui").info("ui: menu bar click -- quit")
            self._app._menubar_intent = "quit"

    try:
        # Pin the Q as the RIGHTMOST app icon -- right up against the
        # system's own zone (the camera pill, control centre, battery),
        # which is the closest any app is allowed to sit: everything
        # from the camera pill rightward is Apple's, for every app.
        # A tiny preferred distance from the right edge gets clamped to
        # the nearest legal spot; set only once, so a hand-dragged
        # position is respected afterwards.
        from Foundation import NSUserDefaults

        defaults = NSUserDefaults.standardUserDefaults()
        key = "NSStatusItem Preferred Position QRUDO"
        if defaults.objectForKey_(key) is None:
            defaults.setFloat_forKey_(40.0, key)

        bar = NSStatusBar.systemStatusBar()
        item = bar.statusItemWithLength_(NSVariableStatusItemLength)
        item.setAutosaveName_("QRUDO")

        image = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
        if image is None:
            return False
        image.setSize_((18, 18))
        image.setTemplate_(True)     # white on dark bars, dark on light
        item.button().setImage_(image)
        item.button().setToolTip_("QRUDO -- gestures are running")

        target = _MenuTarget.alloc().initWithApp_(app)

        menu = NSMenu.alloc().init()

        show = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Show QRUDO", "showWindow:", "")
        show.setTarget_(target)
        menu.addItem_(show)

        menu.addItem_(NSMenuItem.separatorItem())

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit QRUDO", "quitApp:", "")
        quit_item.setTarget_(target)
        menu.addItem_(quit_item)

        item.setMenu_(menu)

        # Keep every ObjC object alive for the life of the app: a
        # garbage-collected status item vanishes from the bar.
        app._menubar = (item, menu, target, image)

        from control import log as _log
        _log.get_logger("ui").info("ui: menu bar mark standing")

        return True
    except Exception as exc:
        try:
            from control import log as _log
            _log.get_logger("ui").error(
                "ui: menu bar mark failed: %s", exc)
        except Exception:
            pass
        return False
