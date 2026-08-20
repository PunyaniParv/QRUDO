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
        """The menu's callbacks, marshalled onto Tk's own scheduler --
        the safe place to touch the window from."""

        def initWithApp_(self, tk_app):
            self = objc.super(_MenuTarget, self).init()
            if self is None:
                return None
            self._app = tk_app
            return self

        def showWindow_(self, sender):
            self._app.root.after(0, self._app.show_from_menubar)

        def quitApp_(self, sender):
            self._app.root.after(0, self._app.quit)

    try:
        bar = NSStatusBar.systemStatusBar()
        item = bar.statusItemWithLength_(NSVariableStatusItemLength)

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

        return True
    except Exception:
        return False
