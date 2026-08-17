"""Which app the targeted commands should land in.

Most commands are system-wide, but play/pause and seeking have to be
delivered *somewhere*.  ``target_app`` in the config names that
somewhere once and for all; this module makes it a decision that can
also be made freshly -- from what is running, what is playing and what
is focused -- and changed with a gesture or a chord without touching a
file.

The resolver never guesses silently.  Every switch comes back as a
command result the overlay shows, and naming an app in the config pins
it and disables the guessing entirely, so the old contract stays one
config line away.

Resolution, in auto mode, prefers in order: the focused app if it is a
candidate (you are looking at it), then a player that says it is
playing, then the app the config prefers, then whatever candidate is
left.  Ties are broken by that order rather than by guessing, and the
answer is always visible before the next fist lands.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

from .commands import Command

#: How often the background refresh asks the OS what is around.
POLL_SECONDS = 2.0

#: Players that can be asked "are you playing" on macOS, and whose
#: window titles mark them as candidates on Windows.
PLAYERS = ("Spotify", "Music", "VLC", "QuickTime Player", "TV")

#: Browsers cannot be asked; being open -- or focused -- is their
#: evidence.
BROWSERS = ("Google Chrome", "Safari", "Arc", "Firefox", "Microsoft Edge")

#: The config value (and cycle stop) that means "work it out".
AUTO = "auto"


class TargetResolver:
    """Keeps ``config.target_app`` pointed at the right app.

    ``choice`` is the user's word: None means auto, a name means pinned
    -- seeded from the config, moved by ``cycle``.  ``probe`` is a
    callable returning what the OS can see; injectable so every test is
    hermetic and the platform split lives in one place.
    """

    def __init__(self, config, probe=None):
        self.config = config
        self.probe = probe if probe is not None else _platform_probe

        configured = (config.target_app or config.seek_target_app).strip()
        self.preferred = "" if configured.lower() == AUTO else configured
        self.choice = self.preferred or None

        self.candidates: list[str] = []
        self.playing: list[str] = []
        self.frontmost: str | None = None

        self._seen_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- looking around -----------------------------------------------------

    def refresh(self, force=False):
        """Ask the OS what is around, at most once per poll interval.

        A probe that fails answers with nothing rather than an
        exception: resolution then falls back to the configured
        preference, which is exactly the old behaviour.
        """

        now = time.monotonic()

        if not force and now - self._seen_at < POLL_SECONDS:
            return

        try:
            seen = self.probe() or {}
        except Exception:
            seen = {}

        with self._lock:
            self._seen_at = now
            self.frontmost = seen.get("frontmost")
            running = [name for name in seen.get("running", ())]
            self.playing = [name for name in seen.get("playing", ())]

            ordered: list[str] = []

            def note(name):
                if name and name not in ordered:
                    ordered.append(name)

            if self.frontmost in running:
                note(self.frontmost)

            for name in self.playing:
                note(name)

            for name in running:
                note(name)

            note(self.preferred)

            self.candidates = ordered

    def resolved(self):
        """The app targeted right now, or "" for nobody in particular."""

        if self.choice:
            return self.choice

        if self.frontmost and self.frontmost in self.candidates:
            return self.frontmost

        if self.playing:
            return self.playing[0]

        if self.preferred:
            return self.preferred

        return self.candidates[0] if self.candidates else ""

    def apply(self):
        """Point the config at the resolution, where the backends read."""

        self.config.target_app = self.resolved()

    # -- the user's word ----------------------------------------------------

    def cycle(self, step):
        """Move the pin through auto and every candidate.  Returns the
        detail string the overlay shows, so a switch is always seen."""

        self.refresh(force=True)

        options = [AUTO] + list(self.candidates)
        current = self.choice or AUTO

        if current not in options:
            options.append(current)

        picked = options[(options.index(current) + step) % len(options)]
        self.choice = None if picked == AUTO else picked

        self.apply()

        if picked == AUTO:
            landing = self.resolved()
            return f"target -> auto ({landing})" if landing \
                else "target -> auto (nobody playing yet)"

        return f"target -> {picked}"

    # -- the background refresh --------------------------------------------

    def start(self):
        """Poll on a worker thread, so the camera loop never waits."""

        if self._thread is not None:
            return

        def keep_fresh():
            while not self._stop.wait(POLL_SECONDS):
                self.refresh(force=True)

                if self.choice is None:
                    self.apply()

        self._thread = threading.Thread(
            target=keep_fresh, name="qrudo-targets", daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return

        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None


def handlers(resolver):
    """Command -> the call that performs it, engine-side."""

    return {
        Command.TARGET_NEXT: lambda: resolver.cycle(+1),
        Command.TARGET_PREV: lambda: resolver.cycle(-1),
    }


# ---------------------------------------------------------------- probes

def _platform_probe():
    if sys.platform == "darwin":
        return _macos_probe()

    if sys.platform == "win32":
        return _windows_probe()

    return {}


def _macos_probe():
    """Ask what is running, playing, and focused.

    Two rounds, deliberately.  The first asks only System Events, which
    every Mac can answer.  The "are you playing" question goes to each
    player in its own little script, and only once it is known to be
    running -- for two reasons: telling an application anything launches
    it, and the first gesture of a demo must not open Music; and merely
    *compiling* a tell block needs the app's dictionary, so one script
    naming Spotify would fail whole on a machine without Spotify and
    take the entire probe with it.
    """

    who = """
    tell application "System Events"
        set frontName to name of first application process whose frontmost is true
        set runningNames to name of every application process
    end tell
    set report to "front|" & frontName
    repeat with candidate in runningNames
        set report to report & linefeed & "running|" & candidate
    end repeat
    return report
    """

    try:
        answer = subprocess.run(
            ["osascript", "-e", who],
            capture_output=True, text=True, timeout=3.0,
        ).stdout
    except Exception:
        return {}

    seen = {"running": [], "playing": []}
    known = PLAYERS + BROWSERS

    for line in answer.splitlines():
        kind, _, name = line.partition("|")
        name = name.strip()

        if not name:
            continue

        if kind == "front":
            seen["frontmost"] = name
        elif kind == "running" and name in known:
            seen["running"].append(name)

    for player in ("Spotify", "Music"):
        if player not in seen["running"]:
            continue

        ask = f'tell application "{player}" to player state as text'

        try:
            state = subprocess.run(
                ["osascript", "-e", ask],
                capture_output=True, text=True, timeout=2.0,
            ).stdout.strip().lower()
        except Exception:
            continue

        if state == "playing":
            seen["playing"].append(player)

    return seen


def _windows_probe():
    """Window titles, which is what Windows targeting matches anyway.

    No playing signal here yet: that needs the system media-session API
    and a package this project does not depend on.  Candidates come
    from visible window titles carrying a known player or browser name,
    so auto mode degrades to focused-else-configured -- the safe half
    of the behaviour -- rather than to a guess.
    """

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)

        titles = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)

                if length:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    titles.append(buffer.value)

            return True

        user32.EnumWindows(visit, 0)

        front = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(user32.GetForegroundWindow(), front, 256)

        def known(title):
            for name in PLAYERS + BROWSERS:
                if name.lower() in title.lower():
                    return name

            return None

        running = []

        for title in titles:
            name = known(title)

            if name and name not in running:
                running.append(name)

        return {
            "frontmost": known(front.value),
            "running": running,
            "playing": [],
        }
    except Exception:
        return {}
