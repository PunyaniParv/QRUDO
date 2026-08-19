"""The connection between a named job and the action that does it.

"Next track for YouTube Music" is not a puzzle an AI has to solve -- it
is Shift+N, a fact.  This is the table of those facts: common jobs a
person names, and the action each one turns into, per app where the app
matters.  It is what fills the first box of the add-a-gesture form, so
someone picks "Next track" rather than knowing the shortcut, and no
model is asked to guess what any word means.

Anything not in the table is still reachable: the form lets a person
type a shortcut, or choose open-app / open-file / open-website
directly.  The catalog is the convenience, not the ceiling.

A job that is a keystroke resolves differently per app, because the
same job is a different key on each site.  A job that opens something,
or is a built-in QRUDO command, is the same everywhere.
"""

from __future__ import annotations

#: The apps whose shortcuts we know, by a short key used below.  The
#: display name is what the third box shows and what the target matches.
APPS = {
    "youtube": "YouTube (in a browser)",
    "youtube_music": "YouTube Music",
    "spotify": "Spotify",
    "any": "Any app",
}

#: The catalog: a job name -> how it resolves.  Each entry is either
#:   {"kind": "keystroke", "keys": {app: combo, ...}}   -- per-app keys
#:   {"kind": "builtin", "command": "VOLUME_UP"}         -- a QRUDO action
#:   {"kind": "open_app", "app": "Google Chrome"}        -- launch an app
#: A keystroke job lists the combo for each app that differs; "any" is
#: the fallback when an app is not named.  A builtin or open_app job is
#: the same everywhere and needs no per-app keys.
JOBS = {
    "Play / pause": {"kind": "builtin", "command": "PLAY_PAUSE"},
    "Volume up": {"kind": "builtin", "command": "VOLUME_UP"},
    "Volume down": {"kind": "builtin", "command": "VOLUME_DOWN"},
    "Rewind": {"kind": "builtin", "command": "REWIND"},
    "Forward": {"kind": "builtin", "command": "FORWARD"},
    "Brightness up": {"kind": "builtin", "command": "BRIGHTNESS_UP"},
    "Brightness down": {"kind": "builtin", "command": "BRIGHTNESS_DOWN"},

    "Next track": {"kind": "keystroke", "keys": {
        "youtube_music": "shift+n",
        "youtube": "shift+n",
        "spotify": "cmd+right",
        "any": "shift+n",
    }},
    "Previous track": {"kind": "keystroke", "keys": {
        "youtube_music": "shift+p",
        "youtube": "shift+p",
        "spotify": "cmd+left",
        "any": "shift+p",
    }},
    "Full screen": {"kind": "keystroke", "keys": {
        "youtube": "f", "youtube_music": "f", "any": "f",
    }},
    "Mute (in the app)": {"kind": "keystroke", "keys": {
        "youtube": "m", "youtube_music": "m", "any": "m",
    }},
    "New tab": {"kind": "keystroke", "keys": {"any": "cmd+t"}},
    "Close tab": {"kind": "keystroke", "keys": {"any": "cmd+w"}},

    "Open Chrome": {"kind": "open_app", "app": "Google Chrome"},
}


def job_names():
    """The jobs a person can pick from, in the order shown."""

    return list(JOBS)


def apps_for(job_name):
    """Which apps this job knows a key for, for the third box.

    A built-in or open-app job needs no app (one is system-wide, the
    other always launches the same app), so it returns just "any".  A
    keystroke job returns the apps it has keys for.
    """

    entry = JOBS.get(job_name)

    if entry is None or entry["kind"] in ("builtin", "open_app"):
        return ["any"]

    return list(entry["keys"])


#: The app each catalog key belongs to, as its real macOS app name, so
#: a keystroke locked to it can be delivered to that app by name.  A
#: browser-site job ("youtube") has no app of its own -- it lives in a
#: browser -- so it stays unlocked and rides the front window.
APP_NAMES = {
    "youtube_music": "YouTube Music",
    "spotify": "Spotify",
}


def resolve(job_name, app="any", lock_to_app=False):
    """The action a named job becomes, for a chosen app.

    Returns a single action dict (the same shape control/actions uses),
    or None if the job is unknown.  A keystroke job with no key for the
    chosen app falls back to its "any" key, and if it has none, to None
    -- the form then asks the person to type the shortcut instead.  A
    builtin job maps to its QRUDO command; an open_app job names the app
    to launch.

    ``lock_to_app`` is the "global trigger, land it in one app" wish: a
    keystroke is tagged with the app's real name so it is delivered
    there whatever window is focused.  Only apps with a known real name
    (a real application, not a browser site) can be locked; a site job
    stays on the front window because it has no app of its own.
    """

    # Case- and space-insensitive, so a typed "next track" finds "Next
    # track" without the person matching the menu's exact casing.
    entry = JOBS.get(job_name)

    if entry is None:
        folded = job_name.strip().lower()
        entry = next((v for k, v in JOBS.items()
                      if k.lower() == folded), None)

    if entry is None:
        return None

    if entry["kind"] == "builtin":
        # Built-ins are system-wide already; the target is the config's
        # job, not this action's.
        return {"type": "builtin", "command": entry["command"]}

    if entry["kind"] == "open_app":
        # Launches one app, the same everywhere -- no per-app keys.
        return {"type": "open_app", "app": entry["app"]}

    keys = entry["keys"]
    combo = keys.get(app) or keys.get("any")

    if not combo:
        return None

    action = {"type": "keystroke", "combo": combo}

    if lock_to_app and app in APP_NAMES:
        action["target_app"] = APP_NAMES[app]

    return action
