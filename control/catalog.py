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
#: A keystroke job lists the combo for each app that differs; "any" is
#: the fallback when an app is not named.
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
}


def job_names():
    """The jobs a person can pick from, in the order shown."""

    return list(JOBS)


def apps_for(job_name):
    """Which apps this job knows a key for, for the third box.

    A built-in job needs no app (it is system-wide), so it returns just
    "any".  A keystroke job returns the apps it has keys for.
    """

    entry = JOBS.get(job_name)

    if entry is None or entry["kind"] == "builtin":
        return ["any"]

    return list(entry["keys"])


def resolve(job_name, app="any"):
    """The action a named job becomes, for a chosen app.

    Returns a single action dict (the same shape control/actions uses),
    or None if the job is unknown.  A keystroke job with no key for the
    chosen app falls back to its "any" key, and if it has none, to None
    -- the form then asks the person to type the shortcut instead.
    """

    entry = JOBS.get(job_name)

    if entry is None:
        return None

    if entry["kind"] == "builtin":
        return {"type": "builtin", "command": entry["command"]}

    keys = entry["keys"]
    combo = keys.get(app) or keys.get("any")

    if not combo:
        return None

    return {"type": "keystroke", "combo": combo}
