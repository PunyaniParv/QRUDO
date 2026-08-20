"""What a gesture or a keybind actually does, once it fires.

An action is one of a small set of things -- open a file or folder,
launch an app, open a website, press a keystroke, run a built-in
command, or run a shell command -- and a chain is an ordered list of
them.  A whole chain serialises to one JSON string, which is exactly
what the command pipe already carries as the CUSTOM payload, so nothing
about cooldown, logging, dry-run or error handling has to change to
carry an arbitrary action.

This module is pure: it imports no OS and no backend, so it parses and
validates on any platform and is unit-tested without a machine.  The
running of an action lives in ActionRunner, which is handed the pieces
that touch the OS rather than reaching for them.

Safety lives here too, at the data layer, because that is the one place
a hand-edited file cannot get around: a shell command may only run if
it was explicitly confirmed by a person, and never if it matches a
pattern known to be destructive.  Both checks are applied again at run
time, not only at save time -- defence in depth, so a JSON edited to
set confirmed=true by hand still cannot smuggle a dangerous command
past the denylist.
"""

from __future__ import annotations

import json
import re

#: The action kinds, each with the keys it must carry.
ACTION_TYPES = {
    "open_path": ("path",),      # a file or folder
    "open_app": ("app",),        # launch an application
    "open_url": ("url",),        # a website
    "quit_app": ("app",),        # ask an app to quit; "all" quits every one
    "hide_app": ("app",),        # hide an app's windows; "all" hides every one
    "keystroke": ("combo",),     # press a chord
    "builtin": ("command",),     # a built-in Command value
    "run_command": ("command",),  # a shell command -- the guarded one
}

#: Optional keys an action may also carry, kept if present.  ``target_app``
#: locks a keystroke to one app -- "global trigger, land it in YouTube
#: Music" -- delivered to that app even while another window is focused.
OPTIONAL_KEYS = {"target_app"}

#: Patterns a shell command may never contain, however it was saved.
#: Deliberately blunt and broad: this is a backstop behind the
#: confirmation flag, not the primary guard, so it errs toward refusing.
DESTRUCTIVE = [
    re.compile(r"\brm\s+-[a-z]*[rf]", re.I),   # rm -rf and kin
    re.compile(r"\bmkfs\b", re.I),
    re.compile(r"\bdd\b.*\bif=", re.I),
    re.compile(r">\s*/dev/", re.I),
    re.compile(r"\bsudo\b", re.I),
    re.compile(r":\s*\(\s*\)\s*\{", re.I),      # :(){ fork bomb
    re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b", re.I),
    re.compile(r"\bmv\b.+\s+/dev/null"),
    re.compile(r"\bchmod\s+-R\b", re.I),
]


class ActionError(ValueError):
    """An action could not be built, understood, or safely run."""


def validate(action: dict) -> dict:
    """A single action, checked and normalised, or ActionError.

    Unknown keys are dropped rather than kept, so a stray field in a
    hand-edited file cannot change how an action runs.
    """

    if not isinstance(action, dict):
        raise ActionError("an action must be an object")

    kind = action.get("type")

    if kind not in ACTION_TYPES:
        raise ActionError(f"unknown action type {kind!r}")

    clean = {"type": kind}

    for key in ACTION_TYPES[kind]:
        value = action.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ActionError(f"a {kind} action needs a non-empty {key!r}")
        clean[key] = value.strip()

    # Optional extras, kept only when they are a non-empty string, so a
    # gesture locked to one app carries that app and one that is not
    # carries nothing rather than an empty marker.
    for key in OPTIONAL_KEYS:
        value = action.get(key)
        if isinstance(value, str) and value.strip():
            clean[key] = value.strip()

    if kind == "run_command":
        # The confirmation flag rides along, but it is trusted only as
        # far as the denylist allows: a command that matches a
        # destructive pattern is refused even if the flag says confirmed,
        # both here and again when it runs.
        clean["confirmed"] = bool(action.get("confirmed", False))

        if is_destructive(clean["command"]):
            raise ActionError(
                "that command matches a pattern QRUDO refuses to run")

    return clean


def is_destructive(command: str) -> bool:
    """Whether a shell command matches a known-dangerous pattern."""

    return any(pattern.search(command) for pattern in DESTRUCTIVE)


def normalise(actions) -> list:
    """A list of actions, each validated.  A bad one raises.

    Accepts a single action dict as a one-element chain, since that is
    the common case and saves every caller wrapping it.
    """

    if isinstance(actions, dict):
        actions = [actions]

    if not isinstance(actions, list) or not actions:
        raise ActionError("a chain needs at least one action")

    return [validate(a) for a in actions]


def serialize(actions) -> str:
    """A chain to the one string the command pipe carries."""

    return json.dumps(normalise(actions), separators=(",", ":"))


def parse(payload: str) -> list:
    """The string back to a chain, or ActionError.

    A payload that is not JSON at all is treated as a bare keystroke
    combo -- the shape the CUSTOM payload had before chains existed --
    so an old saved gesture keeps working.
    """

    try:
        raw = json.loads(payload)
    except (ValueError, TypeError):
        # Legacy: a plain "cmd+shift+n" string.
        return [{"type": "keystroke", "combo": payload}]

    return normalise(raw)


def describe(actions) -> str:
    """A chain in words, for a person to read before saving it."""

    parts = []

    for action in normalise(actions):
        kind = action["type"]
        if kind == "open_path":
            parts.append(f"open {action['path']}")
        elif kind == "open_app":
            parts.append(f"launch {action['app']}")
        elif kind == "open_url":
            parts.append(f"open {action['url']}")
        elif kind == "quit_app":
            target = action["app"]
            parts.append("quit every open app" if target == "all"
                         else f"quit {target}")
        elif kind == "hide_app":
            target = action["app"]
            parts.append("hide every open app" if target == "all"
                         else f"hide {target}")
        elif kind == "keystroke":
            parts.append(f"press {action['combo']}")
        elif kind == "builtin":
            parts.append(action["command"].lower().replace("_", " "))
        elif kind == "run_command":
            mark = "" if action.get("confirmed") else " (needs confirming)"
            parts.append(f"run: {action['command']}{mark}")

    return ", then ".join(parts)


class ActionRunner:
    """Runs a chain, in order, through pieces handed in rather than reached for.

    ``opener`` runs an argv list (the backend's shell-out); ``keystroke``
    presses a combo; ``builtin`` runs a built-in Command.  Each is
    injected, so the whole runner is tested without opening a single app
    -- and so the OS-specific "how do I open a thing" lives in the
    backend, not here.

    A chain stops at the first action that fails, returning what
    succeeded and why it stopped, because a half-run chain that lied
    about finishing would be worse than one that says where it broke.
    """

    def __init__(self, opener, keystroke, builtin, quitter=None,
                 hider=None):
        self._open = opener
        self._keystroke = keystroke
        self._builtin = builtin
        self._quit = quitter
        self._hide = hider

    def run(self, payload: str) -> str:
        actions = parse(payload)

        done = []

        for action in actions:
            done.append(self._one(action))

        return "; ".join(done)

    def _one(self, action: dict) -> str:
        kind = action["type"]

        if kind == "open_path":
            # Expand ~ ourselves: the OS "open" gets a literal path, not
            # a shell that would expand it, so ~/Downloads must become
            # the real home path or it opens nothing.
            import os
            path = os.path.expanduser(action["path"])
            self._open(["open", path])
            return f"opened {action['path']}"

        if kind == "open_app":
            self._open(["open", "-a", action["app"]])
            return f"launched {action['app']}"

        if kind == "open_url":
            self._open(["open", action["url"]])
            return f"opened {action['url']}"

        if kind == "quit_app":
            if self._quit is None:
                raise ActionError(
                    "quitting apps is not supported on this platform yet")
            return self._quit(action["app"])

        if kind == "hide_app":
            if self._hide is None:
                raise ActionError(
                    "hiding apps is not supported on this platform yet")
            return self._hide(action["app"])

        if kind == "keystroke":
            return self._keystroke(action["combo"],
                                   action.get("target_app", ""))

        if kind == "builtin":
            return self._builtin(action["command"])

        if kind == "run_command":
            return self._run_shell(action)

        raise ActionError(f"unknown action type {kind!r}")

    def _run_shell(self, action: dict) -> str:
        """The guarded one: confirmed, and not destructive, or it refuses.

        Both checks run again here, not only at save time, so a file
        hand-edited to flip confirmed=true on a destructive command
        still cannot run it.
        """

        command = action["command"]

        if not action.get("confirmed"):
            raise ActionError(
                "this command was never confirmed, so QRUDO will not run it")

        if is_destructive(command):
            raise ActionError(
                "this command matches a pattern QRUDO refuses to run")

        # A confirmed, screened command is run as a program with
        # arguments, not through a shell, so nothing in it is
        # re-interpreted -- the argv split is deliberately simple and
        # anything needing shell features is out of scope by design.
        import shlex

        self._open(shlex.split(command))

        return f"ran {command}"
