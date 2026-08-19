"""Turning a typed phrase into an action, by plain rules -- no AI.

The first box of the add-a-gesture form takes a typed phrase as well as
a menu pick.  A phrase like "open Downloads" or "launch Spotify" or "go
to gmail.com" follows a handful of patterns that a few rules read
perfectly, instantly, offline -- which is faster and surer than asking
a model to guess what a short, plain sentence means.

What the rules cannot place, they say so about, and the form then lets
the person pick an action type by hand.  The rules are the
convenience; the menu behind them is the floor.  This is the whole of
"our own logic linking words to work" that the user asked for.
"""

from __future__ import annotations

import re

#: A word that starts a phrase and says which kind of action it is.
_OPENERS = {
    "launch": "open_app", "start": "open_app", "run app": "open_app",
    "open app": "open_app",
    "go to": "open_url", "visit": "open_url", "browse": "open_url",
    "open website": "open_url", "open url": "open_url",
    "open folder": "open_path", "open file": "open_path",
    "open": None,   # ambiguous: decided by what follows
}

#: The folders every Mac and Windows account has under home, by the
#: name a person says.  "open downloads" means this folder, not an app
#: called downloads -- which was the mis-guess.  Matched case- and
#: "folder"-word-insensitively.
_HOME_FOLDERS = {
    "downloads": "~/Downloads",
    "documents": "~/Documents",
    "desktop": "~/Desktop",
    "pictures": "~/Pictures",
    "music": "~/Music",
    "movies": "~/Movies",
    "home": "~",
    "applications": "/Applications",
    "trash": "~/.Trash",
}


def _home_folder(text: str):
    """A known home folder for a plain name like "downloads" or
    "downloads folder", or None."""

    name = text.strip().lower()

    for suffix in (" folder", " directory"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()

    if name.startswith("my "):
        name = name[3:].strip()

    return _HOME_FOLDERS.get(name)


def looks_like_url(text: str) -> bool:
    """A bare domain or a full URL, not a folder or an app."""

    text = text.strip()

    if text.startswith(("http://", "https://")):
        return True

    # something.tld with no spaces -- gmail.com, music.youtube.com
    return bool(re.fullmatch(r"[\w.-]+\.[a-z]{2,}(/\S*)?", text, re.I))


def looks_like_path(text: str) -> bool:
    """An absolute path or a ~ path, not an app or a site."""

    text = text.strip()

    return text.startswith(("/", "~", "./"))


def parse(phrase: str):
    """A typed phrase -> a single action dict, or None if unclear.

    None is not failure, it is honesty: the form takes None as "let the
    person choose the type", rather than guessing wrong.
    """

    text = phrase.strip()

    if not text:
        return None

    lowered = text.lower()

    # A leading opener word picks the kind; the rest is the target.
    for opener, kind in sorted(_OPENERS.items(), key=lambda kv: -len(kv[0])):
        if lowered.startswith(opener + " "):
            rest = text[len(opener):].strip()

            if not rest:
                return None

            if kind == "open_path":
                # "open folder downloads" -> the known folder, else the
                # literal path typed.
                return _build("open_path", _home_folder(rest) or rest)

            if kind is not None:
                return _build(kind, rest)

            # Bare "open X": decide by what X is.
            folder = _home_folder(rest)
            if folder:
                return _build("open_path", folder)
            if looks_like_url(rest):
                return _build("open_url", rest)
            if looks_like_path(rest):
                return _build("open_path", rest)
            # A plain name after "open" is most often an app.
            return _build("open_app", rest)

    # No opener word, but the whole phrase is clearly a folder, a site,
    # or a path.
    folder = _home_folder(text)
    if folder:
        return _build("open_path", folder)
    if looks_like_url(text):
        return _build("open_url", text)
    if looks_like_path(text):
        return _build("open_path", text)

    return None


def _build(kind: str, target: str):
    if kind == "open_url":
        if not target.startswith(("http://", "https://")):
            target = "https://" + target
        return {"type": "open_url", "url": target}

    if kind == "open_app":
        return {"type": "open_app", "app": target}

    if kind == "open_path":
        return {"type": "open_path", "path": target}

    return None
