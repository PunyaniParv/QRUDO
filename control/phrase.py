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
    "quit": "quit_app", "close app": "quit_app", "exit": "quit_app",
    "close": "quit_app",
    "go to": "open_url", "visit": "open_url", "browse": "open_url",
    "open website": "open_url", "open url": "open_url",
    "open folder": "open_path", "open file": "open_path",
    "open": None,   # ambiguous: decided by what follows
}

#: The words that mean "every app at once" after a quit verb.
_ALL_WORDS = {"all", "everything", "all apps", "every app", "all of them"}

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


def _strip_words(name: str):
    """Drop the wrapping words a person adds -- 'the', 'my', 'folder'."""

    name = name.strip()
    lowered = name.lower()

    for suffix in (" folder", " directory"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)].strip()
            lowered = name.lower()

    for prefix in ("the ", "my "):
        if lowered.startswith(prefix):
            name = name[len(prefix):].strip()
            lowered = name.lower()

    return name


def _home_folder(text: str):
    """A known home folder for a plain name like "downloads", or None."""

    return _HOME_FOLDERS.get(_strip_words(text).lower())


def _found_on_disk(name: str):
    """A real folder or file matching this name in the usual places, or
    None -- so 'open qfo' finds ~/qfo without it being hardcoded.

    Checks, in order: an absolute/~ path as given, then the name under
    home and under each common folder.  The first that exists wins.
    Case-insensitive on the leaf so 'downloads' finds 'Downloads'.
    """

    import os

    bare = _strip_words(name)

    # An explicit path, expanded.
    if bare.startswith(("/", "~", "./")):
        full = os.path.expanduser(bare)
        return bare if os.path.exists(full) else None

    home = os.path.expanduser("~")
    roots = [home] + [os.path.expanduser(p) for p in _HOME_FOLDERS.values()]

    for root in roots:
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for entry in entries:
            if entry.lower() == bare.lower():
                found = os.path.join(root, entry)
                # Return a ~-relative path when under home, for a tidy,
                # portable stored action.
                if found.startswith(home):
                    return "~" + found[len(home):]
                return found

    return None


def _app_exists(name: str):
    """Whether an application by this name is installed, so a real app
    beats a same-named file and a typo does not silently 'launch' air."""

    import os

    bare = _strip_words(name)

    for apps in ("/Applications", os.path.expanduser("~/Applications"),
                 "/System/Applications"):
        try:
            entries = os.listdir(apps)
        except OSError:
            continue
        for entry in entries:
            stem = entry[:-4] if entry.endswith(".app") else entry
            if stem.lower() == bare.lower():
                return True

    return False


#: Endings that mean a file, not a website -- so "report.pdf" opens a
#: document rather than being mistaken for a domain.  Broad on purpose:
#: a person opening a thing by name almost always means a real file.
_FILE_EXTENSIONS = {
    "pdf", "doc", "docx", "txt", "md", "rtf", "pages",
    "xls", "xlsx", "csv", "numbers", "ppt", "pptx", "key",
    "png", "jpg", "jpeg", "gif", "heic", "webp", "svg", "bmp",
    "mp3", "wav", "m4a", "aac", "flac",
    "mp4", "mov", "avi", "mkv", "webm",
    "zip", "dmg", "app", "pkg",
    "py", "js", "html", "css", "json", "sh",
}


def _extension(text: str) -> str:
    text = text.strip().rstrip("/")
    if "." in text:
        return text.rsplit(".", 1)[-1].lower()
    return ""


def looks_like_file(text: str) -> bool:
    """A bare filename with a known file extension: report.pdf, a.png."""

    return _extension(text) in _FILE_EXTENSIONS and "/" not in text.strip()


def looks_like_url(text: str) -> bool:
    """A bare domain or a full URL, not a folder, a file, or an app."""

    text = text.strip()

    if text.startswith(("http://", "https://")):
        return True

    # A filename with a document/media extension is a file, never a URL.
    if looks_like_file(text):
        return False

    # something.tld with no spaces -- gmail.com, music.youtube.com
    return bool(re.fullmatch(r"[\w.-]+\.[a-z]{2,}(/\S*)?", text, re.I))


def looks_like_path(text: str) -> bool:
    """An absolute path, a ~ path, or a bare filename with an extension."""

    text = text.strip()

    return (text.startswith(("/", "~", "./"))
            or looks_like_file(text))


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

            # An explicit "folder"/"file" opener forces a path: the
            # known home folder, a real thing found on disk, or the
            # literal path typed.
            if kind == "open_path":
                return _build("open_path",
                              _home_folder(rest) or _found_on_disk(rest)
                              or _strip_words(rest))

            if kind == "open_app":
                return _build("open_app", _strip_words(rest))

            # "quit spotify" asks that app to close; "quit all" asks
            # every open app.  The target need not be running at save
            # time -- quitting an app that is not running succeeds by
            # doing nothing, which is what was asked for.
            if kind == "quit_app":
                bare = _strip_words(rest)
                if bare.lower() in _ALL_WORDS:
                    bare = "all"
                return _build("quit_app", bare)

            if kind is not None:
                return _build(kind, rest)

            # Bare "open X": look at reality.  With the word "open"
            # present, an unresolved plain name is taken as an app to
            # try -- the person clearly meant to open something.
            return _resolve_open(rest, guess_app=True)

    # No opener word at all: only resolve what is unambiguous on its own
    # -- a known folder, a real path on disk, a website, a file.  A bare
    # phrase like "do the thing" or a catalog job like "next track" is
    # not a path, so it returns None and the caller (catalog, or the
    # form) decides -- rather than blindly "launching an app" named it.
    return _resolve_open(text, guess_app=False)


def _resolve_open(target: str, guess_app: bool):
    """Decide what a target means by what it actually is.

    Order so the likeliest right answer wins: a known home folder, then
    a real folder or file on disk (this is what makes 'open qfo' find
    ~/qfo), then a genuine website, then a path or file by shape.  With
    ``guess_app`` -- set only when the word "open" was present -- a plain
    name that matched nothing is taken as an app to try.  Without it, an
    unmatched phrase returns None: the caller then knows this was not a
    path, and lets the catalog or the person decide.
    """

    folder = _home_folder(target)
    if folder:
        return _build("open_path", folder)

    on_disk = _found_on_disk(target)
    if on_disk:
        return _build("open_path", on_disk)

    if looks_like_url(target):
        return _build("open_url", target)

    if looks_like_path(target) or looks_like_file(target):
        return _build("open_path", target)

    if guess_app:
        return _build("open_app", _strip_words(target))

    return None


def _build(kind: str, target: str):
    if kind == "open_url":
        if not target.startswith(("http://", "https://")):
            target = "https://" + target
        return {"type": "open_url", "url": target}

    if kind == "open_app":
        return {"type": "open_app", "app": target}

    if kind == "quit_app":
        return {"type": "quit_app", "app": target}

    if kind == "open_path":
        return {"type": "open_path", "path": target}

    return None
