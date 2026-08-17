"""Where QRUDO keeps what it learns about you.

Run from the repository, everything sits next to main.py, exactly as
it always has: the calibration, the local config, the logs.  Packaged
as an app, none of that may live where the code lives -- writing into
a signed bundle breaks its seal, and the working directory of an app
launched from an icon is nowhere useful -- so it moves to the folder
each platform sets aside for exactly this:

    macOS      ~/Library/Application Support/QRUDO
    Windows    %APPDATA%\\QRUDO
    elsewhere  $XDG_DATA_HOME/QRUDO, or ~/.local/share/QRUDO

One function answers for every file, so the two ways of running
cannot disagree about where anything is.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def data_dir() -> Path:
    """The folder QRUDO's own files live in, created if needed."""

    if not getattr(sys, "frozen", False):
        return REPO

    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home())
    else:
        base = Path(os.environ.get("XDG_DATA_HOME")
                    or Path.home() / ".local" / "share")

    directory = base / "QRUDO"
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def resolve(path: str | Path) -> Path:
    """A path from the config, anchored where QRUDO's files live.

    Absolute paths are the user's own choice and pass through; a bare
    name like the default ``logs`` means "ours", wherever ours is.
    """

    path = Path(path)

    if path.is_absolute():
        return path

    return data_dir() / path
