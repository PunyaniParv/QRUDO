"""Whether a newer QRUDO exists, asked quietly, answered on screen.

A customer's app cannot git pull -- it must notice its own age.  This
asks the release page for the newest version and compares.  The asking
is built to be invisible when it cannot work: no network, no releases
yet, a laptop behind a proxy -- every failure is a quiet None, because
an update reminder is never worth interrupting a gesture session for.

Stage one of staying current: the app *knows* and says so in its
window; the customer clicks a download.  Stage two -- downloading and
replacing itself -- rides on code signing, because an app that swaps
its own binaries had better be able to prove the new one came from us.
When distribution moves to a proper website, LATEST_URL is the one
line that changes.
"""

from __future__ import annotations

from version import VERSION

#: The release feed: GitHub's "latest release" for the repository.
#: Answers with a tag like "v0.2.0" once releases exist there.
LATEST_URL = ("https://api.github.com/repos/PunyaniParv/QRUDO"
              "/releases/latest")

#: Where a person goes to get it.  Shown, never opened uninvited.
DOWNLOAD_PAGE = "github.com/PunyaniParv/QRUDO/releases"


def check(timeout=4.0):
    """The newer version waiting for this machine, or None.

    None for every possible failure as well as for "already current":
    this runs on a customer's machine at startup, and nothing about
    checking may ever break starting.
    """

    try:
        import json
        import ssl
        from urllib.request import urlopen

        # A frozen app carries no system certificate store, so https
        # verification fails inside it unless the certificates ship
        # too.  They already do -- certifi rides in with the vision
        # stack -- it just has to be pointed at.
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()

        with urlopen(LATEST_URL, timeout=timeout,
                     context=context) as response:
            tag = json.load(response).get("tag_name", "")

        latest = tag.lstrip("vV")

        if latest and newer(latest, VERSION):
            return latest
    except Exception:
        return None

    return None


def newer(candidate, current):
    """Whether candidate is a later version than current.

    Compared piece by piece as numbers -- "0.10.0" beats "0.9.1",
    which a string comparison gets wrong.  A piece that is not a
    number ends the comparison as "not newer", so a malformed tag on
    the release page can never nag anyone.
    """

    def parts(version):
        pieces = []
        for piece in version.split("."):
            if not piece.isdigit():
                return None
            pieces.append(int(piece))
        return pieces

    ours, theirs = parts(current), parts(candidate)

    if ours is None or theirs is None:
        return False

    return theirs > ours
