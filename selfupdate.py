"""Stage two of staying current: the app replaces itself.

Stage one (updates.py) is the noticing; this is the doing.  The chain
has three links, each allowed to fail without consequence until the
last: the release's files are fetched over https with our own
certificates, every downloaded byte is checked against the SHA256SUMS
the release publishes before anything is touched, and only then is
the installed app swapped -- the old one set aside, the new one moved
into its place, whole or not at all.

The download and the checking happen in the background while the
camera runs; the swap happens only when asked -- the U key in the
app, or --update in a terminal.  An update must never surprise anyone
mid-gesture.

What code signing adds later is not this mechanism but its manners:
a Developer ID lets macOS keep the camera and Accessibility grants
across updates instead of asking again, and lets Gatekeeper vouch for
the download.  The checksum keeps the transport honest either way.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import updates

#: What each platform's release asset is called.  The Mac updates from
#: the zip; Windows hands the work to the installer, which already
#: knows how to replace a running app and its shortcuts.
ASSET = {"darwin": "QRUDO.zip", "win32": "QRUDO-Setup.exe"}

SUMS = "SHA256SUMS"


def prepare():
    """Download and verify the newer release, or None.

    Returns (staged file, version) once the bytes are on disk and
    their checksum matches the release's word for it.  Every failure
    -- no release, no asset for this platform, a bad checksum, any
    network trouble -- is a quiet None: this runs behind a live
    camera session, and nothing about it may ever intrude.
    """

    try:
        found = updates.release()

        if not found:
            return None

        version = str(found.get("tag_name", "")).lstrip("vV")

        if not version or not updates.newer(version, updates.VERSION):
            return None

        assets = {asset.get("name"): asset.get("browser_download_url")
                  for asset in found.get("assets", [])}

        wanted = ASSET.get(sys.platform)
        url = assets.get(wanted)
        sums_url = assets.get(SUMS)

        if not url or not sums_url:
            return None

        from paths import data_dir

        staging = data_dir() / "updates" / version
        staging.mkdir(parents=True, exist_ok=True)
        staged = staging / wanted

        with updates.fetch(sums_url, timeout=10) as response:
            sums = response.read().decode("utf-8", "replace")

        expected = expected_digest(sums, wanted)

        if not expected:
            return None

        # Not re-downloaded if a verified copy is already waiting --
        # the check costs a read, the download costs hundreds of MB.
        if not (staged.exists() and digest(staged) == expected):
            with updates.fetch(url, timeout=30) as response, \
                    staged.open("wb") as handle:
                shutil.copyfileobj(response, handle)

            if digest(staged) != expected:
                staged.unlink()
                return None

        return staged, version
    except Exception:
        return None


def apply(staged, version):
    """Swap the installed app for the verified download.

    macOS: the running bundle is set aside and the new one moved into
    its place -- safe while running, because the running process keeps
    its files until it exits.  Windows: the downloaded installer is
    started and this process steps out of its way.

    Returns False when it cannot apply (a development checkout, an
    unpacking failure); on success the macOS swap returns True and the
    Windows path exits into the installer.
    """

    if not getattr(sys, "frozen", False):
        print("  a development checkout updates with git pull, not this")
        return False

    staged = Path(staged)

    if sys.platform == "win32":
        subprocess.Popen([str(staged), "/SILENT"])
        raise SystemExit(0)

    app_root = Path(sys.executable).resolve().parents[2]

    if app_root.suffix != ".app":
        print(f"  cannot find the app bundle from {sys.executable}")
        return False

    unpacked = staged.parent / "unpacked"

    if unpacked.exists():
        shutil.rmtree(unpacked)

    # ditto, not zipfile: an app bundle is symlinks and executable
    # bits, and zipfile flattens both -- an app unpacked by it never
    # launches again.
    done = subprocess.run(
        ["ditto", "-xk", str(staged), str(unpacked)],
        capture_output=True, text=True)

    if done.returncode != 0:
        print(f"  could not unpack the update: {done.stderr.strip()}")
        return False

    new_app = unpacked / app_root.name

    if not new_app.exists():
        print(f"  the update holds no {app_root.name}")
        return False

    swap(app_root, new_app, keep=staged.parent / "previous")

    return True


def swap(app_root, new_app, keep):
    """The old app aside, the new app in its place.

    Aside rather than deleted: the one thing a self-updating app owes
    its user is a way back, and the previous version next to the
    download is that way.
    """

    if keep.exists():
        shutil.rmtree(keep)

    keep.mkdir(parents=True)
    shutil.move(str(app_root), str(keep / app_root.name))
    shutil.move(str(new_app), str(app_root))


def relaunch(app_root):
    """The new version, started; this process ends here."""

    subprocess.Popen(["open", str(app_root)])
    os._exit(0)


def expected_digest(sums, name):
    """The checksum SHA256SUMS promises for ``name``, or empty.

    The format is the classic one: hex, whitespace, filename, one file
    a line.  Anything unreadable answers empty, and an empty answer
    refuses the download -- unverifiable is the same as wrong here.
    """

    for line in sums.splitlines():
        parts = line.split()

        if len(parts) >= 2 and parts[-1] == name and len(parts[0]) == 64:
            return parts[0].lower()

    return ""


def digest(path):
    """The SHA-256 of a file on disk, hex."""

    sha = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)

    return sha.hexdigest()


def run_cli():
    """The --update mode: check, fetch, verify, swap, report.

    No relaunch at the end -- a person at a terminal decides when to
    start things.  The U key inside the app is the path that relaunches.
    """

    latest = updates.check()

    if not latest:
        print(f"  this is QRUDO {updates.VERSION}, and it is current.")
        return 0

    print(f"  QRUDO {latest} is out (this is {updates.VERSION}); "
          f"downloading...")

    ready = prepare()

    if not ready:
        print("  could not fetch a verifiable update -- try again later, "
              f"or download it yourself: {updates.DOWNLOAD_PAGE}")
        return 1

    staged, version = ready

    if not apply(staged, version):
        return 1

    print(f"  updated to QRUDO {version}.  Start it normally when ready.")
    return 0
