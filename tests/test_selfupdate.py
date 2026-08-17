"""Stage two: an app that replaces itself had better be paranoid.

The promises pinned here: nothing moves unless the checksum the
release published matches the bytes that arrived; a development
checkout is never touched; the old app is set aside, not destroyed;
and every failure in the background path is a quiet None, because it
runs behind a live camera session.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import selfupdate
import updates


class TestExpectedDigest(unittest.TestCase):
    SUMS = ("0" * 64 + "  QRUDO.zip\n"
            + "f" * 64 + "  QRUDO-Setup.exe\n")

    def test_finds_the_right_line(self):
        self.assertEqual(selfupdate.expected_digest(self.SUMS, "QRUDO.zip"),
                         "0" * 64)

    def test_a_missing_file_answers_empty(self):
        """Empty refuses the download: unverifiable equals wrong."""

        self.assertEqual(selfupdate.expected_digest(self.SUMS, "other.zip"),
                         "")

    def test_a_mangled_line_answers_empty(self):
        self.assertEqual(selfupdate.expected_digest("not a sums file",
                                                    "QRUDO.zip"), "")


class TestDigest(unittest.TestCase):
    def test_matches_a_known_answer(self):
        path = Path(tempfile.mkdtemp()) / "bytes"
        path.write_bytes(b"qrudo")

        import hashlib
        self.assertEqual(selfupdate.digest(path),
                         hashlib.sha256(b"qrudo").hexdigest())


class TestPrepare(unittest.TestCase):
    def test_every_failure_is_a_quiet_none(self):
        """It runs behind a live camera; it may never intrude."""

        with mock.patch.object(updates, "release", lambda **kw: None):
            self.assertIsNone(selfupdate.prepare())

        def explode(**kw):
            raise OSError("network gone")

        with mock.patch.object(updates, "release", explode):
            self.assertIsNone(selfupdate.prepare())

    def test_an_old_release_is_not_an_update(self):
        stale = {"tag_name": "v0.0.1", "assets": []}

        with mock.patch.object(updates, "release", lambda **kw: stale):
            self.assertIsNone(selfupdate.prepare())

    def test_no_asset_for_this_platform_is_none(self):
        bare = {"tag_name": "v99.0.0", "assets": []}

        with mock.patch.object(updates, "release", lambda **kw: bare):
            self.assertIsNone(selfupdate.prepare())

    def test_a_wrong_checksum_rejects_the_download(self):
        """The whole point of the SUMS file, exercised end to end:
        bytes arrive, the checksum disagrees, nothing is kept."""

        tmp = Path(tempfile.mkdtemp())
        name = selfupdate.ASSET[sys.platform]

        found = {"tag_name": "v99.0.0",
                 "assets": [{"name": name, "browser_download_url": "asset"},
                            {"name": "SHA256SUMS",
                             "browser_download_url": "sums"}]}

        import io

        def fake_fetch(url, timeout=0):
            class R(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

            if url == "sums":
                return R(("a" * 64 + f"  {name}\n").encode())
            return R(b"the wrong bytes")

        with mock.patch.object(updates, "release", lambda **kw: found), \
                mock.patch.object(updates, "fetch", fake_fetch), \
                mock.patch("paths.data_dir", lambda: tmp):
            self.assertIsNone(selfupdate.prepare())

        self.assertFalse((tmp / "updates" / "99.0.0" / name).exists(),
                         "a failed download must not be left behind")

    def test_a_good_checksum_stages_the_file(self):
        tmp = Path(tempfile.mkdtemp())
        name = selfupdate.ASSET[sys.platform]
        payload = b"the actual release"

        import hashlib
        import io

        good = hashlib.sha256(payload).hexdigest()

        found = {"tag_name": "v99.0.0",
                 "assets": [{"name": name, "browser_download_url": "asset"},
                            {"name": "SHA256SUMS",
                             "browser_download_url": "sums"}]}

        def fake_fetch(url, timeout=0):
            class R(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

            if url == "sums":
                return R((good + f"  {name}\n").encode())
            return R(payload)

        with mock.patch.object(updates, "release", lambda **kw: found), \
                mock.patch.object(updates, "fetch", fake_fetch), \
                mock.patch("paths.data_dir", lambda: tmp):
            ready = selfupdate.prepare()

        self.assertIsNotNone(ready)
        staged, version = ready
        self.assertEqual(version, "99.0.0")
        self.assertEqual(staged.read_bytes(), payload)


class TestApply(unittest.TestCase):
    def test_a_development_checkout_is_never_touched(self):
        self.assertFalse(selfupdate.apply("/nowhere", "9.9.9"))


class TestSwap(unittest.TestCase):
    def test_old_aside_new_in_place(self):
        """The one thing a self-updating app owes its user is a way
        back; the previous version set aside is that way."""

        tmp = Path(tempfile.mkdtemp())
        old = tmp / "QRUDO.app"
        old.mkdir()
        (old / "flavour").write_text("old")

        new = tmp / "staged" / "QRUDO.app"
        new.mkdir(parents=True)
        (new / "flavour").write_text("new")

        selfupdate.swap(old, new, keep=tmp / "previous")

        self.assertEqual((old / "flavour").read_text(), "new")
        self.assertEqual(
            (tmp / "previous" / "QRUDO.app" / "flavour").read_text(), "old")


if __name__ == "__main__":
    unittest.main(verbosity=2)
