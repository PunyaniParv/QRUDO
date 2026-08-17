"""The update check: a customer's app noticing its own age, safely.

Two promises.  The comparison must be numeric -- "0.10.0" beats
"0.9.1", which strings get backwards -- and checking must never be
able to break starting: every failure, from no network to a malformed
tag on the release page, is a quiet None.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import updates


class TestNewer(unittest.TestCase):
    def test_plainly_newer(self):
        self.assertTrue(updates.newer("0.2.0", "0.1.0"))

    def test_the_same_is_not_newer(self):
        self.assertFalse(updates.newer("0.1.0", "0.1.0"))

    def test_older_is_not_newer(self):
        self.assertFalse(updates.newer("0.0.9", "0.1.0"))

    def test_ten_beats_nine(self):
        """The reason strings cannot do this job."""

        self.assertTrue(updates.newer("0.10.0", "0.9.1"))

    def test_a_malformed_tag_never_nags(self):
        for tag in ("beta", "0.2.x", "", "1.2.3-rc1"):
            with self.subTest(tag=tag):
                self.assertFalse(updates.newer(tag, "0.1.0"))


def answering(payload):
    """A stand-in for urlopen that answers with the given JSON."""

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    return lambda url, timeout, context=None: Response(
        json.dumps(payload).encode())


class TestCheck(unittest.TestCase):
    def test_a_newer_release_is_reported(self):
        with mock.patch("urllib.request.urlopen",
                        answering({"tag_name": "v9.9.9"})):
            self.assertEqual(updates.check(), "9.9.9")

    def test_being_current_is_quiet(self):
        with mock.patch("urllib.request.urlopen",
                        answering({"tag_name": f"v{updates.VERSION}"})):
            self.assertIsNone(updates.check())

    def test_no_releases_yet_is_quiet(self):
        with mock.patch("urllib.request.urlopen", answering({})):
            self.assertIsNone(updates.check())

    def test_no_network_is_quiet(self):
        """The promise that matters most: checking can never break
        starting, whatever the network is doing."""

        def unreachable(url, timeout, context=None):
            raise OSError("no route to host")

        with mock.patch("urllib.request.urlopen", unreachable):
            self.assertIsNone(updates.check())


if __name__ == "__main__":
    unittest.main(verbosity=2)
