"""Typed phrases to actions, by plain rules -- the no-AI link.

The first box takes a typed phrase as well as a menu pick.  These pin
the patterns a person will actually type, and that an unclear phrase
returns None (so the form asks rather than guesses wrong).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import phrase
from control.actions import validate


class TestParse(unittest.TestCase):
    def test_launch_an_app(self):
        self.assertEqual(phrase.parse("launch Spotify"),
                         {"type": "open_app", "app": "Spotify"})

    def test_quit_an_app(self):
        self.assertEqual(phrase.parse("quit Spotify"),
                         {"type": "quit_app", "app": "Spotify"})

    def test_close_means_quit_too(self):
        self.assertEqual(phrase.parse("close chrome"),
                         {"type": "quit_app", "app": "chrome"})

    def test_quit_all_is_the_everything_keyword(self):
        for said in ("quit all", "quit everything", "quit every app",
                     "close all apps"):
            with self.subTest(said=said):
                self.assertEqual(phrase.parse(said),
                                 {"type": "quit_app", "app": "all"})

    def test_open_a_folder_path(self):
        self.assertEqual(phrase.parse("open ~/Downloads"),
                         {"type": "open_path", "path": "~/Downloads"})

    def test_go_to_a_site(self):
        action = phrase.parse("go to gmail.com")
        self.assertEqual(action["type"], "open_url")
        self.assertEqual(action["url"], "https://gmail.com")

    def test_a_bare_domain(self):
        self.assertEqual(phrase.parse("music.youtube.com")["url"],
                         "https://music.youtube.com")

    def test_a_full_url_is_kept(self):
        self.assertEqual(phrase.parse("go to https://x.com/y")["url"],
                         "https://x.com/y")

    def test_open_a_plain_name_is_an_app(self):
        self.assertEqual(phrase.parse("open Calculator"),
                         {"type": "open_app", "app": "Calculator"})

    def test_open_something_that_looks_like_a_url(self):
        self.assertEqual(phrase.parse("open gmail.com")["type"], "open_url")

    def test_open_an_absolute_path(self):
        self.assertEqual(phrase.parse("open /Applications")["type"],
                         "open_path")

    def test_a_known_folder_name_is_a_folder_not_an_app(self):
        """The reported bug: 'open downloads' launched an app called
        downloads instead of opening the Downloads folder."""

        for text in ("open downloads", "downloads", "open Downloads",
                     "open my downloads folder"):
            with self.subTest(phrase=text):
                action = phrase.parse(text)
                self.assertEqual(action["type"], "open_path")
                self.assertEqual(action["path"], "~/Downloads")

    def test_common_folders(self):
        self.assertEqual(phrase.parse("open documents")["path"],
                         "~/Documents")
        self.assertEqual(phrase.parse("open desktop")["path"], "~/Desktop")

    def test_a_file_is_a_file_not_a_website(self):
        """'report.pdf' was mistaken for a domain (https://report.pdf)."""

        for text in ("open report.pdf", "open my_photo.png",
                     "open notes.txt"):
            with self.subTest(phrase=text):
                self.assertEqual(phrase.parse(text)["type"], "open_path")

    def test_a_full_file_path(self):
        self.assertEqual(phrase.parse("open ~/Desktop/notes.txt")["path"],
                         "~/Desktop/notes.txt")

    def test_a_real_website_still_wins(self):
        self.assertEqual(phrase.parse("go to gmail.com")["type"], "open_url")
        self.assertEqual(phrase.parse("open youtube.com")["type"], "open_url")

    def test_a_real_folder_on_disk_is_found(self):
        """The reported bug: 'open qfo' launched a phantom app instead of
        opening the real ~/qfo folder.  Now it checks disk."""

        import tempfile
        from pathlib import Path
        from unittest import mock

        tmp = Path(tempfile.mkdtemp())
        (tmp / "myproject").mkdir()

        with mock.patch("os.path.expanduser",
                        side_effect=lambda p: p.replace("~", str(tmp))):
            action = phrase.parse("open myproject")

        self.assertEqual(action["type"], "open_path")
        self.assertTrue(action["path"].endswith("myproject"))

    def test_an_unknown_name_still_tries_as_an_app(self):
        """'open Spotify' before Spotify exists on disk should still try
        to launch it, not fail."""

        action = phrase.parse("open SomeAppNobodyHas")
        self.assertEqual(action, {"type": "open_app",
                                  "app": "SomeAppNobodyHas"})

    def test_something_unclear_is_none(self):
        for unclear in ("do the thing", "make me a sandwich", "next track",
                        ""):
            with self.subTest(phrase=unclear):
                self.assertIsNone(phrase.parse(unclear))

    def test_every_parsed_phrase_is_a_valid_action(self):
        for text in ("launch Spotify", "open ~/Downloads", "go to gmail.com",
                     "open Calculator", "music.youtube.com"):
            with self.subTest(phrase=text):
                validate(phrase.parse(text))   # raises if malformed


if __name__ == "__main__":
    unittest.main(verbosity=2)
