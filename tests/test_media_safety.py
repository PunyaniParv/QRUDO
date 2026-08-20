"""Music must never volunteer.

The media key, posted with no session claiming it, is answered by
macOS opening the Music app -- the single most-reported accident in
QRUDO's history, killed and reborn three times.  This file pins the
current shape of the cure: the key is only ever posted while audio is
provably playing; a resume goes LOOKING for the paused video by
script and plays it where it sits; and when nothing is found, the
letter -- which cannot launch anything -- takes over.  No path posts
the media key into a void.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import log

import tempfile

log.setup(tempfile.mkdtemp(), console=False)


@unittest.skipUnless(sys.platform == "darwin",
                     "the media-key routing lives in the macOS backend")
class TestMusicNeverVolunteers(unittest.TestCase):
    def controller(self):
        from control.backends.macos import MacOSController
        from control.config import ControlConfig

        return MacOSController(ControlConfig(browser_play_key="media"))

    def test_a_dead_resume_takes_the_letter_never_the_media_key(self):
        """QRUDO paused something that has since vanished: the resume
        finds nothing, and the letter -- not the media key -- carries
        the press.  The media key here is how Music kept launching."""

        c = self.controller()
        c._paused_it = True

        with mock.patch.object(c, "_audio_playing", return_value=False), \
                mock.patch.object(c, "_resume_paused_video",
                                  return_value=None), \
                mock.patch.object(c, "_refuse_to_type_into_a_text_box"), \
                mock.patch.object(c, "_post_key") as letter, \
                mock.patch.object(c, "_post_media_key") as media:
            c._play_pause_key("Google Chrome", None)

        media.assert_not_called()
        letter.assert_called_once()

    def test_a_living_resume_is_played_where_it_sits(self):
        c = self.controller()
        c._paused_it = True

        with mock.patch.object(c, "_audio_playing", return_value=False), \
                mock.patch.object(c, "_resume_paused_video",
                                  return_value='resumed "x" where it sat'), \
                mock.patch.object(c, "_post_key") as letter, \
                mock.patch.object(c, "_post_media_key") as media:
            said = c._play_pause_key("Google Chrome", None)

        self.assertIn("resumed", said)
        media.assert_not_called()
        letter.assert_not_called()

    def test_audible_playback_still_pauses_by_media_key(self):
        """The one safe case stays: something is audible, a session
        claims the key, the pause is precise."""

        c = self.controller()

        with mock.patch.object(c, "_audio_playing", return_value=True), \
                mock.patch.object(c, "_post_media_key") as media:
            said = c._play_pause_key("Google Chrome", None)

        media.assert_called_once()
        self.assertIn("paused", said)


if __name__ == "__main__":
    unittest.main(verbosity=2)
