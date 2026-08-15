"""Which app a play/pause reaches on macOS.

Twice now, pausing has opened Apple Music instead.  The media key looks
like the obvious way to play or pause anything, and it is a message to the
system rather than to a player: with nothing currently playing, macOS
answers it by opening Music.  So it is not used, and these pin that.

The real backend cannot be built here without a Mac's frameworks, so the
parts that touch them are replaced and the routing is what is tested.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import ControlConfig, log
from control.backends.macos import (KEY_J, KEY_L, KEY_RIGHT_ARROW,
                                    MacOSController)
from control.executor import UnsupportedCommand

log.setup(console=False)


class FakeMac(MacOSController):
    """The routing, with everything that needs macOS replaced."""

    def __init__(self, running=(), target="", play_key="media",
                 seek_keys="arrows", playing=False):
        self.config = ControlConfig(target_app=target,
                                    browser_play_key=play_key,
                                    browser_seek_keys=seek_keys)
        self.log = log.get_logger("test")
        self.running = set(running)
        self.told = []          # apps spoken to by AppleScript
        self.keyed = []         # apps sent a keystroke
        self.codes = []         # the keys themselves
        self.media_keys = []    # system-wide media keys, which must stay empty
        self._quartz = object()
        self._workspace = None
        self.playing = playing      # what the speakers are doing
        self.focus_in_text_box = False

    def _audio_playing(self):
        return self.playing

    def _refuse_to_type_into_a_text_box(self, pid):
        if self.focus_in_text_box:
            raise UnsupportedCommand("the cursor is in a text box")

    def _running_apps(self):
        return self.running

    def _osascript(self, script):
        self.told.append(script)
        return ""

    def _target_pid(self, name=None):
        name = name if name is not None else self.config.app
        return 4242 if name in self.running else None

    def _post_key(self, key_code, *, to_pid=None):
        self.keyed.append(to_pid)
        self.codes.append(key_code)

    def _post_media_key(self, key):
        self.media_keys.append(key)


class TestNothingPlaying(unittest.TestCase):
    def test_with_no_player_open_it_refuses(self):
        """Rather than pressing a key macOS answers by opening Music."""

        controller = FakeMac(running={"Finder", "Terminal"})

        with self.assertRaises(UnsupportedCommand):
            controller.play_pause()

        self.assertEqual(controller.media_keys, [])

    def test_a_browser_open_but_idle_does_not_get_the_media_key(self):
        """The case that reopened Music: Chrome was running, not playing.

        A running browser is not a playing one, and the media key goes
        wherever the system thinks the music is -- which is Music.
        """

        controller = FakeMac(running={"Google Chrome"}, play_key="k")
        controller.play_pause()

        self.assertEqual(controller.media_keys, [])
        self.assertEqual(controller.keyed, [4242])


class TestWhichKeyTheBrowserGets(unittest.TestCase):
    """Which key plays and pauses depends on the site, not the browser.

    Reported: a fist started skipping tracks instead of pausing.  It was
    sending k, which is YouTube's, to whatever was playing -- and on
    another player that key means something else.
    """

    def test_the_default_is_the_keyboards_own_key(self):
        """It reaches whatever is playing, whichever site that is.

        Guessing a letter means guessing the site's shortcut, and a wrong
        guess is not a no-op: k is play/pause on YouTube and skips a track
        elsewhere, which is how a fist came to change songs.
        """

        controller = FakeMac(running={"Google Chrome"}, playing=True)
        controller.play_pause()

        self.assertEqual(len(controller.media_keys), 1)

    def test_youtubes_letter_for_anyone_who_wants_it(self):
        controller = FakeMac(running={"Google Chrome"}, play_key="k")

        self.assertIn("(k)", controller.play_pause())

    def test_the_spacebar_for_players_that_use_it(self):
        controller = FakeMac(running={"Google Chrome"}, play_key="space")

        self.assertIn("(space)", controller.play_pause())
        self.assertEqual(controller.media_keys, [])

    def test_a_scriptable_player_ignores_the_setting(self):
        """Spotify is told in words; no key is involved."""

        controller = FakeMac(running={"Spotify"}, play_key="space")
        controller.play_pause()

        self.assertEqual(controller.keyed, [])
        self.assertEqual(controller.media_keys, [])


class TestNothingReachesMusic(unittest.TestCase):
    """Music opening has been reported, fixed, and reported again three
    times.  Each fix guarded the system-wide media key instead of not
    sending it, and the last guard asked CoreAudio whether audio was
    playing -- which is a different question from whether any app has
    claimed the system's now-playing role, and it is the second that
    decides where the key goes.  Sound can come from something that never
    claimed it, and with the role unclaimed macOS answers the key by
    opening Music.

    So the key is not sent.  Play/pause goes to the app, as the app's own
    shortcut, on every path.
    """

    def controller(self, **kwargs):
        kwargs.setdefault("play_key", ControlConfig().browser_play_key)
        return FakeMac(target="Google Chrome", **kwargs)

    def test_the_default_never_sends_a_system_key(self):
        for running in ({"Google Chrome"}, {"Google Chrome", "Music"}):
            for playing in (True, False):
                with self.subTest(open=sorted(running), playing=playing):
                    controller = self.controller(running=running,
                                                 playing=playing)
                    controller.play_pause()

                    self.assertEqual(controller.media_keys, [])

    def test_nor_any_apple_event_to_music(self):
        controller = self.controller(running={"Google Chrome", "Music"},
                                     playing=True)
        controller.play_pause()

        self.assertFalse([said for said in controller.told if "Music" in said])

    def test_it_reaches_the_browser_instead(self):
        controller = self.controller(running={"Google Chrome"}, playing=False)

        self.assertIn("Chrome", controller.play_pause())
        self.assertEqual(controller.keyed, [4242])


class TestNoLetterIntoATextBox(unittest.TestCase):
    """The letter lands wherever the browser's keyboard focus is.

    Reported twice as a fist typing k.  The letter is the only safe way
    to reach a player -- the system media key opens Music -- so it stays,
    and the focus is asked about first: Accessibility says what element
    has the keyboard, and a text field refuses the letter rather than
    receiving it.
    """

    def test_focus_in_a_text_box_refuses_rather_than_types(self):
        controller = FakeMac(running={"Google Chrome"},
                             target="Google Chrome")
        controller.focus_in_text_box = True

        with self.assertRaises(UnsupportedCommand):
            controller.play_pause()

        self.assertEqual(controller.keyed, [], "the letter was typed anyway")
        self.assertEqual(controller.media_keys, [])

    def test_focus_anywhere_else_gets_the_letter(self):
        controller = FakeMac(running={"Google Chrome"},
                             target="Google Chrome")

        controller.play_pause()

        self.assertEqual(controller.keyed, [4242])

    def test_not_being_able_to_tell_sends_the_letter(self):
        """The real query answers False on any error, including the app
        having no focused element to report.  The guard is for a text box
        known to be there; "cannot tell" keeps today's behaviour."""

        from control.backends import macos

        self.assertFalse(macos._focus_is_a_text_box(99999999))


class TestPlayPauseTakesWhicheverRouteIsSafe(unittest.TestCase):
    """Two jobs that look like one, and they need different routes.

    Stopping something that is playing is best done with the keyboard's
    own play/pause key: it reaches whatever is playing, whichever site or
    app that is, and no shortcut has to be guessed for it.  Its one fault
    is that it is a message to the system, and the system answers one with
    nothing playing by opening Music.

    Starting something is the other job.  There is nothing for the system
    key to reach, so it goes to the app instead, as the app's own
    shortcut -- which can neither be diverted nor open anything.
    """

    def test_something_playing_gets_the_media_key(self):
        controller = FakeMac(running={"Google Chrome"}, playing=True)
        detail = controller.play_pause()

        self.assertEqual(len(controller.media_keys), 1)
        self.assertEqual(controller.keyed, [], "no letter should be typed")
        self.assertIn("paused", detail)

    def test_nothing_playing_starts_it_through_the_app(self):
        """Reported: a fist could pause a video and never start one.

        Refusing here was the earlier answer, and it meant nothing could
        be set going without reaching for the keyboard.
        """

        controller = FakeMac(running={"Google Chrome"}, playing=False)
        detail = controller.play_pause()

        self.assertEqual(controller.keyed, [4242])
        self.assertEqual(controller.media_keys, [],
                         "the system key is what opens Music")
        self.assertIn("nothing was playing", detail)

    def test_the_system_key_never_goes_out_with_nothing_playing(self):
        """Which is the whole of the Music guarantee.

        It cannot open Music if it is only ever pressed while something
        else is already playing.
        """

        for running in ({"Google Chrome"}, {"Google Chrome", "Music"}):
            controller = FakeMac(running=running, playing=False)
            controller.play_pause()

            self.assertEqual(controller.media_keys, [], str(running))

    def test_music_being_open_changes_nothing(self):
        controller = FakeMac(running={"Music", "Google Chrome"}, playing=True)
        controller.play_pause()

        self.assertEqual(controller.keyed, [])
        self.assertEqual(len(controller.media_keys), 1)

    def test_pausing_then_starting_again(self):
        """The pair, in the order anyone would do them."""

        controller = FakeMac(running={"Google Chrome"}, playing=True)
        self.assertIn("paused", controller.play_pause())

        controller.playing = False          # it stopped, because we stopped it
        self.assertIn("nothing was playing", controller.play_pause())

    def test_a_machine_that_cannot_be_asked_uses_the_app(self):
        """Without CoreAudio there is no telling, and of the two risks
        typing a letter is the smaller one -- it does not leave a music
        player open and holding every media key afterwards."""

        controller = FakeMac(running={"Google Chrome"})
        controller._audio_playing = lambda: None

        controller.play_pause()

        self.assertEqual(controller.media_keys, [])
        self.assertEqual(controller.keyed, [4242])

    def test_asking_for_a_letter_still_gets_one(self):
        controller = FakeMac(running={"Google Chrome"}, play_key="k")
        controller.play_pause()

        self.assertEqual(controller.keyed, [4242])
        self.assertEqual(controller.media_keys, [])


class TestMusicIsNeverChosenForYou(unittest.TestCase):
    """Reported as "fist always triggers apple music", and it did.

    macOS opens Music by itself in answer to a media key with nothing
    playing.  Once it is open it is the first scriptable player found, so
    every fist from then on drove Music -- and the media key kept going
    there too, which kept it open.  Its being open was never evidence
    that anybody wanted it.
    """

    def test_an_open_music_is_not_picked(self):
        controller = FakeMac(running={"Music", "Google Chrome"},
                             play_key="k")
        detail = controller.play_pause()

        self.assertIn("Chrome", detail)
        self.assertFalse([said for said in controller.told if "Music" in said])

    def test_naming_music_still_reaches_it(self):
        """Skipped when guessing, honoured when asked for."""

        controller = FakeMac(running={"Music"}, target="Music")

        self.assertIn("Music", controller.play_pause())

    def test_music_alone_and_unasked_for_refuses(self):
        controller = FakeMac(running={"Music"})

        with self.assertRaises(UnsupportedCommand):
            controller.play_pause()

    def test_music_being_open_does_not_divert_to_a_letter(self):
        """Music open used to send the browser's own shortcut instead, so
        as not to hand the media key to Music.

        That is only a danger for a media key sent with nothing playing,
        which is refused now -- and the diversion typed a letter into
        whatever had the keyboard focus, which is worse than the thing it
        avoided.
        """

        controller = FakeMac(running={"Music", "Google Chrome"}, playing=True)
        controller.play_pause()

        self.assertEqual(controller.keyed, [], "a letter was typed")
        self.assertEqual(len(controller.media_keys), 1)

    def test_music_open_with_nothing_playing_reaches_the_browser(self):
        """Not Music, and not the system key that would find Music."""

        controller = FakeMac(running={"Music", "Google Chrome"}, playing=False)
        controller.play_pause()

        self.assertEqual(controller.keyed, [4242])
        self.assertEqual(controller.media_keys, [])
        self.assertFalse([said for said in controller.told if "Music" in said])

    def test_with_music_shut_the_media_key_is_still_the_default(self):
        controller = FakeMac(running={"Google Chrome"}, playing=True)
        controller.play_pause()

        self.assertEqual(len(controller.media_keys), 1)


class TestWhichKeySeeks(unittest.TestCase):
    """Reported: seeking worked on YouTube in Chrome but not YouTube Music.

    Both are Chrome, so the browser was never the difference -- the site
    was.  Arrow keys seek on one and not the other, and there is no key
    that seeks on all of them, so which key to send is a setting.
    """

    def test_arrows_by_default(self):
        controller = FakeMac(running={"Google Chrome"}, target="Google Chrome")
        detail = controller.forward(10)

        self.assertEqual(controller.codes, [KEY_RIGHT_ARROW, KEY_RIGHT_ARROW])
        self.assertIn("2x arrow", detail)

    def test_the_ten_second_keys_for_players_that_use_them(self):
        controller = FakeMac(running={"Google Chrome"}, target="Google Chrome",
                             seek_keys="jl")

        self.assertIn("1x j/l", controller.forward(10))
        self.assertEqual(controller.codes, [KEY_L])

    def test_back_is_the_other_one(self):
        controller = FakeMac(running={"Google Chrome"}, target="Google Chrome",
                             seek_keys="jl")
        controller.rewind(10)

        self.assertEqual(controller.codes, [KEY_J])

    def test_it_still_covers_the_distance_asked_for(self):
        """Ten seconds is ten seconds, whichever key is doing it."""

        for keys, presses in (("arrows", 4), ("jl", 2)):
            controller = FakeMac(running={"Google Chrome"},
                                 target="Google Chrome", seek_keys=keys)
            controller.forward(20)

            self.assertEqual(len(controller.codes), presses, keys)

    def test_an_unknown_setting_falls_back_rather_than_failing(self):
        controller = FakeMac(running={"Google Chrome"}, target="Google Chrome",
                             seek_keys="whatever")
        controller.forward(10)

        self.assertEqual(controller.codes, [KEY_RIGHT_ARROW, KEY_RIGHT_ARROW])


class TestWhichAppItReaches(unittest.TestCase):
    def test_a_scriptable_player_is_told_directly(self):
        controller = FakeMac(running={"Spotify", "Google Chrome"})
        detail = controller.play_pause()

        self.assertIn("Spotify", detail)
        self.assertTrue(any("Spotify" in said for said in controller.told))
        self.assertEqual(controller.keyed, [])

    def test_a_browser_gets_the_players_own_shortcut(self):
        controller = FakeMac(running={"Google Chrome"}, play_key="k")
        detail = controller.play_pause()

        self.assertIn("Chrome", detail)
        self.assertEqual(controller.keyed, [4242])

    def test_a_named_app_wins_over_whatever_else_is_open(self):
        controller = FakeMac(running={"Spotify", "Google Chrome"},
                             target="Google Chrome", play_key="k")
        detail = controller.play_pause()

        self.assertIn("Chrome", detail)
        self.assertEqual(controller.told, [])

    def test_a_player_that_is_shut_is_never_spoken_to(self):
        """Naming an app in AppleScript is enough to launch it.

        The obvious "if it is running then tell it" starts the very thing
        it was written to avoid starting, so whether it is running has to
        be settled before saying anything at all.
        """

        controller = FakeMac(running={"Google Chrome"}, playing=True)
        controller.play_pause()

        self.assertFalse([said for said in controller.told if "Music" in said])


if __name__ == "__main__":
    unittest.main(verbosity=2)
