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

# Scratch logs, not the live ones -- see test_control for why.
import tempfile

log.setup(tempfile.mkdtemp(), console=False)
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

        Nothing playing and QRUDO did not pause it, so the media key
        would go wherever the system thinks the music is -- which is
        Music.  The letter starts the fresh video instead.
        """

        controller = FakeMac(running={"Google Chrome"})
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


class TestTheMediaKeyIsSafeWhenAPlayerIsThere(unittest.TestCase):
    """The system now-playing key -- the default -- fixes three faults
    at once: it reaches the playing tab wherever it sits, it is genuine
    play/pause on YouTube Music, and it never lands in a chat box.  Its
    one hazard is the empty case, where macOS opens Music, and the
    backend is built to avoid exactly that: the key goes out only when
    something is playing, or when QRUDO paused it and so knows a player
    is sitting there.  These pin both halves.
    """

    def controller(self, **kwargs):
        kwargs.setdefault("play_key", ControlConfig().browser_play_key)
        return FakeMac(target="Google Chrome", **kwargs)

    def test_playing_now_gets_the_system_key(self):
        controller = self.controller(running={"Google Chrome"},
                                     playing=True)
        controller.play_pause()

        self.assertEqual(len(controller.media_keys), 1)

    def test_paused_by_us_resumes_with_the_system_key(self):
        """Pause something, and the next fist resumes it -- no audio to
        detect, but QRUDO knows it left a player paused."""

        controller = self.controller(running={"Google Chrome"},
                                     playing=True)
        controller.play_pause()                 # pauses; playing was True
        controller.playing = False              # now silent, because paused

        controller.play_pause()                 # must still take the key

        self.assertEqual(len(controller.media_keys), 2)

    def test_nothing_playing_never_sends_the_system_key(self):
        """The empty case, which is the one that opened Music: silent,
        and QRUDO did not pause it, so the letter is used instead."""

        controller = self.controller(running={"Google Chrome"},
                                     playing=False)
        note = controller.play_pause()

        self.assertEqual(controller.media_keys, [])
        self.assertEqual(controller.keyed, [4242])
        self.assertIn("nothing was playing", note)

    def test_the_empty_case_never_events_music(self):
        controller = self.controller(running={"Google Chrome", "Music"},
                                     playing=False)
        controller.play_pause()

        self.assertEqual(controller.media_keys, [])
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
        """A machine where the question cannot be asked keeps the old
        behaviour -- the report comes back unknown, and unknown sends."""

        from control.backends import macos

        kind, _ = macos._focus_report(99999999)

        self.assertIn(kind, ("unknown", "none"))


class TestTheLetterOnlyGoesToTheVideo(unittest.TestCase):
    """The letter lands in the browser's front tab, and nowhere else.

    Reported: a k typed into ChatGPT's chat box, with the browser in the
    background.  A background browser reports no keyboard focus at all --
    the first guard read that as safe -- while still delivering the
    letter to a composer that focuses itself.  The front tab's title
    still answers, and it names the only place the letter can go.
    """

    def controller(self, report):
        from control.backends import macos

        # target "auto" (empty), so the cautious front-tab guard is the
        # one under test; the trust-the-chosen-target tests set a
        # concrete target themselves.
        made = FakeMac(running={"Google Chrome"}, target="")
        # the real refusal logic, fed a controlled focus report
        made._refuse_to_type_into_a_text_box = (
            lambda pid: macos.MacOSController._refuse_to_type_into_a_text_box(
                made, pid))
        self._patch(macos, report)

        return made

    def _patch(self, macos, report):
        self._was = macos._focus_report
        macos._focus_report = lambda pid: report
        self.addCleanup(lambda: setattr(macos, "_focus_report", self._was))

    def test_a_background_browser_on_a_chat_tab_refuses(self):
        """The reported failure, exactly."""

        controller = self.controller(("none", "ChatGPT - Google Chrome"))

        with self.assertRaises(UnsupportedCommand):
            controller.play_pause()

        self.assertEqual(controller.keyed, [], "the letter was sent anyway")

    def test_a_background_browser_on_the_video_still_pauses(self):
        controller = self.controller(
            ("none", "Some Video - YouTube - Google Chrome"))

        controller.play_pause()

        self.assertEqual(controller.keyed, [4242])

    def test_an_editable_focus_refuses_whatever_the_tab(self):
        controller = self.controller(
            ("editable", "Search - YouTube - Google Chrome"))

        with self.assertRaises(UnsupportedCommand):
            controller.play_pause()

        self.assertEqual(controller.keyed, [])

    def test_a_video_not_yet_played_still_gets_the_letter(self):
        """Reported: the fist could pause a video but never start one.

        Until the video is clicked, the page's focus is its own container
        -- and a container's value is settable (its scroll position), so
        a probe that read "settable" as "takes typing" classified every
        fresh page as a text box.  A container is an element, not an
        editable, and the letter must flow.
        """

        controller = self.controller(
            ("element", "Fresh Video - YouTube - Google Chrome"))

        controller.play_pause()

        self.assertEqual(controller.keyed, [4242])

    def test_the_video_tab_with_ordinary_focus_gets_the_letter(self):
        controller = self.controller(
            ("element", "Some Video - YouTube - Google Chrome"))

        controller.play_pause()

        self.assertEqual(controller.keyed, [4242])

    def test_a_chosen_target_trusts_you_past_the_front_tab(self):
        """The user's words: once a target is chosen, control the chosen
        one.  A pinned or resolved browser sends the letter even to a
        background video tab -- the person said which app they mean."""

        controller = self.controller(
            ("none", "Some Other Tab - Google Chrome"))
        controller.config.target_app = "Google Chrome"   # chosen, not auto

        controller.play_pause()

        self.assertEqual(controller.keyed, [4242],
                         "a chosen target should not refuse a background tab")

    def test_a_chosen_target_still_refuses_a_text_box(self):
        """Trust does not extend to typing into a search field: k there
        is a literal k, never a play/pause, whoever aimed."""

        controller = self.controller(
            ("editable", "Search - YouTube - Google Chrome"))
        controller.config.target_app = "Google Chrome"

        with self.assertRaises(UnsupportedCommand):
            controller.play_pause()

    def test_a_front_tab_that_is_not_the_video_refuses(self):
        """Even with harmless focus: the letter can only go to that tab,
        where it is at best useless."""

        controller = self.controller(("element", "ChatGPT - Google Chrome"))

        with self.assertRaises(UnsupportedCommand):
            controller.play_pause()

    def test_unknown_keeps_the_old_behaviour(self):
        controller = self.controller(("unknown", None))

        controller.play_pause()

        self.assertEqual(controller.keyed, [4242])

    def test_other_sites_are_a_setting_away(self):
        from control.backends import macos

        made = FakeMac(running={"Google Chrome"}, target="Google Chrome")
        made.config = ControlConfig(target_app="Google Chrome",
                                    browser_play_key="k",
                                    browser_video_titles="youtube, vimeo")
        made._refuse_to_type_into_a_text_box = (
            lambda pid: macos.MacOSController._refuse_to_type_into_a_text_box(
                made, pid))
        self._patch(macos, ("element", "A Film on Vimeo - Google Chrome"))

        made.play_pause()

        self.assertEqual(made.keyed, [4242])


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
        """The pair, in the order anyone would do them.

        The second press resumes through the system key, not the
        letter: QRUDO paused it and so knows a player is sitting there,
        which is the case the letter used to have to cover because
        nothing tracked it.  Resuming on the same route that paused
        reaches the same tab, wherever it sits."""

        controller = FakeMac(running={"Google Chrome"}, playing=True)
        self.assertIn("paused", controller.play_pause())

        controller.playing = False          # it stopped, because we stopped it
        self.assertIn("resumed", controller.play_pause())
        self.assertEqual(len(controller.media_keys), 2)

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
