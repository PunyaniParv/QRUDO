"""Tests for the voice-to-control fast path (Milestone: voice intent router).

Pure tests: they exercise VoiceIntentRouter and the dry-run router->ControlEngine
routing via NullController, so they run with no mic, no Bluetooth device, and
no PICOVOICE credentials / .ppn file.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep the control logger out of the live logs/ directory during tests.
from control import log

log.setup(tempfile.mkdtemp(), console=False)

from control import Command, ControlConfig, ControlEngine
from control import catalog
from control.actions import parse
from control.backends.null import NullController
from voice.bridge import VoiceIntentRouter
from voice.bridge import _SUPPORTED  # noqa: E402


class VoiceIntentRouterCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = VoiceIntentRouter()

    def assert_maps(self, text, expected):
        """Assert several spellings (case/padding variants) map to ``expected``."""
        for phrase in (text, text.upper(), "  " + text + "  "):
            with self.subTest(phrase=phrase):
                self.assertIs(self.router.classify(phrase), expected)

    def assert_none(self, *phrases):
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertIsNone(self.router.classify(phrase))

    # -- 1. every supported command has phrases ------------------------------
    def test_every_supported_command_has_phrases(self):
        covered = {
            Command.VOLUME_UP: "volume up",
            Command.VOLUME_DOWN: "volume down",
            Command.PLAY_PAUSE: "pause",
            Command.REWIND: "rewind",
            Command.FORWARD: "forward",
            Command.BRIGHTNESS_UP: "increase brightness",
            Command.BRIGHTNESS_DOWN: "decrease brightness",
            Command.TARGET_NEXT: "next target",
            Command.TARGET_PREV: "previous target",
        }
        # The router promises exactly these commands, nothing more.
        self.assertEqual(set(_SUPPORTED), set(covered))
        for command, phrase in covered.items():
            with self.subTest(command=command):
                self.assert_maps(phrase, command)

    # -- 2. case-insensitivity ----------------------------------------------
    def test_case_insensitive(self):
        self.assert_maps("PaUsE tHe MuSiC", Command.PLAY_PAUSE)
        self.assert_maps("VOLUME UP", Command.VOLUME_UP)
        self.assert_maps("DeCrEaSe BrIgHtNeSs", Command.BRIGHTNESS_DOWN)

    # -- 3. punctuation / whitespace ----------------------------------------
    def test_punctuation_and_whitespace(self):
        self.assert_maps("turn the volume up!!", Command.VOLUME_UP)
        self.assert_maps("...pause...", Command.PLAY_PAUSE)
        self.assert_maps("fast forward, now", Command.FORWARD)
        self.assert_maps("  \t\n  rewind  \n", Command.REWIND)

    # -- 4. natural-language variations -------------------------------------
    def test_volume_up_variations(self):
        self.assert_maps("volume up", Command.VOLUME_UP)
        self.assert_maps("turn the volume up", Command.VOLUME_UP)
        self.assert_maps("increase volume", Command.VOLUME_UP)
        self.assert_maps("increase the volume", Command.VOLUME_UP)
        self.assert_maps("raise the volume", Command.VOLUME_UP)
        self.assert_maps("turn it up", Command.VOLUME_UP)
        self.assert_maps("make it louder", Command.VOLUME_UP)
        self.assert_maps("crank it up", Command.VOLUME_UP)

    def test_volume_down_variations(self):
        self.assert_maps("volume down", Command.VOLUME_DOWN)
        self.assert_maps("turn the volume down", Command.VOLUME_DOWN)
        self.assert_maps("decrease volume", Command.VOLUME_DOWN)
        self.assert_maps("lower the volume", Command.VOLUME_DOWN)
        self.assert_maps("reduce volume", Command.VOLUME_DOWN)
        self.assert_maps("turn it down", Command.VOLUME_DOWN)
        self.assert_maps("make it quieter", Command.VOLUME_DOWN)

    def test_play_pause_variations(self):
        self.assert_maps("pause", Command.PLAY_PAUSE)
        self.assert_maps("pause the music", Command.PLAY_PAUSE)
        self.assert_maps("play pause", Command.PLAY_PAUSE)
        self.assert_maps("play music", Command.PLAY_PAUSE)

    def test_rewind_variations(self):
        self.assert_maps("rewind", Command.REWIND)
        self.assert_maps("go back", Command.REWIND)
        self.assert_maps("rewind the video", Command.REWIND)
        self.assert_maps("skip back", Command.REWIND)

    def test_forward_variations(self):
        self.assert_maps("forward", Command.FORWARD)
        self.assert_maps("skip forward", Command.FORWARD)
        self.assert_maps("fast forward", Command.FORWARD)

    def test_brightness_up_variations(self):
        self.assert_maps("increase brightness", Command.BRIGHTNESS_UP)
        self.assert_maps("brightness up", Command.BRIGHTNESS_UP)
        self.assert_maps("make the screen brighter", Command.BRIGHTNESS_UP)
        self.assert_maps("make it brighter", Command.BRIGHTNESS_UP)

    def test_brightness_down_variations(self):
        self.assert_maps("decrease brightness", Command.BRIGHTNESS_DOWN)
        self.assert_maps("brightness down", Command.BRIGHTNESS_DOWN)
        self.assert_maps("make the screen darker", Command.BRIGHTNESS_DOWN)
        self.assert_maps("make it darker", Command.BRIGHTNESS_DOWN)

    def test_target_next_variations(self):
        self.assert_maps("next target", Command.TARGET_NEXT)
        self.assert_maps("next app", Command.TARGET_NEXT)
        self.assert_maps("switch app", Command.TARGET_NEXT)
        self.assert_maps("switch to next app", Command.TARGET_NEXT)

    def test_target_prev_variations(self):
        self.assert_maps("previous target", Command.TARGET_PREV)
        self.assert_maps("previous app", Command.TARGET_PREV)
        self.assert_maps("prev app", Command.TARGET_PREV)
        self.assert_maps("switch to previous app", Command.TARGET_PREV)


    # -- 5. empty input ----------------------------------------------------
    def test_empty_input_returns_none(self):
        self.assert_none("", "   ", "\t\n", None, "!!!", "...")

    # -- 6. unknown/gibberish input ----------------------------------------
    def test_gibberish_returns_none(self):
        self.assert_none("blue", "aardvark xyzzy", "12345", "the")

    # -- 7. unsupported capabilities return None ----------------------------
    def test_unsupported_capabilities_return_none(self):
        # Capabilities QRUDO still has no route for -- and never a route
        # through the built-in vocabulary.  (Phrases that now name a catalog
        # job -- "open Chrome", "mute", "next song" -- resolve through
        # route() into a CUSTOM payload and are tested there; classify()
        # still answers None for them, as it always did.)
        self.assert_none(
            "close Chrome",
            "close this tab",
            "open YouTube",
            "type hello",
            "click here",
            "move the mouse",
            "take a screenshot",
            "shutdown the computer",
            "turn off",
            "sleep",
            "lock the screen",
            "open the browser",
            "search the web",
            "what is the weather",
            "send an email",
        )

    # -- 8. does not accidentally map unrelated phrases ----------------------
    def test_no_stray_word_mapping(self):
        # Words like "up"/"down"/"open" alone must not trigger anything.
        self.assert_none(
            "up",
            "down",
            "open",
            "close",
            "the up is on the left",
            "sound of the down",
        )

    def test_next_previous_song_are_not_targets(self):
        # Skipping songs is NOT the TARGET_NEXT/TARGET_PREV semantics (which
        # moves where commands are aimed), so these must be None.
        self.assert_none("next song", "previous song", "skip to next song",
                         "skip to previous song")

    # -- 9. exact Command enum identity --------------------------------------
    def test_matched_phrases_produce_correct_enum(self):
        self.assertIs(self.router.classify("volume up"), Command.VOLUME_UP)
        self.assertIs(self.router.classify("pause"), Command.PLAY_PAUSE)
        self.assertIs(self.router.classify("brightness up"),
                      Command.BRIGHTNESS_UP)


class CatalogVoiceRoutingCase(unittest.TestCase):
    """Phrases that name a catalog job resolve to the existing CUSTOM payload.

    The router never re-defines what a job does: it maps a phrase to the
    catalog's job *name*, and control/catalog turns that into the exact
    action chain a taught gesture would carry.  Unknown or broken jobs are
    unhandled, never an error.
    """

    def setUp(self):
        self.router = VoiceIntentRouter()

    def _custom_payload(self, text):
        """Assert ``text`` routes to a CUSTOM command; return its payload."""
        route = self.router.route(text)
        self.assertIsNotNone(route)
        self.assertEqual(route.command, Command.CUSTOM)
        return route.payload

    # -- the task's two examples ------------------------------------------
    def test_next_track_resolves_to_the_catalog_keystroke(self):
        payload = self._custom_payload("next track")
        self.assertEqual(parse(payload),
                         [{"type": "keystroke", "combo": "shift+n"}])

    def test_next_song_means_next_track(self):
        payload = self._custom_payload("next song")
        self.assertEqual(parse(payload),
                         [{"type": "keystroke", "combo": "shift+n"}])

    def test_open_chrome_resolves_to_the_catalog_open_app(self):
        payload = self._custom_payload("open Chrome")
        self.assertEqual(parse(payload),
                         [{"type": "open_app", "app": "Google Chrome"}])

    def test_launch_chrome_means_open_chrome(self):
        payload = self._custom_payload("launch chrome")
        self.assertEqual(parse(payload)[0]["app"], "Google Chrome")

    # -- the payload is the existing serialised action chain ---------------
    def test_custom_payload_is_exactly_the_serialized_chain(self):
        from control.actions import serialize

        payload = self._custom_payload("next track")
        self.assertEqual(
            payload, serialize({"type": "keystroke", "combo": "shift+n"}))

    def test_every_catalog_job_is_reachable_by_its_own_name(self):
        """A job added to the catalog is speakable without editing the
        router -- the exact-name fallback resolves it the same way."""
        for job in catalog.job_names():
            with self.subTest(job=job):
                self.assertIsNotNone(self.router.route(job.lower()))

    def test_builtin_catalog_jobs_stay_plain_commands(self):
        """"Play / pause" exists in the catalog as a builtin, but a phrase
        the built-in vocabulary owns must keep its plain Command -- never
        wrap itself in a CUSTOM payload."""
        route = self.router.route("play pause")
        self.assertIs(route.command, Command.PLAY_PAUSE)
        self.assertEqual(route.payload, "")

    # -- unknown / malformed stay unhandled --------------------------------
    def test_unknown_phrases_stay_unhandled(self):
        self.assertIsNone(self.router.route("open Televator"))
        self.assertIsNone(self.router.route("xyzzy"))
        self.assertIsNone(self.router.route("next: nothing"))

    def test_an_unavailable_catalog_job_is_unhandled(self):
        with mock.patch.object(catalog, "resolve", return_value=None):
            self.assertIsNone(self.router.route("next track"))

    def test_a_malformed_catalog_action_is_unhandled(self):
        # A keystroke action with no "combo" cannot serialise; the router
        # must treat it as unknown rather than raise.
        with mock.patch.object(catalog, "resolve",
                               return_value={"type": "keystroke"}):
            self.assertIsNone(self.router.route("next track"))

    # -- the classify() view keeps its contract ----------------------------
    def test_classify_keeps_its_plain_command_contract(self):
        # A catalog route carries a payload, so the payload-free classify()
        # view answers None -- exactly as it did before the catalog existed.
        self.assertIsNone(self.router.classify("next track"))
        self.assertIsNone(self.router.classify("open chrome"))
        self.assertIs(self.router.classify("volume up"), Command.VOLUME_UP)

    # -- built-in phrases are untouched by the catalog ---------------------
    def test_route_matches_the_built_in_vocabulary(self):
        route = self.router.route("turn the volume up")
        self.assertIs(route.command, Command.VOLUME_UP)
        self.assertEqual(route.payload, "")


class VoiceToControlRoutingCase(unittest.TestCase):
    """Dry-run: matched phrases reach the engine; unmatched ones do not.

    Uses NullController, so no OS action and no microphone is needed.
    """

    def _make_engine(self):
        config = ControlConfig(cooldown_seconds=0.0)
        controller = NullController(config)
        engine = ControlEngine(controller=controller, config=config)
        engine.targets.probe = lambda: {}
        return engine, controller

    def test_recognized_command_reaches_engine(self):
        router = VoiceIntentRouter()
        engine, controller = self._make_engine()
        with engine:
            command = router.classify("turn the volume up")
            engine.submit(command)
            time.sleep(0.2)
        self.assertIn("volume +10%", controller.calls)

    def test_unrecognized_text_does_not_reach_engine(self):
        router = VoiceIntentRouter()
        engine, controller = self._make_engine()
        with engine:
            self.assertIsNone(router.classify("open Chrome"))
        self.assertEqual(controller.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

