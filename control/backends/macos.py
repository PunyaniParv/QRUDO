"""macOS implementation of the control layer.

Three different mechanisms, because macOS has no single "control the machine"
API:

* **Volume** -- AppleScript (``osascript``).  Always available, no permissions.
* **Brightness** -- the private ``DisplayServices`` framework via ctypes.
  Works on the built-in display without any extra install; falls back to the
  ``brightness`` Homebrew CLI, then to the brightness HID keys.
* **Media (play/pause, seek)** -- synthetic keyboard events via Quartz, which
  is what a real keyboard's media keys send.  This keeps us
  application-independent (spec C) but needs Accessibility permission for the
  app running QRUDO (Terminal / VS Code / Python.app).
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess

from ..commands import NO_CHANGE
from ..config import ControlConfig
from ..executor import Controller, ControlError, UnsupportedCommand
from ..log import get_logger

# --- HID "special key" codes, from IOKit's ev_keymap.h -----------------------
NX_KEYTYPE_BRIGHTNESS_UP = 2
NX_KEYTYPE_BRIGHTNESS_DOWN = 3
NX_KEYTYPE_PLAY = 16
NX_KEYTYPE_NEXT = 17
NX_KEYTYPE_PREVIOUS = 18

# --- Virtual key codes for ordinary keys ------------------------------------
KEY_J = 38          # YouTube's back ten seconds
KEY_K = 40          # YouTube's play/pause
KEY_L = 37          # YouTube's forward ten seconds
KEY_SPACE = 49      # most other players'
KEY_LEFT_ARROW = 123
KEY_RIGHT_ARROW = 124

_OSASCRIPT_TIMEOUT = 5.0


class MacOSController(Controller):
    name = "macOS"

    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or ControlConfig()
        self.log = get_logger("macos")
        self._quartz = _load_quartz()
        self._display_services = _load_display_services()
        self._core_audio = _load_core_audio()
        self._workspace = _load_workspace()
        self._brightness_cli = shutil.which("brightness")

    # ------------------------------------------------------------------ volume

    def volume_up(self, step: int) -> str:
        return self._change_volume(step)

    def volume_down(self, step: int) -> str:
        return self._change_volume(-step)

    def _change_volume(self, delta: int) -> str:
        """CoreAudio if we can, AppleScript if we must."""
        if self._core_audio is not None:
            detail = self._change_volume_fast(delta)
            if detail is not None:
                return detail
        return self._change_volume_applescript(delta)

    def _change_volume_fast(self, delta: int) -> str | None:
        """~0.1 ms.  Returns None if this device cannot be driven this way."""
        device = self._core_audio.device()
        if device is None:
            return None
        current = self._core_audio.get_volume(device)
        if current is None:
            return None

        if self._core_audio.is_muted(device) and delta > 0 and self.config.unmute_on_volume_up:
            if not self._core_audio.unmute(device):
                return None
            return f"unmuted (volume {current * 100:.0f}%)"

        target = min(1.0, max(0.0, current + delta / 100.0))
        old, new = round(current * 100), round(target * 100)
        if old == new:
            return f"volume {NO_CHANGE} {'maximum' if delta > 0 else 'minimum'} ({old}%)"
        if not self._core_audio.set_volume(device, target):
            return None
        return f"volume {old}% -> {new}%"

    def _change_volume_applescript(self, delta: int) -> str:
        """Read and write the volume in a single osascript call.

        Two calls would take ~400 ms (most of it process startup) and could
        interleave with the user's own volume keys; one call is ~200 ms and is
        atomic.
        """
        unmute = "true" if self.config.unmute_on_volume_up else "false"
        raw = self._osascript(f"""
            set s to (get volume settings)
            set cur to output volume of s
            if (output muted of s) and {delta} > 0 and {unmute} then
                set volume without output muted
                return (cur as text) & "|" & (cur as text) & "|unmuted"
            end if
            set target to cur + ({delta})
            if target > 100 then set target to 100
            if target < 0 then set target to 0
            set volume output volume target
            return (cur as text) & "|" & (target as text) & "|ok"
        """)
        try:
            old, new, outcome = raw.split("|")
        except ValueError as exc:
            raise ControlError(f"unexpected volume output {raw!r}") from exc
        if outcome == "unmuted":
            return f"unmuted (volume {old}%)"
        if old == new:
            return f"volume {NO_CHANGE} {'maximum' if delta > 0 else 'minimum'} ({old}%)"
        return f"volume {old}% -> {new}%"

    def _volume_settings(self) -> dict:
        # "output volume:63, input volume:27, alert volume:100, output muted:false"
        raw = self._osascript("get volume settings")
        parsed = {}
        for chunk in raw.split(","):
            key, _, value = chunk.partition(":")
            parsed[key.strip()] = value.strip()
        try:
            volume = int(parsed["output volume"])
        except (KeyError, ValueError) as exc:
            raise ControlError(f"could not read output volume from {raw!r}") from exc
        return {"volume": volume, "muted": parsed.get("output muted") == "true"}

    # -------------------------------------------------------------- brightness

    def brightness_up(self, step: int) -> str:
        return self._nudge_brightness(step)

    def brightness_down(self, step: int) -> str:
        return self._nudge_brightness(-step)

    def _nudge_brightness(self, delta_percent: int) -> str:
        """Try the precise API first, then the CLI, then the HID keys.

        Only the first gives us a readable before/after level, which is why it
        is preferred -- the demo is much easier to narrate with real numbers.
        """
        if self._display_services is not None:
            current = self._get_brightness()
            if current is not None:
                target = min(1.0, max(0.0, current + delta_percent / 100.0))
                if abs(target - current) < 0.005:
                    # Already at the end of the scale.  Say so explicitly, so
                    # the self-test knows there is nothing to undo.
                    return (f"brightness {NO_CHANGE} "
                            f"{'maximum' if delta_percent > 0 else 'minimum'} "
                            f"({current * 100:.0f}%)")
                if self._set_brightness(target):
                    return f"brightness {current * 100:.0f}% -> {target * 100:.0f}%"

        if self._brightness_cli:
            direction = "+" if delta_percent > 0 else "-"
            amount = abs(delta_percent) / 100.0
            self._run([self._brightness_cli, f"{direction}{amount}"])
            return f"brightness {direction}{abs(delta_percent)}% (brightness CLI)"

        # Last resort: the keyboard's brightness keys.  No readback, and some
        # external monitors ignore them entirely.
        key = NX_KEYTYPE_BRIGHTNESS_UP if delta_percent > 0 else NX_KEYTYPE_BRIGHTNESS_DOWN
        self._post_media_key(key)
        return f"brightness key {'up' if delta_percent > 0 else 'down'} (one hardware step)"

    def _get_brightness(self) -> float | None:
        level = ctypes.c_float()
        rc = self._display_services.DisplayServicesGetBrightness(
            self._main_display_id(), ctypes.byref(level))
        return level.value if rc == 0 else None

    def _set_brightness(self, level: float) -> bool:
        return self._display_services.DisplayServicesSetBrightness(
            self._main_display_id(), ctypes.c_float(level)) == 0

    def _main_display_id(self) -> int:
        core_graphics = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        core_graphics.CGMainDisplayID.restype = ctypes.c_uint32
        return core_graphics.CGMainDisplayID()

    # ------------------------------------------------------------------- media

    def play_pause(self) -> str:
        """Play or pause, in the named app if there is one.

        The media key is the obvious way to do this and works in any app,
        but it is a message to the system rather than to a player: with
        nothing currently playing, macOS answers it by opening Music.
        Naming the app avoids that, and also settles which one gets it when
        both a browser and Spotify are open.
        """

        target = self.config.app

        if target:
            played = self._play_pause_app(target)

            if played is not None:
                return played

        # No app named: find one rather than fall back to the media key.
        #
        # The media key is a message to the system, not to a player, and
        # macOS answers one with nothing currently playing by opening
        # Music.  Checking that some player is running is not enough --
        # a browser can be open without playing anything, and the key
        # still goes to Music.  So it is not used here at all.
        running = self._running_apps()

        for name in self.SCRIPTABLE:
            if name not in running or name in self.NOT_BY_ACCIDENT:
                continue

            played = self._play_pause_app(name)

            if played is not None:
                return played

        for name in self.BROWSERS:
            pid = self._target_pid(name)

            if pid is not None:
                return self._play_pause_key(name, pid)

        raise UnsupportedCommand(
            "nothing to play or pause -- open a player, or name one as "
            "target_app in qrudo_config.json")

    #: Players that can simply be told, without being brought forward.
    SCRIPTABLE = ("Spotify", "Music", "VLC", "QuickTime Player", "TV")

    #: Players that being open is no evidence for.  macOS opens Music by
    #: itself in answer to a media key with nothing playing, and once it
    #: is open it is the first scriptable player found -- so every fist
    #: from then on drives Music, which is what happened.  It was not
    #: chosen and its being open does not mean it was.
    #:
    #: Naming one as target_app still reaches it.  This is only about
    #: what gets picked when nothing was named.
    NOT_BY_ACCIDENT = ("Music", "TV")

    #: Browsers, which take the video player's own keyboard shortcut.
    BROWSERS = ("Google Chrome", "Safari", "Firefox", "Microsoft Edge",
                "Arc", "Brave Browser", "Opera", "Vivaldi")

    def _running_apps(self) -> set:
        """What is open right now, without disturbing any of it."""

        if self._workspace is None:
            return set()

        return {
            app.localizedName()
            for app in self._workspace.sharedWorkspace().runningApplications()
            if app.localizedName()
        }

    def _play_pause_app(self, name: str) -> str | None:
        """Play/pause a named app, or None if it cannot be reached."""

        running = self._running_apps()

        # Players that can simply be told.  This is exact, and it neither
        # needs the app focused nor disturbs anything else.
        #
        # Whether it is running is settled before saying anything to it:
        # naming an app in AppleScript at all is enough to launch it, so
        # the obvious "if it is running then tell it" starts the very
        # thing it was written to avoid starting.
        for known in self.SCRIPTABLE:
            if known.lower() in name.lower():
                if known not in running:
                    return None

                self._osascript(f'tell application "{known}" to playpause')

                return f"play/pause in {known}"

        # A browser has no such interface, so use the video player's own
        # keyboard shortcut, delivered to the browser whether or not it is
        # focused.  k is YouTube's, and is safer than space, which scrolls
        # the page when the player is not selected.
        pid = self._target_pid(name)

        if pid is None:
            return None

        return self._play_pause_key(name, pid)

    def _play_pause_key(self, name: str, pid: int) -> str:
        """Play or pause a browser, by the surest route available.

        Typing a letter at the browser was the old default, and it is
        the source of three complaints at once: it lands in whatever
        tab is at the *front*, so the video has to be the front tab or
        nothing happens; the letter it types, ``k``, is next-track on
        YouTube Music rather than play/pause; and when the front tab is
        a chat box the letter is refused outright, which reads as the
        gesture doing nothing.

        The system's own now-playing key has none of those faults.  It
        reaches whichever tab actually holds the playing media, in any
        browser, wherever that tab sits -- and it is genuinely
        play/pause on YouTube Music.  Its one hazard is the empty case:
        sent with nothing playing and no app owning the now-playing
        role, macOS answers it by opening Music.  So it is used exactly
        when that cannot happen -- when something is playing now, or
        when QRUDO is the one that paused it and so knows a player is
        sitting there paused, waiting to be resumed.

        Only when the system reports nothing playing and QRUDO did not
        pause it does the letter come back, for the first press that
        starts a fresh, silent video -- and ``browser_play_key`` still
        forces either route by hand: "media" or "k"/"space".
        """

        wanted = self.config.browser_play_key.strip().lower()

        # A pinned TAB is controlled IN THE BACKGROUND: the screen must
        # not move, so no key can carry this -- keys only reach the
        # front tab -- and the media key follows whatever was already
        # playing.  The browser is scripted to drive that tab's video
        # directly, wherever the tab sits.
        pinned = getattr(self.config, "target_tab", "").strip()
        if pinned:
            return self._drive_pinned_tab(
                name, pinned,
                "var vs=document.querySelectorAll('video'),v=null,i;"
                "for(i=0;i<vs.length;i++){if(!vs[i].paused)"
                "{v=vs[i];break}}"
                "if(!v){for(i=0;i<vs.length;i++)"
                "{if(vs[i].currentTime>0){v=vs[i];break}}}"
                "if(!v&&vs.length){v=vs[0]}"
                "if(!v){'novideo'}else if(v.paused){v.play();'played'}"
                "else{v.pause();'paused'}")

        # No pin: the script route is still first, because it is the
        # only one that acts on the video itself.  A browser with no
        # video refuses plainly -- there is nothing a fist could have
        # meant -- and only a browser that cannot be scripted at all
        # falls to the letters and keys below.
        scripted = self._browser_toggle_by_script(name)

        if scripted:
            return scripted

        if scripted == "":
            raise UnsupportedCommand(
                f"no video to play or pause in {name} -- open one "
                f"first")

        if wanted in ("space", "spacebar", "k"):
            key = KEY_SPACE if wanted != "k" else KEY_K
            self._refuse_to_type_into_a_text_box(pid)
            self._post_key(key, to_pid=pid)

            return f"play/pause ({wanted}) to {name}"

        # "media" or unset: the system now-playing key, when it is safe.
        playing = self._audio_playing()

        # Safe only while something is AUDIBLE: a session provably
        # claims the key, so it pauses that and nothing else.
        if playing:
            self._post_media_key(NX_KEYTYPE_PLAY)
            self._paused_it = True

            return "paused what was playing"

        # Resuming is where Music kept sneaking in: "QRUDO paused it"
        # proves the pause happened, not that the paused thing still
        # EXISTS -- the video ends, the tab closes, and the resume key
        # lands in a void that macOS answers by opening Music.  So the
        # resume goes looking instead: the paused video is found by
        # script and played where it sits, background included.  Found
        # nothing?  Then it is genuinely gone, and the letter -- which
        # can never launch anything -- takes over.
        if self._paused_it:
            self._paused_it = False
            said = self._resume_paused_video(name)

            if said:
                return said

        if playing is None:
            # The audio question could not be asked, so the safe line
            # cannot be drawn -- fall to the letter, which at worst is
            # useless and never opens Music.
            self._refuse_to_type_into_a_text_box(pid)
            self._post_key(KEY_K, to_pid=pid)

            return f"play/pause (k) to {name}"

        # Nothing playing and we did not pause it: the first press on a
        # fresh video.  The letter starts it in the front tab; from the
        # next press on, it is audible and the now-playing route takes
        # over.
        self._refuse_to_type_into_a_text_box(pid)
        self._post_key(KEY_K, to_pid=pid)

        return f"play/pause (k) to {name} -- nothing was playing"

    def rewind(self, seconds: int) -> str:
        return self._seek(seconds, forward=False)

    def forward(self, seconds: int) -> str:
        return self._seek(seconds, forward=True)

    def _seek(self, seconds: int, *, forward: bool) -> str:
        direction = "forward" if forward else "back"

        if self.config.seek_mode == "track":
            # The track keys are media keys, and a media key with no
            # session claiming it is answered by macOS opening Music.
            # Skipping tracks only means anything while something is
            # playing, so that is the only time the key is sent.
            if not self._audio_playing():
                raise UnsupportedCommand(
                    "nothing is playing to skip -- start something "
                    "first")

            self._post_media_key(NX_KEYTYPE_NEXT if forward else NX_KEYTYPE_PREVIOUS)
            return f"{'next' if forward else 'previous'} track"

        delta = seconds if forward else -seconds
        seek_js = (
            f"var vs=document.querySelectorAll('video'),v=null,i;"
            f"for(i=0;i<vs.length;i++){{if(!vs[i].paused)"
            f"{{v=vs[i];break}}}}"
            f"if(!v){{for(i=0;i<vs.length;i++)"
            f"{{if(vs[i].currentTime>0){{v=vs[i];break}}}}}}"
            f"if(!v){{'novideo'}}else{{"
            f"v.currentTime=Math.max(0,v.currentTime+({delta}));"
            f"'sought'}}")

        # A pinned tab seeks in the background by the same script that
        # plays it -- arrow keys only ever reach the front tab.
        pinned = getattr(self.config, "target_tab", "").strip()
        if pinned:
            said = self._drive_pinned_tab(
                self.config.target_app or "the browser", pinned, seek_js)
            return said.replace("sought",
                                f"seek {direction} {seconds}s")

        # No pin, browser target: still by script -- gestures must not
        # depend on typing keys into pages.  The last-controlled tab
        # answers first; a browser that cannot be scripted falls to the
        # arrow keys below, exactly as before.
        browser = (self.config.target_app or "").strip()
        if browser in ("Google Chrome", "Safari"):
            last = getattr(self, "_last_media_title", None)
            if last:
                try:
                    said = self._drive_pinned_tab(browser, last, seek_js)
                    if said == "sought":
                        return f"seek {direction} {seconds}s"
                except UnsupportedCommand:
                    self._last_media_title = None

            said = self._browser_seek_by_script(browser, seek_js)
            if said:
                return f"seek {direction} {seconds}s " \
                       f'("{said if len(said) <= 28 else said[:27] + "…"}")'

        # Seek within the current track by repeating the player's own arrow-key
        # shortcut.  seek_step_seconds says how far one press moves; the config
        # turns that into "about `seconds` seconds".
        key, name, step = self._seek_key(forward)
        presses = max(1, round(seconds / step))
        pid = self._target_pid()

        for _ in range(presses):
            self._post_key(key, to_pid=pid)

        where = f" to {self.config.app}" if pid else ""

        return (f"seek {direction} ~{presses * step}s{where} "
                f"({presses}x {name}, {seconds}s requested)")

    #: Set when QRUDO was the one that paused, so it knows it may resume.
    _paused_it = False

    #: The title of the media tab last played, paused or sought, so the
    #: next gesture means THAT tab -- "the one I was just controlling"
    #: -- rather than whichever tab enumeration reaches first.
    _last_media_title = None

    def _media_play_pause(self, name: str, pid: int) -> str:
        """Play or pause, by whichever route is safe just now.

        Only reached when someone asks for the media key by name.  It is
        no longer the default, and the reason is worth keeping:

        Whether audio is playing and whether an app has claimed the
        system's now-playing role are different questions, and only the
        second decides where this key goes.  Sound can be coming out of
        something that never claimed it -- and with the role unclaimed,
        macOS answers the key by opening Music, which then holds the role
        and takes every media key after that.

        So the check below is a proxy for the thing that matters, and a
        proxy is what let Music open again after it had been fixed.  The
        honest answer is not to send a key the system may redirect.
        """

        if self._audio_playing():
            self._paused_it = True
            self._post_media_key(NX_KEYTYPE_PLAY)

            return "paused what was playing"

        self._paused_it = False
        self._refuse_to_type_into_a_text_box(pid)
        self._post_key(KEY_K, to_pid=pid)

        return f"play/pause (k) to {name} -- nothing was playing"

    def _audio_playing(self) -> bool | None:
        """Whether anything is coming out of the speakers.  None if unknown."""

        audio = getattr(self, "_core_audio", None)

        if audio is None:
            return None

        device = audio.device()

        return None if device is None else audio.is_playing(device)

    def _chosen_target(self) -> bool:
        """Whether QRUDO is aimed at a browser on purpose, not by guess.

        The user's words: once a target is chosen with ctrl+shift+arrows,
        the chosen one is what gets controlled.  A chosen target names a
        concrete app in the config -- "Google Chrome", not "auto" or
        empty -- because the resolver writes the settled name there.  So
        a concrete browser name is a deliberate aim, and a deliberate aim
        earns trust: the letter may go to that browser even when the
        video is a background tab, because the person said which app they
        mean.  The empty/auto case stays cautious, since there nobody
        said.
        """

        aimed = self.config.app

        return bool(aimed) and any(
            browser.lower() in aimed.lower() for browser in self.BROWSERS)

    def _refuse_to_type_into_a_text_box(self, pid: int) -> None:
        """A letter lands in the browser's front tab, wherever its focus
        is.  So it only goes when the front tab looks like the video and
        the focus is not something that takes typing.

        The first version refused only a focused text field, and a k
        landed in ChatGPT anyway: the browser was in the background,
        reporting no focus at all while still delivering the letter to a
        chat box that focuses itself.  The front tab's title answers even
        then, and it names the only place the letter can go -- so a tab
        that is not the video refuses, whatever the focus, because the
        letter is at best useless there and at worst typing.
        """

        if not pid:
            return

        kind, title = _focus_report(pid)

        # An editable focus refuses on every path, chosen target or not:
        # typing k into a search box is wrong whoever aimed there, and
        # the letter would be swallowed as text rather than reaching a
        # player anyway.
        if kind == "editable":
            raise UnsupportedCommand(
                "the cursor is in a text box, so play/pause would type "
                "into it -- click the video first")

        # A deliberately chosen browser is trusted past here: the person
        # pointed QRUDO at it with ctrl+shift+arrows (or pinned it), so
        # the letter goes even when the video is a background tab.  Only
        # the unaimed auto case still demands the front tab be the video,
        # because there nobody said which tab is meant.
        if self._chosen_target():
            return

        if kind in ("none", "element"):
            wanted = [part.strip().lower()
                      for part in self.config.browser_video_titles.split(",")
                      if part.strip()]

            if title is None or not any(part in title.lower()
                                        for part in wanted):
                where = f'"{title}"' if title else "not the video"

                raise UnsupportedCommand(
                    f"the browser's front tab is {where}, and the "
                    f"play/pause letter can only go there -- switch to "
                    f"the video tab first")

    def _seek_key(self, forward: bool) -> tuple[int, str, int]:
        """Which key seeks, what to call it, and how far one press moves.

        Sites do not agree.  Arrow keys move five seconds on YouTube; j and
        l move ten.  A key that does not seek on a given site is rarely
        idle there -- it usually does something else -- which is why this
        is a setting rather than a guess.
        """

        if self.config.browser_seek_keys.strip().lower() == "jl":
            return (KEY_L if forward else KEY_J), "j/l", 10

        return ((KEY_RIGHT_ARROW if forward else KEY_LEFT_ARROW), "arrow",
                max(1, self.config.seek_step_seconds))

    def _target_pid(self, name: str | None = None) -> int | None:
        """The process to aim keys at, or None to use the focused window."""
        name = (name if name is not None else self.config.app).strip()
        if not name or self._workspace is None:
            return None
        wanted = name.lower()
        for app in self._workspace.sharedWorkspace().runningApplications():
            running = app.localizedName()
            if running and wanted in running.lower():
                return app.processIdentifier()
        self.log.warning("target_app %r is not running; "
                         "sending to the focused window instead", name)
        return None

    def _applescript_play_pause(self) -> str:
        """Fallback when Quartz is unavailable: drive a known player directly."""
        for app in ("Spotify", "Music", "VLC", "QuickTime Player"):
            script = (
                f'if application "{app}" is running then\n'
                f'  tell application "{app}" to playpause\n'
                f'  return "ok"\n'
                f'end if\n'
                f'return ""'
            )
            if self._osascript(script) == "ok":
                return f"play/pause via {app}"
        raise UnsupportedCommand(
            "no media key support (install pyobjc-framework-Quartz) and no known player running")

    def _browser_toggle_by_script(self, browser: str):
        """Play/pause the browser's video BY SCRIPT, in one pass.

        The whole letters-and-media-key tangle existed because QRUDO
        could not talk to the video itself; with scripting allowed it
        can, so this is the primary route.  Playing video found: pause
        it.  Otherwise the best paused one (real progress first, a
        restored 0:00 as second chance): play it.  Nothing typed, no
        media key, no void for Music to answer.

        Returns the sentence to show, "" when the browser has no video
        at all (the caller refuses plainly), or None when scripting is
        unavailable (the caller falls back to the old routes).
        """

        if browser not in ("Google Chrome", "Safari"):
            return None

        # The tab acted on LAST answers first: with several media tabs
        # in one window, a fist means "the one I was just controlling",
        # not whichever tab enumeration happens to reach first.
        last = getattr(self, "_last_media_title", None)
        if last:
            try:
                said = self._drive_pinned_tab(
                    browser, last,
                    "var vs=document.querySelectorAll('video'),v=null,i;"
                    "for(i=0;i<vs.length;i++){if(!vs[i].paused)"
                    "{v=vs[i];break}}"
                    "if(!v){for(i=0;i<vs.length;i++)"
                    "{if(vs[i].currentTime>0){v=vs[i];break}}}"
                    "if(!v&&vs.length){v=vs[0]}"
                    "if(!v){'novideo'}else if(v.paused){v.play();"
                    "'played'}else{v.pause();'paused'}")
                if said:
                    return said
            except UnsupportedCommand:
                self._last_media_title = None   # gone; the pass decides

        # ALL of a page's videos, not the first: a Shorts feed keeps a
        # stack of preloaded <video> elements, and querySelector kept
        # answering for a preloaded one while the short actually
        # playing went untouched.
        act_js = (
            "var vs=document.querySelectorAll('video'),v=null,i;"
            "for(i=0;i<vs.length;i++){if(!vs[i].paused){v=vs[i];break}}"
            "if(v){v.pause();'pausednow'}else{"
            "for(i=0;i<vs.length;i++){if(vs[i].currentTime>0)"
            "{v=vs[i];break}}"
            "if(v){'p1'}else if(vs.length){'p0'}else{'no'}}")
        play_js = (
            "var vs=document.querySelectorAll('video'),v=null,i;"
            "for(i=0;i<vs.length;i++){if(vs[i].currentTime>0)"
            "{v=vs[i];break}}"
            "if(!v&&vs.length){v=vs[0]}"
            "if(v){v.play()};'ok'")

        if browser == "Safari":
            field = "name"
            act = f'do JavaScript "{act_js}" in (tab t of window w)'
            play = f'do JavaScript "{play_js}" in (tab fbT of window fbW)'
        else:
            field = "title"
            act = f'execute (tab t of window w) javascript "{act_js}"'
            play = (f'execute (tab fbT of window fbW) javascript '
                    f'"{play_js}"')

        script = f'''
        set fbW to 0
        set fbT to 0
        set fb0W to 0
        set fb0T to 0
        set blocked to 0
        tell application "{browser}"
            repeat with w from 1 to count of windows
                repeat with t from 1 to count of tabs of window w
                    try
                        set r to {act}
                        if r is "pausednow" then
                            return "paused|" & ({field} of tab t of window w)
                        end if
                        if r is "p1" and fbW is 0 then
                            set fbW to w
                            set fbT to t
                        end if
                        if r is "p0" and fb0W is 0 then
                            set fb0W to w
                            set fb0T to t
                        end if
                    on error m
                        if m contains "turned off" then set blocked to 1
                    end try
                end repeat
            end repeat
            if fbW is 0 then
                set fbW to fb0W
                set fbT to fb0T
            end if
            if fbW > 0 then
                {play}
                return "played|" & ({field} of tab fbT of window fbW)
            end if
        end tell
        if blocked is 1 then return "OFF"
        return ""
        '''

        try:
            proc = subprocess.run(["osascript", "-e", script],
                                  capture_output=True, text=True,
                                  timeout=10.0)
        except Exception:
            return None

        answer = (proc.stdout or "").strip()

        if proc.returncode != 0 or answer == "OFF":
            return None

        if not answer:
            return ""

        verb, _, title = answer.partition("|")
        # Remembered, so the next fist means THIS tab again.
        self._last_media_title = title
        short = title if len(title) <= 28 else title[:27] + "…"

        return f'{verb} "{short}"'

    def _browser_seek_by_script(self, browser: str, seek_js: str):
        """Seek whichever tab holds the live video, by script.

        One pass: the first tab whose video answers "sought" wins, and
        its title is remembered as the last-controlled tab.  Returns
        the title, or None when nothing could be sought.
        """

        safe_js = seek_js.replace("\\", "\\\\").replace('"', '\\"')

        if browser == "Safari":
            field = "name"
            run = f'do JavaScript "{safe_js}" in (tab t of window w)'
        else:
            field = "title"
            run = f'execute (tab t of window w) javascript "{safe_js}"'

        script = f'''
        tell application "{browser}"
            repeat with w from 1 to count of windows
                repeat with t from 1 to count of tabs of window w
                    try
                        if ({run}) is "sought" then
                            return {field} of tab t of window w
                        end if
                    end try
                end repeat
            end repeat
        end tell
        return ""
        '''

        try:
            proc = subprocess.run(["osascript", "-e", script],
                                  capture_output=True, text=True,
                                  timeout=10.0)
        except Exception:
            return None

        title = (proc.stdout or "").strip()

        if proc.returncode != 0 or not title:
            return None

        self._last_media_title = title

        return title

    def _resume_paused_video(self, name: str) -> str | None:
        """Find the video QRUDO paused and play it where it sits.

        One pass over the browser's tabs: the first paused video with
        real progress (currentTime > 0 -- a video someone was actually
        watching, not a thumbnail) is played by script, background or
        not.  Returns None when nothing qualifies or the browser is
        not scriptable, and the caller falls back to the letter --
        which can never launch Music.
        """

        browser = self.config.target_app or "Google Chrome"

        probe_js = ("var v=document.querySelector('video');"
                    "if(v&&v.paused){if(v.currentTime>0){v.play();"
                    "'resumed'}else{'paused0'}}else{'no'}")
        play_js = ("var v=document.querySelector('video');"
                   "if(v){v.play()};'ok'")

        if browser == "Safari":
            field = "name"
            probe = f'do JavaScript "{probe_js}" in (tab t of window w)'
            play = (f'do JavaScript "{play_js}" in '
                    f'(tab fbT of window fbW)')
        else:
            field = "title"
            probe = (f'execute (tab t of window w) javascript '
                     f'"{probe_js}"')
            play = (f'execute (tab fbT of window fbW) javascript '
                    f'"{play_js}"')

        # First choice: a paused video with real progress -- the one
        # somebody was actually watching.  Second chance: a paused
        # video at 0:00, which is what a restored session leaves
        # behind.  Either way the play happens IN the tab, so nothing
        # is ever posted into a void.
        script = f'''
        set fbW to 0
        set fbT to 0
        tell application "{browser}"
            repeat with w from 1 to count of windows
                repeat with t from 1 to count of tabs of window w
                    try
                        set r to {probe}
                        if r is "resumed" then
                            return {field} of tab t of window w
                        end if
                        if r is "paused0" and fbW is 0 then
                            set fbW to w
                            set fbT to t
                        end if
                    end try
                end repeat
            end repeat
            if fbW > 0 then
                {play}
                return {field} of tab fbT of window fbW
            end if
        end tell
        return ""
        '''

        try:
            proc = subprocess.run(["osascript", "-e", script],
                                  capture_output=True, text=True,
                                  timeout=8.0)
        except Exception:
            return None

        title = (proc.stdout or "").strip()

        if proc.returncode != 0 or not title:
            return None

        short = title if len(title) <= 28 else title[:27] + "…"

        return f'resumed "{short}" where it sat'

    def _drive_pinned_tab(self, name: str, title: str, js: str) -> str:
        """Run one line of JavaScript in the pinned tab, wherever it
        sits -- background control, nothing moves on screen.

        The tab is found at fire time by MATCHING titles, not equality:
        YouTube retitles tabs constantly -- "(1) " notification
        prefixes come and go, autoplay swaps the video -- and an exact
        lookup answered "the pinned tab is gone" about a tab sitting
        right there.  The matching happens in Python over an enumerated
        list, which also survives titles containing quotes.  A tab
        that is truly gone clears the pin, so the NEXT gesture falls
        back to normal behaviour instead of failing forever.
        """

        browser = self.config.target_app or "Google Chrome"

        field = "name" if browser == "Safari" else "title"
        listing = f'''
        tell application "{browser}"
            set out to ""
            repeat with w from 1 to count of windows
                repeat with t from 1 to count of tabs of window w
                    set out to out & w & "|" & t & "|" & ({field} of tab t of window w) & linefeed
                end repeat
            end repeat
            return out
        end tell
        '''

        try:
            lines = subprocess.run(
                ["osascript", "-e", listing],
                capture_output=True, text=True, timeout=5.0,
            ).stdout
        except Exception as exc:
            raise UnsupportedCommand(f"could not reach {browser}: {exc}")

        tabs = []
        for line in (lines or "").splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                try:
                    tabs.append((int(parts[0]), int(parts[1]),
                                 parts[2].strip()))
                except ValueError:
                    continue

        hit = _match_tab_title(title, [t[2] for t in tabs])
        short = title if len(title) <= 28 else title[:27] + "…"

        if hit is None:
            self.config.target_tab = ""
            raise UnsupportedCommand(
                f'the pinned tab ("{short}") is gone -- pin cleared, '
                f"the next gesture works normally; point again to pin "
                f"a new tab")

        window, index, _ = tabs[hit]
        safe_js = js.replace("\\", "\\\\").replace('"', '\\"')

        if browser == "Safari":
            run = (f'do JavaScript "{safe_js}" in '
                   f'(tab {index} of window {window})')
        else:
            run = (f'execute (tab {index} of window {window}) '
                   f'javascript "{safe_js}"')

        try:
            proc = subprocess.run(
                ["osascript", "-e",
                 f'tell application "{browser}" to {run}'],
                capture_output=True, text=True, timeout=5.0)
        except Exception as exc:
            raise UnsupportedCommand(f"could not reach {browser}: {exc}")

        answer = (proc.stdout or "").strip()

        if proc.returncode != 0 or not answer:
            stderr = (proc.stderr or "").strip()

            # Only claim the switch is off when the browser SAID so --
            # a sleeping tab or a chrome:// page fails differently, and
            # sending someone to a menu that is already on reads as
            # the app being broken.
            if "turned off" in stderr or "1743" in stderr:
                hint = ("Safari's Develop menu > Allow JavaScript from "
                        "Apple Events" if browser == "Safari" else
                        f"{browser}: View > Developer > Allow "
                        f"JavaScript from Apple Events")
                raise UnsupportedCommand(
                    f"controlling a background tab needs one browser "
                    f"switch, once: {hint} -- then it works silently "
                    f"forever")

            raise UnsupportedCommand(
                f"the pinned tab could not be reached just now "
                f"({stderr.splitlines()[-1][:80] if stderr else 'no answer'}) "
                f"-- it may be asleep; click it once or point again")

        if answer in ("played", "paused", "sought"):
            self._last_media_title = title

        if answer == "played":
            return f'played "{short}" in the background'
        if answer == "paused":
            return f'paused "{short}" in the background'
        if answer == "novideo":
            raise UnsupportedCommand(
                f'no video found in the pinned tab "{short}"')

        return answer

    # ------------------------------------------------------------ quitting

    #: Never quit by "quit all": the desktop itself, and QRUDO -- an app
    #: that closes its own controller mid-gesture is a very short demo.
    _NEVER_QUIT = ("finder", "qrudo")

    def quit_app(self, name: str) -> str:
        """Ask an app -- or with "all", every regular app -- to quit.

        Asked, not forced: terminate() is the polite quit, the same as
        cmd-Q, so an app with unsaved work shows its own save dialog
        instead of losing anything.  Quitting an app that is not
        running succeeds by doing nothing -- that IS the asked-for
        state -- so a gesture never errors over an already-closed app.
        """

        import os

        if self._workspace is None:
            raise UnsupportedCommand(
                "quitting apps needs pyobjc (pip install -r "
                "requirements.txt)")

        me = os.getpid()
        running = [
            app for app in
            self._workspace.sharedWorkspace().runningApplications()
            if app.activationPolicy() == 0      # regular, visible apps
            and app.processIdentifier() != me
            and (app.localizedName() or "").lower() not in self._NEVER_QUIT
        ]

        if name.strip().lower() == "all":
            for app in running:
                app.terminate()
            if not running:
                return "nothing was open to quit"
            return f"asked {len(running)} apps to quit"

        bare = name.strip().lower()
        hits = [a for a in running
                if (a.localizedName() or "").lower() == bare]
        if not hits:
            hits = [a for a in running
                    if bare in (a.localizedName() or "").lower()]

        if not hits:
            return f"{name} was not running"

        for app in hits:
            app.terminate()

        return f"asked {hits[0].localizedName()} to quit"

    def hide_app(self, name: str) -> str:
        """Put an app's -- or every regular app's -- windows away.

        Hidden, not closed: the same as cmd-H, so everything keeps
        running and one click on the Dock brings it back.  QRUDO and
        the desktop are left visible -- hiding the controller mid-
        gesture would look like a crash.
        """

        import os

        if self._workspace is None:
            raise UnsupportedCommand(
                "hiding apps needs pyobjc (pip install -r "
                "requirements.txt)")

        me = os.getpid()
        running = [
            app for app in
            self._workspace.sharedWorkspace().runningApplications()
            if app.activationPolicy() == 0
            and app.processIdentifier() != me
            and (app.localizedName() or "").lower() not in self._NEVER_QUIT
        ]

        if name.strip().lower() == "all":
            visible = [a for a in running if not a.isHidden()]
            for app in visible:
                app.hide()
            if not visible:
                return "nothing was showing to hide"
            return f"hid {len(visible)} apps"

        bare = name.strip().lower()
        hits = [a for a in running
                if (a.localizedName() or "").lower() == bare]
        if not hits:
            hits = [a for a in running
                    if bare in (a.localizedName() or "").lower()]

        if not hits:
            return f"{name} was not running"

        for app in hits:
            app.hide()

        return f"hid {hits[0].localizedName()}"

    # ------------------------------------------------------- event plumbing

    _ax = None   # ApplicationServices, loaded once, kept for re-asking

    def _require_event_trust(self):
        """Refuse to post keys the system would silently drop.

        Synthetic key events need the app in System Settings > Privacy &
        Security > Accessibility.  Untrusted, CGEventPost DROPS the
        event and says nothing -- so seeking and media keys "succeeded"
        while nothing happened, and the log swore everything was OK.
        And the grant dies quietly on every rebuild: an ad-hoc re-sign
        changes the signature the grant is pinned to, so the switch in
        Settings still shows ON while the system no longer honours it.

        Asked fresh on every send, not cached, so granting access mid-
        session starts working immediately.  If the check itself is
        unavailable, the event is sent anyway -- a maybe is better than
        a certain refusal.
        """

        try:
            if MacOSController._ax is None:
                import ctypes
                MacOSController._ax = ctypes.CDLL(
                    "/System/Library/Frameworks/ApplicationServices."
                    "framework/ApplicationServices")
            trusted = bool(MacOSController._ax.AXIsProcessTrusted())
        except Exception:
            return

        if not trusted:
            raise UnsupportedCommand(
                "macOS is silently dropping QRUDO's key presses -- "
                "System Settings > Privacy & Security > Accessibility: "
                "add QRUDO, or if it is already listed, toggle it OFF "
                "and ON (an updated QRUDO must be re-trusted)")

    def _post_media_key(self, key: int) -> None:
        """Send an NSSystemDefined event -- the same thing a media key sends."""
        if self._quartz is None:
            raise UnsupportedCommand(
                "media keys need pyobjc-framework-Quartz (pip install -r requirements.txt)")
        self._require_event_trust()
        quartz, ns_event = self._quartz
        for pressed in (True, False):
            flags = 0xA if pressed else 0xB
            event = ns_event.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                14,                      # NSEventTypeSystemDefined
                (0, 0), flags << 8, 0, 0, None,
                8,                       # NX_SUBTYPE_AUX_CONTROL_BUTTONS
                (key << 16) | (flags << 8),
                -1,
            )
            quartz.CGEventPost(quartz.kCGHIDEventTap, event.CGEvent())

    def _post_key(self, key_code: int, *, to_pid: int | None = None) -> None:
        """Send a normal keystroke (used for arrow-key seeking).

        With ``to_pid`` the key goes straight to that process even while
        another app is focused, which is what lets seeking work without
        switching windows.
        """
        if self._quartz is not None:
            self._require_event_trust()
            quartz, _ = self._quartz
            for pressed in (True, False):
                event = quartz.CGEventCreateKeyboardEvent(None, key_code, pressed)
                if to_pid is None:
                    quartz.CGEventPost(quartz.kCGHIDEventTap, event)
                else:
                    quartz.CGEventPostToPid(to_pid, event)
            return
        self._osascript(f'tell application "System Events" to key code {key_code}')

    def send_combo(self, combo: str, target_app: str = "") -> str:
        """Press a user-taught keyboard shortcut, e.g. "cmd+shift+n".

        The base key is pressed with the modifier flags set on the event,
        so the system reads it as the whole chord.  This is how a taught
        gesture reaches an action QRUDO has no handler for -- next track,
        mute, a window shortcut -- with the key the app already uses.

        With ``target_app`` the chord is delivered to that app's process
        even while another window is focused -- which is what "global
        trigger, locked to YouTube Music" needs: the swipe fires from
        anywhere and the key still lands in YouTube Music.  Without it,
        the chord goes to whatever has focus.  A named app that is not
        running refuses, rather than firing the key into the wrong
        window.
        """

        from ..keystroke import parse

        parsed = parse(combo)          # raises ComboError -> UNSUPPORTED

        to_pid = None
        if target_app:
            to_pid = self._target_pid(target_app)
            if to_pid is None:
                raise UnsupportedCommand(
                    f"{target_app} is not running, so its shortcut has "
                    f"nowhere to go")

        if self._quartz is not None:
            self._require_event_trust()

            # Delivered to the FOCUSED window, a taught chord is typed
            # wherever the cursor sits -- which printed a literal N
            # into a document when a hole-gesture's shift+n landed in a
            # text box.  The same guard the play/pause letters use: a
            # text box refuses, a video page takes the chord.  A chord
            # aimed at a NAMED app is deliberate and goes through.
            if to_pid is None:
                self._refuse_to_type_into_a_text_box(None)

            quartz, _ = self._quartz
            for pressed in (True, False):
                event = quartz.CGEventCreateKeyboardEvent(
                    None, parsed.key_code, pressed)
                quartz.CGEventSetFlags(event, parsed.flags)
                if to_pid is None:
                    quartz.CGEventPost(quartz.kCGHIDEventTap, event)
                else:
                    quartz.CGEventPostToPid(to_pid, event)

            where = f" to {target_app}" if target_app else ""
            return f"sent {parsed.describe()}{where}"

        raise UnsupportedCommand(
            "custom keystrokes need Quartz (pyobjc) -- not available here")

    # ------------------------------------------------------------- subprocess

    def _osascript(self, script: str) -> str:
        return self._run(["osascript", "-e", script])

    def open_argv(self, argv: list[str]) -> str:
        """Run an argv list for a custom action -- open a file, launch an
        app, a confirmed command.  A list, never a shell string, so
        nothing in it is re-interpreted; the danger screening happened
        in control/actions before this is ever reached."""

        return self._run(argv)

    def _run(self, argv: list[str]) -> str:
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=_OSASCRIPT_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise ControlError(f"{argv[0]} timed out after {_OSASCRIPT_TIMEOUT}s") from exc
        except FileNotFoundError as exc:
            raise ControlError(f"{argv[0]} not found on PATH") from exc
        if proc.returncode != 0:
            raise ControlError(
                f"{argv[0]} failed ({proc.returncode}): {proc.stderr.strip() or 'no output'}")
        return proc.stdout.strip()

    # -------------------------------------------------------------- preflight

    def preflight(self) -> list[str]:
        warnings: list[str] = []
        if self._quartz is None:
            warnings.append(
                "pyobjc-framework-Quartz is missing: PLAY_PAUSE/REWIND/FORWARD fall back to "
                "AppleScript and only work with a known player running. "
                "Fix: pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa")
        elif not self._can_post_events():
            # Packaged, the grant belongs to QRUDO itself; run from a
            # terminal, it belongs to whatever launched us -- and naming
            # the wrong one sends the user to a checkbox that fixes
            # nothing.
            import sys as _sys
            grantee = ("QRUDO" if getattr(_sys, "frozen", False)
                       else "the app that launches QRUDO "
                            "(Terminal / iTerm / VS Code)")
            warnings.append(
                "Accessibility permission not granted: media keys and seeking will do nothing. "
                "Fix: System Settings > Privacy & Security > Accessibility, "
                f"enable {grantee}.")
        if self._display_services is None and not self._brightness_cli:
            warnings.append(
                "no precise brightness control available; falling back to brightness HID keys "
                "(external monitors may ignore them)")
        return warnings

    def read_state(self) -> dict[str, float]:
        state: dict[str, float] = {}
        device = self._core_audio.device() if self._core_audio else None
        if device is not None:
            level = self._core_audio.get_volume(device)
            if level is not None:
                state["volume"] = level
                state["muted"] = float(self._core_audio.is_muted(device))
        elif self._volume_settings_safe() is not None:
            settings = self._volume_settings_safe()
            state["volume"] = settings["volume"] / 100
            state["muted"] = float(settings["muted"])
        if self._display_services is not None:
            level = self._get_brightness()
            if level is not None:
                state["brightness"] = level
        return state

    def restore_state(self, state: dict[str, float]) -> None:
        device = self._core_audio.device() if self._core_audio else None
        if device is not None and "volume" in state:
            self._core_audio.set_volume(device, state["volume"])
        elif "volume" in state:
            self._osascript(f"set volume output volume {round(state['volume'] * 100)}")
        if "brightness" in state and self._display_services is not None:
            self._set_brightness(state["brightness"])

    def _volume_settings_safe(self) -> dict | None:
        try:
            return self._volume_settings()
        except ControlError:
            return None

    def _can_post_events(self) -> bool:
        """Best-effort Accessibility check; assume OK if macOS won't tell us."""
        quartz, _ = self._quartz
        check = getattr(quartz, "CGPreflightPostEventAccess", None)
        try:
            return bool(check()) if check else True
        except Exception:  # API missing on this OS version
            return True


#: Focused-element roles that swallow letters.  A keystroke posted to a
#: browser lands wherever its keyboard focus is, and if that is a text
#: box the "shortcut" is typing.
_TEXT_ROLES = {"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"}


def _focus_report(pid: int) -> tuple:
    """Where a letter posted to this app would land.

    Returns (kind, front window title).  ``kind`` is "editable" when the
    focus is known to take typing, "element" when it is known not to,
    "none" when the app reports no focus at all -- which is what a
    background browser says, while still delivering the letter to its
    front tab -- and "unknown" when nothing could be asked.

    Editable is judged by role.  It was briefly also judged by whether
    the element's value could be written, to catch editable regions that
    are not classic text fields -- and that probe read a freshly loaded
    page as a text box, because a page container's value is settable too:
    it is the scroll position, not text.  Focus sits on the container
    until the video is clicked, so a fist could pause a video it could
    never start.

    The chat boxes that motivated the probe are already refused by the
    front-tab title: they are not the video, and the letter can go
    nowhere else.  On the video's own page, the typing targets are real
    text fields -- a search box, a comment box -- which the roles name.

    The title answers even for a background app, and it names the front
    tab -- which is the one place the letter can go.
    """

    try:
        ax = _ax_handles()

        if ax is None:
            return "unknown", None

        cf, services = ax

        def cfstr(text):
            return cf.CFStringCreateWithCString(None, text, 0x08000100)

        def read(element, attribute):
            out = ctypes.c_void_p()
            err = services.AXUIElementCopyAttributeValue(
                element, cfstr(attribute), ctypes.byref(out))
            return err, out.value

        def as_text(ref):
            buffer = ctypes.create_string_buffer(256)
            if ref and cf.CFStringGetCString(ref, buffer, 256, 0x08000100):
                return buffer.value.decode()
            return None

        app = services.AXUIElementCreateApplication(pid)

        if not app:
            return "unknown", None

        title = None
        err, window = read(app, b"AXFocusedWindow")

        if err == 0 and window:
            _, ref = read(window, b"AXTitle")
            title = as_text(ref)

        err, focused = read(app, b"AXFocusedUIElement")

        if err != 0 or not focused:
            # kAXErrorNoValue is a successful "nothing has focus", which
            # is what a background app says.  Anything else is a failure
            # to ask.
            return ("none" if err == -25212 else "unknown"), title

        _, role_ref = read(focused, b"AXRole")
        role = as_text(role_ref)

        if role in _TEXT_ROLES:
            return "editable", title

        return "element", title
    except Exception:
        return "unknown", None


_AX = None


def _ax_handles():
    """The CoreFoundation and Accessibility libraries, loaded once."""

    global _AX

    if _AX is None:
        try:
            cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation"
                             ".framework/CoreFoundation")
            services = ctypes.CDLL("/System/Library/Frameworks/"
                                   "ApplicationServices.framework/"
                                   "ApplicationServices")

            cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            cf.CFStringCreateWithCString.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
            cf.CFStringGetCString.restype = ctypes.c_bool
            cf.CFStringGetCString.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long,
                ctypes.c_uint32]
            services.AXUIElementCreateApplication.restype = ctypes.c_void_p
            services.AXUIElementCreateApplication.argtypes = [ctypes.c_int]
            services.AXUIElementCopyAttributeValue.restype = ctypes.c_int
            services.AXUIElementCopyAttributeValue.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p)]
            services.AXUIElementIsAttributeSettable.restype = ctypes.c_int
            services.AXUIElementIsAttributeSettable.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_bool)]

            _AX = (cf, services)
        except Exception:
            _AX = False

    return _AX or None


def _normalise_title(title: str) -> str:
    """A tab title reduced to what stays put: lowercase, without the
    "(3) " notification prefix browsers bolt on and shake off."""

    import re as _re

    return _re.sub(r"^\(\d+\)\s*", "", title.strip().lower())


def _match_tab_title(stored: str, titles: list) -> int | None:
    """The index of the live tab the stored title means, or None.

    Titles drift -- notification counts appear, players append and
    drop suffixes -- so equality is the wrong question.  Asked
    instead, in order: same normalised title; one contains the other;
    a generous shared prefix.  Autoplay to a NEW video is a genuinely
    different title and stays unmatched on purpose: guessing which
    video a person meant is worse than asking them to point again.
    """

    want = _normalise_title(stored)

    if not want:
        return None

    have = [_normalise_title(title) for title in titles]

    for i, title in enumerate(have):
        if title == want:
            return i

    for i, title in enumerate(have):
        if want in title or (title and title in want):
            return i

    for i, title in enumerate(have):
        shared = 0
        for a, b in zip(title, want):
            if a != b:
                break
            shared += 1
        if shared >= 12:
            return i

    return None


class _CoreAudio:
    """Volume straight from CoreAudio instead of shelling out to osascript.

    osascript costs ~200 ms per command, nearly all of it process startup, which
    is long enough to stutter a 30 fps camera loop.  These calls take ~0.1 ms.
    """

    _SYSTEM_OBJECT = 1

    class _Address(ctypes.Structure):
        _fields_ = [("selector", ctypes.c_uint32),
                    ("scope", ctypes.c_uint32),
                    ("element", ctypes.c_uint32)]

    def __init__(self) -> None:
        self._lib = ctypes.CDLL(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        address = ctypes.POINTER(self._Address)
        self._lib.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32, address, ctypes.c_uint32, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        self._lib.AudioObjectSetPropertyData.argtypes = [
            ctypes.c_uint32, address, ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_void_p]
        self._lib.AudioObjectHasProperty.argtypes = [ctypes.c_uint32, address]
        self._default_output = _fourcc("dOut")
        self._scope_global = _fourcc("glob")
        self._scope_output = _fourcc("outp")
        self._volume = _fourcc("volm")
        self._mute = _fourcc("mute")
        self._running = _fourcc("gone")   # is anything playing right now

    def device(self) -> int | None:
        """The current default output device.

        Looked up per call rather than cached: plugging in headphones changes
        it, and the lookup is far too cheap to be worth caching.
        """
        device = ctypes.c_uint32()
        size = ctypes.c_uint32(4)
        address = self._Address(self._default_output, self._scope_global, 0)
        rc = self._lib.AudioObjectGetPropertyData(
            self._SYSTEM_OBJECT, ctypes.byref(address), 0, None,
            ctypes.byref(size), ctypes.byref(device))
        return device.value if rc == 0 else None

    def is_playing(self, device: int) -> bool | None:
        """Whether any process is sending audio to this device.

        None if the question could not be asked.  This is what makes the
        play/pause media key safe to send: it is a message to the system
        rather than to a player, and the system answers one with nothing
        playing by opening Music.
        """

        value = ctypes.c_uint32()
        size = ctypes.c_uint32(4)
        address = self._Address(self._running, self._scope_global, 0)
        rc = self._lib.AudioObjectGetPropertyData(
            device, ctypes.byref(address), 0, None,
            ctypes.byref(size), ctypes.byref(value))

        return bool(value.value) if rc == 0 else None

    def get_volume(self, device: int) -> float | None:
        address = self._Address(self._volume, self._scope_output, 0)
        if not self._lib.AudioObjectHasProperty(device, ctypes.byref(address)):
            return None  # e.g. some USB/Bluetooth devices have no master channel
        level = ctypes.c_float()
        size = ctypes.c_uint32(4)
        rc = self._lib.AudioObjectGetPropertyData(
            device, ctypes.byref(address), 0, None, ctypes.byref(size),
            ctypes.byref(level))
        return level.value if rc == 0 else None

    def set_volume(self, device: int, level: float) -> bool:
        address = self._Address(self._volume, self._scope_output, 0)
        value = ctypes.c_float(level)
        return self._lib.AudioObjectSetPropertyData(
            device, ctypes.byref(address), 0, None, 4, ctypes.byref(value)) == 0

    def is_muted(self, device: int) -> bool:
        address = self._Address(self._mute, self._scope_output, 0)
        if not self._lib.AudioObjectHasProperty(device, ctypes.byref(address)):
            return False
        muted = ctypes.c_uint32()
        size = ctypes.c_uint32(4)
        rc = self._lib.AudioObjectGetPropertyData(
            device, ctypes.byref(address), 0, None, ctypes.byref(size),
            ctypes.byref(muted))
        return rc == 0 and bool(muted.value)

    def unmute(self, device: int) -> bool:
        address = self._Address(self._mute, self._scope_output, 0)
        value = ctypes.c_uint32(0)
        return self._lib.AudioObjectSetPropertyData(
            device, ctypes.byref(address), 0, None, 4, ctypes.byref(value)) == 0


def _fourcc(code: str) -> int:
    """CoreAudio selectors are four-character codes packed into a uint32."""
    return int.from_bytes(code.encode("ascii"), "big")


def _load_core_audio():
    try:
        return _CoreAudio()
    except OSError:
        return None


def _load_quartz():
    try:
        import Quartz
        from AppKit import NSEvent
    except ImportError:
        return None
    return Quartz, NSEvent


def _load_workspace():
    """NSWorkspace, used to look up a target app's process id by name."""
    try:
        from AppKit import NSWorkspace
    except ImportError:
        return None
    return NSWorkspace


def _load_display_services():
    """Load the private framework that Apple's own brightness slider uses."""
    try:
        lib = ctypes.CDLL(
            "/System/Library/PrivateFrameworks/DisplayServices.framework/DisplayServices")
        lib.DisplayServicesGetBrightness.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.DisplayServicesGetBrightness.restype = ctypes.c_int
        lib.DisplayServicesSetBrightness.argtypes = [ctypes.c_uint32, ctypes.c_float]
        lib.DisplayServicesSetBrightness.restype = ctypes.c_int
        return lib
    except (OSError, AttributeError):
        return None
