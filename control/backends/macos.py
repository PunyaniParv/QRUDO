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
  app running SARV (Terminal / VS Code / Python.app).
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
            "target_app in sarv_config.json")

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
        """Send whichever key plays and pauses inside this browser.

        Which key it is depends on the site rather than the browser: k is
        YouTube's, most other players use the spacebar, and the keyboard's
        own media key works wherever something is genuinely playing.
        """

        wanted = self.config.browser_play_key.strip().lower()

        if wanted == "media":
            # Music being open used to divert this to the browser's own
            # shortcut, on the grounds that the system would hand the
            # media key to Music.  That is only true of a media key sent
            # with nothing playing, which is now refused outright -- and
            # the diversion is what typed a letter into whatever had the
            # keyboard focus.  When something is playing, the key goes to
            # the thing that is playing, which is the whole point of it.
            return self._media_play_pause(name, pid)

        key = KEY_SPACE if wanted in ("space", "spacebar") else KEY_K

        self._refuse_to_type_into_a_text_box(pid)
        self._post_key(key, to_pid=pid)

        return f"play/pause ({wanted}) to {name}"

    def rewind(self, seconds: int) -> str:
        return self._seek(seconds, forward=False)

    def forward(self, seconds: int) -> str:
        return self._seek(seconds, forward=True)

    def _seek(self, seconds: int, *, forward: bool) -> str:
        direction = "forward" if forward else "back"

        if self.config.seek_mode == "track":
            self._post_media_key(NX_KEYTYPE_NEXT if forward else NX_KEYTYPE_PREVIOUS)
            return f"{'next' if forward else 'previous'} track"

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

    #: Set when SARV was the one that paused, so it knows it may resume.
    _paused_it = False

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

    def _refuse_to_type_into_a_text_box(self, pid: int) -> None:
        """A shortcut is a letter, and a letter lands wherever the app's
        keyboard focus is.  When that is known to be a text box, refusing
        beats typing: the k in a search box was reported twice before the
        focus was ever asked about.
        """

        if pid and _focus_is_a_text_box(pid):
            raise UnsupportedCommand(
                "the cursor is in a text box, so play/pause would type "
                "into it -- click the video first")

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

    # ------------------------------------------------------- event plumbing

    def _post_media_key(self, key: int) -> None:
        """Send an NSSystemDefined event -- the same thing a media key sends."""
        if self._quartz is None:
            raise UnsupportedCommand(
                "media keys need pyobjc-framework-Quartz (pip install -r requirements.txt)")
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
            quartz, _ = self._quartz
            for pressed in (True, False):
                event = quartz.CGEventCreateKeyboardEvent(None, key_code, pressed)
                if to_pid is None:
                    quartz.CGEventPost(quartz.kCGHIDEventTap, event)
                else:
                    quartz.CGEventPostToPid(to_pid, event)
            return
        self._osascript(f'tell application "System Events" to key code {key_code}')

    # ------------------------------------------------------------- subprocess

    def _osascript(self, script: str) -> str:
        return self._run(["osascript", "-e", script])

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
            warnings.append(
                "Accessibility permission not granted: media keys and seeking will do nothing. "
                "Fix: System Settings > Privacy & Security > Accessibility, enable the app "
                "that launches SARV (Terminal / iTerm / VS Code).")
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


def _focus_is_a_text_box(pid: int) -> bool:
    """Whether this app's keyboard focus sits in a text field right now.

    Asked through Accessibility, which SARV already holds the permission
    for -- posting keys at all requires it.  Errors, including the app
    having no focused element to report, answer False: the guard exists
    to stop a letter landing in a text box that is known to be there, and
    "cannot tell" is not that.
    """

    try:
        ax = _ax_handles()

        if ax is None:
            return False

        cf, services = ax
        element = services.AXUIElementCreateApplication(pid)

        if not element:
            return False

        out = ctypes.c_void_p()
        name = cf.CFStringCreateWithCString(
            None, b"AXFocusedUIElement", 0x08000100)

        if services.AXUIElementCopyAttributeValue(
                element, name, ctypes.byref(out)) != 0 or not out.value:
            return False

        role = ctypes.c_void_p()
        name = cf.CFStringCreateWithCString(None, b"AXRole", 0x08000100)

        if services.AXUIElementCopyAttributeValue(
                out.value, name, ctypes.byref(role)) != 0 or not role.value:
            return False

        buffer = ctypes.create_string_buffer(128)

        if not cf.CFStringGetCString(role.value, buffer, 128, 0x08000100):
            return False

        return buffer.value.decode() in _TEXT_ROLES
    except Exception:
        return False


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

            _AX = (cf, services)
        except Exception:
            _AX = False

    return _AX or None


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
