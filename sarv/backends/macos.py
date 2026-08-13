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
from ..controller import Controller, ControlError, UnsupportedCommand
from ..log import get_logger

# --- HID "special key" codes, from IOKit's ev_keymap.h -----------------------
NX_KEYTYPE_BRIGHTNESS_UP = 2
NX_KEYTYPE_BRIGHTNESS_DOWN = 3
NX_KEYTYPE_PLAY = 16
NX_KEYTYPE_NEXT = 17
NX_KEYTYPE_PREVIOUS = 18

# --- Virtual key codes for ordinary keys ------------------------------------
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
        self._brightness_cli = shutil.which("brightness")

    # ------------------------------------------------------------------ volume

    def volume_up(self, step: int) -> str:
        return self._change_volume(step)

    def volume_down(self, step: int) -> str:
        return self._change_volume(-step)

    def _change_volume(self, delta: int) -> str:
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
        if self._quartz is not None:
            self._post_media_key(NX_KEYTYPE_PLAY)
            return "sent play/pause media key"
        return self._applescript_play_pause()

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
        presses = self.config.seek_presses
        key = KEY_RIGHT_ARROW if forward else KEY_LEFT_ARROW
        for _ in range(presses):
            self._post_key(key)
        covered = presses * self.config.seek_step_seconds
        return f"seek {direction} ~{covered}s ({presses}x arrow, {seconds}s requested)"

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

    def _post_key(self, key_code: int) -> None:
        """Send a normal keystroke (used for arrow-key seeking)."""
        if self._quartz is not None:
            quartz, _ = self._quartz
            for pressed in (True, False):
                event = quartz.CGEventCreateKeyboardEvent(None, key_code, pressed)
                quartz.CGEventPost(quartz.kCGHIDEventTap, event)
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

    def _can_post_events(self) -> bool:
        """Best-effort Accessibility check; assume OK if macOS won't tell us."""
        quartz, _ = self._quartz
        check = getattr(quartz, "CGPreflightPostEventAccess", None)
        try:
            return bool(check()) if check else True
        except Exception:  # API missing on this OS version
            return True


def _load_quartz():
    try:
        import Quartz
        from AppKit import NSEvent
    except ImportError:
        return None
    return Quartz, NSEvent


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
