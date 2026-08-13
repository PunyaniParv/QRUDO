"""Windows implementation of the control layer.

Deliberately dependency-free: everything here uses ``ctypes`` against the
built-in ``user32.dll`` plus PowerShell, both of which ship with Windows.  That
matters because the macOS half needs pyobjc, which cannot be installed here at
all -- the setup on this side should be "git pull and go".

* **Volume and media** -- the virtual key codes a keyboard's own media keys
  send.  System-wide and application-independent, no permission prompt (the
  Accessibility grant macOS demands has no equivalent here).
* **Brightness** -- WMI via PowerShell, which reports and sets a real
  percentage.  Laptop panels only; see ``brightness_up`` for why.

Windows volume keys move in fixed 2% notches, so unlike macOS we cannot set an
exact percentage without extra packages.  A 5% step becomes the nearest whole
number of presses.
"""

from __future__ import annotations

import ctypes
import subprocess

from ..commands import NO_CHANGE
from ..config import ControlConfig
from ..controller import Controller, ControlError, UnsupportedCommand
from ..log import get_logger

# --- Virtual key codes (winuser.h) ------------------------------------------
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_LEFT = 0x25
VK_RIGHT = 0x27

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

#: Windows moves the volume one fiftieth of full scale per key press.
VOLUME_PERCENT_PER_PRESS = 2

_POWERSHELL_TIMEOUT = 10.0

# PowerShell reads, clamps and writes the brightness in one call, and reports
# both the old and new value so the log can show "40% -> 48%" the way macOS
# does.  Errors come back as text rather than an exit code, because PowerShell
# is inconsistent about exit codes inside -Command.
_BRIGHTNESS_SCRIPT = """
try {{
    $b = @(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness -ErrorAction Stop)[0]
    $cur = [int]$b.CurrentBrightness
    $target = $cur + ({delta})
    if ($target -gt 100) {{ $target = 100 }}
    if ($target -lt 0) {{ $target = 0 }}
    $m = @(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods -ErrorAction Stop)[0]
    Invoke-CimMethod -InputObject $m -MethodName WmiSetBrightness -Arguments @{{Brightness=[byte]$target; Timeout=[uint32]1}} | Out-Null
    Write-Output "$cur|$target|ok"
}} catch {{
    Write-Output "error|$($_.Exception.Message)"
}}
"""


# Read-only version of the above, for the self-test's before/after lines and
# for the --check probe.  Worth keeping separate: the setting script writes the
# brightness even when the delta is zero.
_READ_BRIGHTNESS_SCRIPT = """
try {
    $b = @(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness -ErrorAction Stop)[0]
    Write-Output "$([int]$b.CurrentBrightness)|ok"
} catch {
    Write-Output "error|$($_.Exception.Message)"
}
"""


class WindowsController(Controller):
    name = "Windows"

    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or ControlConfig()
        self.log = get_logger("windows")
        # WinDLL exists only on Windows; keep the import lazy so this file can
        # still be opened and read on a Mac.
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:
            raise ControlError("WindowsController requires Windows")
        self._user32 = win_dll("user32", use_last_error=True)

    # ------------------------------------------------------------------ volume

    def volume_up(self, step: int) -> str:
        return self._nudge_volume(step, VK_VOLUME_UP)

    def volume_down(self, step: int) -> str:
        return self._nudge_volume(step, VK_VOLUME_DOWN)

    def _nudge_volume(self, step: int, key: int) -> str:
        """Press the volume key however many times the step needs.

        Windows unmutes by itself on volume-up, so ``unmute_on_volume_up`` needs
        no special handling here.
        """
        presses = max(1, round(step / VOLUME_PERCENT_PER_PRESS))
        for _ in range(presses):
            self._press(key)
        moved = presses * VOLUME_PERCENT_PER_PRESS
        direction = "up" if key == VK_VOLUME_UP else "down"
        return f"volume {direction} ~{moved}% ({presses}x {VOLUME_PERCENT_PER_PRESS}% key press)"

    # -------------------------------------------------------------- brightness

    def brightness_up(self, step: int) -> str:
        return self._set_brightness(step)

    def brightness_down(self, step: int) -> str:
        return self._set_brightness(-step)

    def _set_brightness(self, delta: int) -> str:
        raw = self._powershell(_BRIGHTNESS_SCRIPT.format(delta=delta))
        if raw.startswith("error|"):
            # Desktops and most external monitors do not expose WMI brightness;
            # unlike macOS there is no key-press fallback to try.
            raise UnsupportedCommand(
                "this display does not support software brightness control "
                f"(WMI said: {raw.split('|', 1)[1]}). Built-in laptop screens usually do; "
                "external monitors usually do not.")
        try:
            old, new, _ = raw.split("|")
        except ValueError as exc:
            raise ControlError(f"unexpected brightness output {raw!r}") from exc
        if old == new:
            return f"brightness {NO_CHANGE} {'maximum' if delta > 0 else 'minimum'} ({old}%)"
        return f"brightness {old}% -> {new}%"

    # ------------------------------------------------------------------- media

    def play_pause(self) -> str:
        self._press(VK_MEDIA_PLAY_PAUSE)
        return "sent play/pause media key"

    def rewind(self, seconds: int) -> str:
        return self._seek(seconds, forward=False)

    def forward(self, seconds: int) -> str:
        return self._seek(seconds, forward=True)

    def _seek(self, seconds: int, *, forward: bool) -> str:
        if self.config.seek_mode == "track":
            self._press(VK_MEDIA_NEXT_TRACK if forward else VK_MEDIA_PREV_TRACK)
            return f"{'next' if forward else 'previous'} track"

        # Same approach as macOS: repeat the player's own arrow-key shortcut.
        presses = self.config.seek_presses
        key = VK_RIGHT if forward else VK_LEFT
        for _ in range(presses):
            self._press(key, extended=True)  # arrows are extended keys
        covered = presses * self.config.seek_step_seconds
        direction = "forward" if forward else "back"
        return f"seek {direction} ~{covered}s ({presses}x arrow, {seconds}s requested)"

    # ------------------------------------------------------- event plumbing

    def _press(self, key: int, *, extended: bool = False) -> None:
        """Tap a key: press then release, as a real keyboard does."""
        flags = KEYEVENTF_EXTENDEDKEY if extended else 0
        self._user32.keybd_event(key, 0, flags, 0)
        self._user32.keybd_event(key, 0, flags | KEYEVENTF_KEYUP, 0)

    def _powershell(self, script: str) -> str:
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=_POWERSHELL_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise ControlError(f"powershell timed out after {_POWERSHELL_TIMEOUT}s") from exc
        except FileNotFoundError as exc:
            raise ControlError("powershell not found on PATH") from exc
        if proc.returncode != 0:
            raise ControlError(
                f"powershell failed ({proc.returncode}): {proc.stderr.strip() or 'no output'}")
        return proc.stdout.strip()

    # -------------------------------------------------------------- preflight

    def _read_brightness(self) -> int | None:
        """Current brightness percentage, or None if this display cannot say."""
        raw = self._powershell(_READ_BRIGHTNESS_SCRIPT)
        if raw.startswith("error|"):
            return None
        try:
            return int(raw.split("|")[0])
        except ValueError:
            return None

    def snapshot(self) -> str:
        # Volume cannot be read back on Windows without an extra package, so
        # brightness is the only state the self-test can verify here.
        try:
            level = self._read_brightness()
        except ControlError:
            return ""
        return f"brightness {level}%" if level is not None else ""

    def preflight(self) -> list[str]:
        warnings = [
            f"volume moves in {VOLUME_PERCENT_PER_PRESS}% steps on Windows, so a "
            f"{self.config.volume_step}% setting becomes "
            f"{max(1, round(self.config.volume_step / VOLUME_PERCENT_PER_PRESS)) * VOLUME_PERCENT_PER_PRESS}%",
            "brightness goes through PowerShell and takes about 1.5s; use "
            "engine.submit() instead of execute() to keep the camera loop smooth",
        ]
        try:
            if self._read_brightness() is None:
                warnings.append(
                    "brightness control unavailable on this display "
                    "(normal for desktops and external monitors); "
                    "the other six commands still work")
        except ControlError as exc:
            warnings.append(f"could not probe brightness support: {exc}")
        return warnings
