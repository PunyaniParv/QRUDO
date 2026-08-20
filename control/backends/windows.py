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
* **Custom actions** -- the CUSTOM command path (``open_argv``, ``send_combo``)
  the taught-gesture and voice-catalog actions ride on.  Apps launch through
  ShellExecute after a dependency-free registry/PATH resolution; keystrokes
  press through the same user32 calls as the media keys.

Windows volume keys move in fixed 2% notches, so unlike macOS we cannot set an
exact percentage without extra packages.  A 5% step becomes the nearest whole
number of presses.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

try:
    import winreg  # Windows only; the module must still import on macOS
except ImportError:  # pragma: no cover - exercised only off-Windows
    winreg = None

from ..commands import NO_CHANGE
from ..config import ControlConfig
from ..executor import Controller, ControlError, UnsupportedCommand
from ..log import get_logger

# --- Virtual key codes (winuser.h) ------------------------------------------
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_J = 0x4A      # YouTube's back ten seconds
VK_K = 0x4B      # YouTube's play/pause
VK_L = 0x4C      # YouTube's forward ten seconds
VK_LEFT = 0x25
VK_RIGHT = 0x27

# --- Virtual key codes for taught keystrokes (winuser.h) ---------------------
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12            # Alt
VK_LWIN = 0x5B            # the "cmd"/Win key a Mac-style combo means here
VK_BACK = 0x08
VK_RETURN = 0x0D
VK_TAB = 0x09
VK_ESCAPE = 0x1B
VK_UP = 0x26
VK_DOWN = 0x28

#: The named base keys the keystroke parser may yield, to their VK code.
#: Letters and digits map by their ASCII value (A-Z 0x41.., 0-9 0x30..).
_NAMED_VK = {
    "space": 0x20, "return": VK_RETURN, "enter": VK_RETURN,
    "tab": VK_TAB, "escape": VK_ESCAPE, "esc": VK_ESCAPE,
    "left": VK_LEFT, "right": VK_RIGHT, "down": VK_DOWN, "up": VK_UP,
    "delete": VK_BACK, "backspace": VK_BACK,
}

#: Canonical modifier names the keystroke parser folds to, and their VK code.
#: "cmd" is the Windows key here, exactly as the parser's aliases intend.
_MOD_VK = {"cmd": VK_LWIN, "ctrl": VK_CONTROL, "shift": VK_SHIFT,
           "alt": VK_MENU}


def _keyname_vk(name: str) -> int | None:
    """A base-key name from the keystroke parser to a VK code, or None.

    Letters and digits share their ASCII VK codes; everything else lives
    in ``_NAMED_VK``.  None means the parser accepted a key this backend
    does not know how to press, which must refuse rather than guess.
    """
    if len(name) == 1 and name.isalnum():
        return ord(name.upper())
    return _NAMED_VK.get(name)

#: Common app display names to their exe, so ``open_app`` needs no registry
#: walk for the apps people say most often.  Everything else resolves through
#: PATH, the App Paths key, or the Uninstall DisplayName search -- this table
#: is a convenience, never the whole story.
_KNOWN_APPS = {
    "google chrome": "chrome",
    "chrome": "chrome",
    "microsoft edge": "msedge",
    "edge": "msedge",
    "mozilla firefox": "firefox",
    "firefox": "firefox",
    "opera": "opera",
    "notepad": "notepad",
    "notepad++": "notepad++",
    "calculator": "calc",
    "paint": "mspaint",
    "windows terminal": "wt",
    "terminal": "wt",
    "spotify": "spotify",
    "vlc": "vlc",
    "slack": "slack",
    "discord": "discord",
    "zoom": "zoom",
    "steam": "steam",
    "itunes": "itunes",
    "microsoft word": "winword",
    "word": "winword",
    "microsoft excel": "excel",
    "excel": "excel",
    "microsoft powerpoint": "powerpnt",
    "powerpoint": "powerpnt",
    "outlook": "outlook",
}

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101

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


#: Signature of the callback EnumWindows/EnumChildWindows hand each window to.
_ENUM_PROC = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)


def _declare(user32) -> None:
    """Give ctypes the real signatures for the user32 calls we make.

    Undeclared functions default to a 32-bit int return and 32-bit int
    arguments.  On 64-bit Python that silently truncates window handles, and
    PostMessageW's lParam -- which carries bit 31 for a key-up -- does not fit
    in a signed 32-bit int at all, so the call would raise instead of running.
    """
    from ctypes import wintypes

    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.EnumWindows.argtypes = [_ENUM_PROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.EnumChildWindows.argtypes = [wintypes.HWND, _ENUM_PROC, wintypes.LPARAM]
    user32.EnumChildWindows.restype = wintypes.BOOL
    user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
    user32.MapVirtualKeyW.restype = wintypes.UINT
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                    wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE,
                                   wintypes.DWORD, ctypes.c_void_p]
    user32.keybd_event.restype = None


class _PowerShellWorker:
    """One long-lived PowerShell that takes brightness commands on stdin.

    Launching ``powershell`` costs about a second, and the brightness commands
    were paying it every single time -- measured at 1.4 s per command on a real
    laptop, which stalls a 30 fps preview for forty frames.  Keeping one process
    alive pays that once.

    Every failure path here falls back to the one-shot method rather than
    raising: a hung helper during a demo would be far worse than a slow one.
    """

    #: Seconds a cached brightness reading stays trustworthy.  Long enough to
    #: cover a burst of gestures, short enough that using the laptop's own
    #: brightness keys in between is picked up on the next command.
    _CACHE_SECONDS = 1.5

    _LOOP = f"""
$out = [Console]::Out
$methods = $null
$last = $null
$lastSet = [DateTime]::MinValue
while ($true) {{
    $line = [Console]::In.ReadLine()
    if ($null -eq $line -or $line -eq 'quit') {{ break }}
    try {{
        if ($null -eq $methods) {{
            $methods = @(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods -ErrorAction Stop)[0]
        }}
        $stale = ((Get-Date) - $lastSet).TotalSeconds -gt {_CACHE_SECONDS}
        if ($line -eq 'read' -or $null -eq $last -or $stale) {{
            $cur = [int](@(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness -ErrorAction Stop)[0]).CurrentBrightness
        }} else {{
            $cur = $last
        }}
        if ($line -eq 'read') {{
            $out.WriteLine("$cur|ok")
        }} else {{
            $target = $cur + [int]$line
            if ($target -gt 100) {{ $target = 100 }}
            if ($target -lt 0) {{ $target = 0 }}
            Invoke-CimMethod -InputObject $methods -MethodName WmiSetBrightness -Arguments @{{Brightness=[byte]$target; Timeout=[uint32]1}} | Out-Null
            $out.WriteLine("$cur|$target|ok")
            $last = $target
            $lastSet = Get-Date
        }}
    }} catch {{
        # Forget everything cached; the display may have changed underneath us.
        $methods = $null
        $last = $null
        $out.WriteLine("error|$($_.Exception.Message)")
    }}
    $out.Flush()
}}
"""

    def __init__(self, log) -> None:
        self.log = log
        self._process: subprocess.Popen | None = None
        self._broken = False
        self._reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qrudo-ps")
        atexit.register(self.close)

    def prewarm(self) -> None:
        """Launch the helper now so the first command does not wait for it.

        Popen returns immediately, and PowerShell does its own start-up while
        QRUDO carries on booting -- so by the time a gesture arrives, the second
        or so of launch cost has already been paid.
        """
        self._start()

    def exchange(self, line: str, timeout: float = 6.0) -> str | None:
        """Send one line, return one reply.  None means "use the slow path"."""
        if self._broken:
            return None
        if not self._start():
            return None
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
            # A blocking readline cannot be given a timeout directly, so it runs
            # on a helper thread.  Killing the process on timeout unblocks it.
            for _ in range(3):
                reply = self._reader.submit(self._process.stdout.readline).result(timeout)
                if not reply:
                    break
                if "|" in reply:
                    return reply.strip()
                # Ignore anything else PowerShell decides to print.
        except Exception as exc:
            self.log.warning("persistent powershell failed (%s); using slow path", exc)
        self._give_up()
        return None

    def _start(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        try:
            self._process = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", self._LOOP],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
            return True
        except OSError as exc:
            self.log.warning("could not start persistent powershell: %s", exc)
            self._broken = True
            return False

    def _give_up(self) -> None:
        self._broken = True
        self.close()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        try:
            process.kill()
        except OSError:
            pass


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
        _declare(self._user32)
        self._worker = (_PowerShellWorker(self.log)
                        if self.config.windows_persistent_powershell else None)
        if self._worker is not None:
            self._worker.prewarm()

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
        raw = self._brightness_call(str(delta), _BRIGHTNESS_SCRIPT.format(delta=delta))
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
        """Play or pause, in the named app if there is one.

        The media key works in any app but is a message to the system
        rather than to a player, so which app answers it is out of our
        hands when several are open.  Naming one settles it.
        """

        if self.config.app:
            window = self._target_window()

            if window:
                # A browser has no scriptable play/pause, so use the video
                # player's own shortcut.  k is YouTube's, and is safer than
                # space, which scrolls when the player is not selected.
                self._post_key_to_window(window, VK_K)
                return f"play/pause key to {self.config.app}"

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

        # Same approach as macOS: repeat the player's own seek shortcut,
        # whichever one the site in question listens for.
        key, name, step = self._seek_key(forward)
        presses = max(1, round(seconds / step))
        window = self._target_window()
        for _ in range(presses):
            if window:
                self._post_key_to_window(window, key)
            else:
                # arrows are extended keys; letters are not
                self._press(key, extended=name == "arrow")
        direction = "forward" if forward else "back"
        where = f" to {self.config.app}" if window else ""
        return (f"seek {direction} ~{presses * step}s{where} "
                f"({presses}x {name}, {seconds}s requested)")

    def _seek_key(self, forward: bool) -> tuple[int, str, int]:
        """Which key seeks, what to call it, and how far one press moves.

        Sites do not agree.  Arrow keys move five seconds on YouTube; j and
        l move ten.  A key that does not seek on a given site is rarely
        idle there -- it usually does something else -- which is why this
        is a setting rather than a guess.
        """

        if self.config.browser_seek_keys.strip().lower() == "jl":
            return (VK_L if forward else VK_J), "j/l", 10

        return ((VK_RIGHT if forward else VK_LEFT), "arrow",
                max(1, self.config.seek_step_seconds))

    # -------------------------------------------------- targeting a window

    def _target_window(self, title: str | None = None) -> int | None:
        """Handle of a window whose title matches ``target_app``.

        Returns None when nothing is configured or nothing matches, in which
        case seeking falls back to the focused window.
        """
        wanted = (title or self.config.app).strip().lower()
        if not wanted:
            return None

        user32 = self._user32
        matches: list[int] = []

        @_ENUM_PROC
        def visit(hwnd, _param):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if not length:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if wanted in buffer.value.lower():
                matches.append(hwnd)
                return False  # first match is enough
            return True

        user32.EnumWindows(visit, 0)
        if not matches:
            self.log.warning("no window matching %r; sending seek to the focused "
                             "window instead", self.config.app)
            return None
        return self._render_surface(matches[0])

    def _render_surface(self, hwnd: int) -> int:
        """Chromium ignores keys posted to its outer frame, so aim at the child
        window that actually hosts the page.  Other apps take the frame."""
        found: list[int] = []

        @_ENUM_PROC
        def visit(child, _param):
            name = ctypes.create_unicode_buffer(256)
            self._user32.GetClassNameW(child, name, 256)
            if name.value == "Chrome_RenderWidgetHostHWND":
                found.append(child)
                return False
            return True

        self._user32.EnumChildWindows(hwnd, visit, 0)
        return found[0] if found else hwnd

    def _post_key(self, hwnd: int, key: int, *, up: bool) -> None:
        """One half of a keystroke delivered to one window."""
        scan = self._user32.MapVirtualKeyW(key, 0)
        # lParam layout: repeat count, scan code, and the extended-key bit that
        # arrow keys need; key-up additionally sets the previous-state and
        # transition bits.
        down = 1 | (scan << 16) | (1 << 24)
        flags = down | (1 << 30) | (1 << 31) if up else down
        self._user32.PostMessageW(
            hwnd, WM_KEYUP if up else WM_KEYDOWN, key, flags)

    def _post_key_to_window(self, hwnd: int, key: int) -> None:
        """Deliver a keystroke to one window without giving it focus."""
        self._post_key(hwnd, key, up=False)
        self._post_key(hwnd, key, up=True)

    def _post_chord(self, hwnd: int, vks: list[int]) -> None:
        """Post a whole chord to one window, modifiers held throughout."""
        mods, base = vks[:-1], vks[-1]
        for mod in mods:
            self._post_key(hwnd, mod, up=False)
        self._post_key(hwnd, base, up=False)
        self._post_key(hwnd, base, up=True)
        for mod in reversed(mods):
            self._post_key(hwnd, mod, up=True)

    # ------------------------------------------------------- event plumbing

    def _key(self, key: int, *, up: bool) -> None:
        """One half of a key tap, so a chord can hold its modifiers down."""
        self._user32.keybd_event(key, 0, KEYEVENTF_KEYUP if up else 0, 0)

    def _press_chord(self, vks: list[int]) -> None:
        """Hold the modifiers, tap the base key, let the modifiers back up."""
        mods, base = vks[:-1], vks[-1]
        for mod in mods:
            self._key(mod, up=False)
        self._key(base, up=False)
        self._key(base, up=True)
        for mod in reversed(mods):
            self._key(mod, up=True)

    def _press(self, key: int, *, extended: bool = False) -> None:
        """Tap a key: press then release, as a real keyboard does."""
        flags = KEYEVENTF_EXTENDEDKEY if extended else 0
        self._user32.keybd_event(key, 0, flags, 0)
        self._user32.keybd_event(key, 0, flags | KEYEVENTF_KEYUP, 0)

    def quit_app(self, name: str) -> str:
        """Ask an app -- or with "all", every windowed app -- to quit.

        CloseMainWindow is the polite close, the same as the X button,
        so unsaved work gets its own save dialog rather than being
        lost.  The desktop (explorer) and QRUDO itself are never
        included in "all".  Quitting an app that is not running
        succeeds by doing nothing -- that IS the asked-for state.
        """

        import re as _re

        if name.strip().lower() == "all":
            script = (
                "$closed = 0; Get-Process | Where-Object { "
                "$_.MainWindowHandle -ne 0 -and $_.Id -ne $PID "
                "-and $_.ProcessName -notin @('explorer','QRUDO') } | "
                "ForEach-Object { if ($_.CloseMainWindow()) "
                "{ $closed++ } }; $closed")
            closed = (self._powershell(script) or "0").strip()
            if closed == "0":
                return "nothing was open to quit"
            return f"asked {closed} apps to quit"

        # The name reaches a PowerShell string: keep it to the plain
        # characters an app name uses, so nothing can escape the quote.
        bare = _re.sub(r"[^A-Za-z0-9 ._-]", "", name).strip()
        if not bare:
            return f"{name} was not running"

        script = (
            f"$closed = 0; Get-Process | Where-Object {{ "
            f"$_.MainWindowHandle -ne 0 -and "
            f"($_.ProcessName -like '*{bare}*' -or "
            f"$_.MainWindowTitle -like '*{bare}*') }} | "
            f"ForEach-Object {{ if ($_.CloseMainWindow()) "
            f"{{ $closed++ }} }}; $closed")
        closed = (self._powershell(script) or "0").strip()

        if closed == "0":
            return f"{name} was not running"

        return f"asked {name} to quit"

    #: PowerShell prologue that teaches it to minimize a window --
    #: Windows' nearest thing to macOS hide: still running, one click
    #: on the taskbar brings it back.
    _MINIMIZER = (
        "$sig = '[DllImport(\"user32.dll\")] public static extern bool "
        "ShowWindowAsync(IntPtr hWnd, int nCmdShow);'; "
        "Add-Type -MemberDefinition $sig -Name Win -Namespace Native; ")

    def hide_app(self, name: str) -> str:
        """Minimize an app's -- or with "all", every -- window."""

        import re as _re

        if name.strip().lower() == "all":
            script = (
                self._MINIMIZER +
                "$hid = 0; Get-Process | Where-Object { "
                "$_.MainWindowHandle -ne 0 -and $_.Id -ne $PID "
                "-and $_.ProcessName -notin @('explorer','QRUDO') } | "
                "ForEach-Object { if ([Native.Win]::ShowWindowAsync("
                "$_.MainWindowHandle, 6)) { $hid++ } }; $hid")
            hid = (self._powershell(script) or "0").strip()
            if hid == "0":
                return "nothing was showing to hide"
            return f"hid {hid} apps"

        bare = _re.sub(r"[^A-Za-z0-9 ._-]", "", name).strip()
        if not bare:
            return f"{name} was not running"

        script = (
            self._MINIMIZER +
            f"$hid = 0; Get-Process | Where-Object {{ "
            f"$_.MainWindowHandle -ne 0 -and "
            f"($_.ProcessName -like '*{bare}*' -or "
            f"$_.MainWindowTitle -like '*{bare}*') }} | "
            f"ForEach-Object {{ if ([Native.Win]::ShowWindowAsync("
            f"$_.MainWindowHandle, 6)) {{ $hid++ }} }}; $hid")
        hid = (self._powershell(script) or "0").strip()

        if hid == "0":
            return f"{name} was not running"

        return f"hid {name}"

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

    # ----------------------------------------------------------- custom actions

    def open_argv(self, argv: list[str]) -> str:
        """Run an argv list for a custom action.

        The ActionRunner emits one macOS-shaped convention -- ``open``,
        ``open -a <app>`` -- because custom actions first landed on macOS;
        "how do I open a thing" is the backend's job, so this method turns
        those shapes into Windows-native launches:

        * ``["open", "-a", <app>]`` -> resolve the installed app and launch it
        * ``["open", <path-or-url>]`` -> ShellExecute (file, folder or URL)
        * anything else (a confirmed, screened shell command) -> run as a
          program with arguments, never through a shell

        Nothing here is re-interpreted beyond those two ``open`` shapes; the
        danger screening already happened in control/actions.
        """
        if not argv:
            raise ControlError("a custom action carried an empty argv")

        if argv[:2] == ["open", "-a"] and len(argv) >= 3:
            app = " ".join(argv[2:])
            exe = self._resolve_app(app)
            if exe is None:
                raise UnsupportedCommand(
                    f"could not find an installed application called {app!r}")
            self._launch(exe)
            return f"launched {app}"

        if argv[0] == "open" and len(argv) == 2:
            target = argv[1]
            try:
                os.startfile(target)
            except OSError as exc:
                raise ControlError(f"could not open {target!r}: {exc}") from exc
            return f"opened {target}"

        return self._run(argv)

    def send_combo(self, combo: str, target_app: str = "") -> str:
        """Press a user-taught keyboard shortcut, e.g. "cmd+shift+n".

        Parses the combo with the same cross-platform parser macOS uses, so a
        gesture saved on a Mac reads the same on Windows, then presses it
        through user32 like the media keys.  With ``target_app`` the chord is
        posted to that app's window even while another window is focused --
        the same route the seek keys use.
        """
        from ..keystroke import ComboError, parse

        try:
            parsed = parse(combo)
        except ComboError as exc:
            raise UnsupportedCommand(str(exc)) from exc

        vks = [_MOD_VK[name] for name in parsed.modifiers]
        base = _keyname_vk(parsed.key_name)
        if base is None:
            raise UnsupportedCommand(
                f"QRUDO does not know the Windows key for {parsed.key_name!r}")

        if target_app:
            hwnd = self._target_window(target_app)
            if hwnd is None:
                raise UnsupportedCommand(
                    f"{target_app} is not running, so its shortcut has "
                    f"nowhere to go")
            self._post_chord(hwnd, vks + [base])
            return f"sent {parsed.describe()} to {target_app}"

        self._press_chord(vks + [base])
        return f"sent {parsed.describe()}"

    def _resolve_app(self, app: str) -> str | None:
        """The executable for an app name, or None if it cannot be found.

        Tries, in order: the name as a literal path, the PATH, the App Paths
        registry key (what ``start`` and ShellExecute actually consult), a
        table of common display names -> exe names, then the Uninstall keys'
        DisplayName -- so "Google Chrome", "chrome", or an explicit path all
        resolve without hardcoding one browser.
        """
        if os.path.isfile(app):
            return app

        exe = shutil.which(app)
        if exe:
            return exe

        exe = self._app_paths(app)
        if exe:
            return exe

        exe_name = _KNOWN_APPS.get(app.strip().lower())
        if exe_name:
            exe = self._app_paths(exe_name) or shutil.which(exe_name)
            if exe:
                return exe

        return self._installed_app(app)

    def _app_paths(self, name: str) -> str | None:
        """The App Paths registry default for an exe name, if it exists."""
        if winreg is None:
            return None

        base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
        for candidate in (name, name + ".exe"):
            for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(root, base + "\\" + candidate) as key:
                        value, _ = winreg.QueryValueEx(key, "")
                except OSError:
                    continue
                if value and os.path.isfile(value):
                    return value
        return None

    def _installed_app(self, app: str) -> str | None:
        """The executable of an installed app found by its DisplayName."""
        if winreg is None:
            return None

        wanted = app.strip().lower()
        uninstall = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
        branches = (
            uninstall,
            r"SOFTWARE\WOW6432Node" + uninstall[len("SOFTWARE"):],
        )
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for branch in branches:
                try:
                    key = winreg.OpenKey(root, branch)
                except OSError:
                    continue
                try:
                    count = winreg.QueryInfoKey(key)[0]
                    for i in range(count):
                        try:
                            sub = winreg.OpenKey(key, winreg.EnumKey(key, i))
                        except OSError:
                            continue
                        try:
                            name, _ = winreg.QueryValueEx(sub, "DisplayName")
                            if name and name.strip().lower() == wanted:
                                exe = self._uninstall_exe(sub, name)
                                if exe:
                                    return exe
                        except OSError:
                            pass
                        finally:
                            sub.Close()
                finally:
                    key.Close()
        return None

    def _uninstall_exe(self, key, display_name: str) -> str | None:
        """The launchable exe from one Uninstall key, or None.

        Prefers the DisplayIcon -- that is what the entry actually launches --
        then an exe named like the display name inside InstallLocation.
        """
        if winreg is None:
            return None

        def value(name):
            try:
                found, _ = winreg.QueryValueEx(key, name)
                return found
            except OSError:
                return None

        icon = value("DisplayIcon")
        if icon:
            icon = icon.split(",", 1)[0].strip('"')
            if icon.lower().endswith(".exe") and os.path.isfile(icon):
                return icon

        location = value("InstallLocation")
        if location and os.path.isdir(location):
            stem = re.sub(r"[^a-z0-9]+", "", display_name.lower())
            for candidate in (stem + ".exe", os.path.basename(location) + ".exe"):
                path = os.path.join(location, candidate)
                if os.path.isfile(path):
                    return path
            exes = [f for f in os.listdir(location)
                    if f.lower().endswith(".exe")
                    and os.path.isfile(os.path.join(location, f))]
            if len(exes) == 1:
                return os.path.join(location, exes[0])
        return None

    def _launch(self, exe: str) -> None:
        """Launch an exe detached, the way ``start`` would."""
        try:
            os.startfile(exe)
        except OSError as exc:
            raise ControlError(f"could not launch {exe!r}: {exc}") from exc

    def _run(self, argv: list[str]) -> str:
        """Run an argv list as a program with its arguments, no shell."""
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=_POWERSHELL_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise ControlError(
                f"{argv[0]} timed out after {_POWERSHELL_TIMEOUT}s") from exc
        except FileNotFoundError as exc:
            raise ControlError(f"{argv[0]} not found on PATH") from exc
        if proc.returncode != 0:
            raise ControlError(
                f"{argv[0]} failed ({proc.returncode}): "
                f"{proc.stderr.strip() or 'no output'}")
        return proc.stdout.strip()

    # -------------------------------------------------------------- preflight

    def _brightness_call(self, worker_line: str, fallback_script: str) -> str:
        """Ask the resident PowerShell; launch a fresh one if it is unavailable."""
        if self._worker is not None:
            reply = self._worker.exchange(worker_line)
            if reply is not None:
                return reply
        return self._powershell(fallback_script)

    def _read_brightness(self) -> int | None:
        """Current brightness percentage, or None if this display cannot say."""
        raw = self._brightness_call("read", _READ_BRIGHTNESS_SCRIPT)
        if raw.startswith("error|"):
            return None
        try:
            return int(raw.split("|")[0])
        except ValueError:
            return None

    def read_state(self) -> dict[str, float]:
        # Volume cannot be read back on Windows without an extra package, so
        # brightness is the only state the self-test can verify here.
        try:
            level = self._read_brightness()
        except ControlError:
            return {}
        return {"brightness": level / 100} if level is not None else {}

    def restore_state(self, state: dict[str, float]) -> None:
        if "brightness" not in state:
            return
        target = round(state["brightness"] * 100)
        current = self._read_brightness()
        if current is not None and current != target:
            # Only relative moves are available, so ask for the exact delta.
            self._set_brightness(target - current)

    def preflight(self) -> list[str]:
        warnings = [
            f"volume moves in {VOLUME_PERCENT_PER_PRESS}% steps on Windows, so a "
            f"{self.config.volume_step}% setting becomes "
            f"{max(1, round(self.config.volume_step / VOLUME_PERCENT_PER_PRESS)) * VOLUME_PERCENT_PER_PRESS}%",
            "brightness goes through PowerShell, which is started at launch and kept "
            "alive; if it ever falls back to launching one per command that costs "
            "~1.4s each, so prefer engine.submit() over execute() in the frame loop",
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
