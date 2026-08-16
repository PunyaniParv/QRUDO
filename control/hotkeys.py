"""Global hotkeys: drive SARV from any app, without the terminal focused.

    python main.py --hotkeys

This is a fallback, not the product.  Gestures do not need it -- the camera
loop reads frames whether or not SARV has focus, so the vision half never has
to capture a keypress.  What this buys is demo insurance: if the lighting is
bad or recognition misfires in front of an audience, you can drive the same
seven commands from the keyboard while the video stays fullscreen.

Chords are ctrl+alt plus the simulator's letters, so a bare "u" keeps typing a
u everywhere on the machine.  Matched chords are swallowed, so the app you are
in never sees them.

Both platforms need permission to watch the keyboard: Accessibility on macOS
(the same grant the media keys need), nothing on Windows.
"""

from __future__ import annotations

import sys

from . import log
from .commands import Command
from .executor import ControlEngine
from .simulator import KEY_MAP

MODIFIERS = "ctrl+alt"

#: macOS virtual key codes for the simulator's letters.
_MAC_KEYCODES = {32: "u", 2: "d", 35: "p", 37: "l", 15: "r", 11: "b", 45: "n"}

#: The arrow keys, for the target chords.
_MAC_ARROWS = {123: "left", 124: "right"}

#: Windows virtual key codes for letters are just the uppercase ASCII
#: values.  Letters only: the target commands travel as ctrl+shift+
#: arrows, not as letter chords.
_WIN_KEYCODES = {ord(letter.upper()): letter
                 for letter in KEY_MAP if letter.isalpha()}


def banner(engine: ControlEngine) -> str:
    lines = ["", f"  SARV global hotkeys  --  backend: {engine.controller.name}", ""]
    for letter, command in KEY_MAP.items():
        lines.append(f"    {MODIFIERS}+{letter}   {command.value}")
    lines += ["", "  works in any app; ctrl+c here to stop", ""]
    return "\n".join(lines)


#: ctrl+shift+arrows step the target -- which app the fist and the
#: seeks land in -- from anywhere, preview window or not.  Available in
#: the --hotkeys listener and, through watch_targets, while the camera
#: runs.
TARGET_KEYS = {"left": Command.TARGET_PREV, "right": Command.TARGET_NEXT}


def run(engine: ControlEngine | None = None, *, targets_only: bool = False,
        announce: bool = True) -> int:
    """Listen for the chords.  ``targets_only`` is the embedded mode: the
    camera loop runs this on a daemon thread for ctrl+shift+arrows, and
    the letter chords stay exclusive to the dedicated --hotkeys mode --
    always-on volume letters would surprise; a deliberate target switch
    does not.  Embedded, the engine is shared, so it is not closed here.
    """

    engine = engine or ControlEngine()
    if announce:
        for warning in engine.preflight():
            print(f"  ! {warning}\n", file=sys.stderr)
        print(banner(engine))

    if sys.platform == "darwin":
        return _run_macos(engine, targets_only=targets_only)
    if sys.platform == "win32":
        return _run_windows(engine, targets_only=targets_only)
    if announce:
        print(f"  global hotkeys are not implemented for {sys.platform}; "
              f"use --simulate instead", file=sys.stderr)
    return 1


def watch_targets(engine: ControlEngine):
    """ctrl+shift+left/right while the camera runs, on a daemon thread.

    Best effort: a machine without the permission (or the platform)
    simply goes without the chords, and the gesture and the config still
    switch targets.  Returns the thread, or None when nothing started.
    """

    if sys.platform not in ("darwin", "win32"):
        return None

    import threading

    def worker():
        try:
            run(engine, targets_only=True, announce=False)
        except Exception as exc:
            log.get_logger("hotkeys").warning(
                "target chords unavailable: %s", exc)

    thread = threading.Thread(target=worker, name="sarv-target-keys",
                              daemon=True)
    thread.start()
    return thread


def _fire(engine: ControlEngine, letter: str) -> None:
    """Queue the command and return immediately.

    Both platforms give the keyboard callback a short deadline -- macOS
    silently disables a tap that takes too long -- and a brightness command can
    take over a second.  submit() hands it to the worker thread instead.
    """
    command = KEY_MAP.get(letter)
    if command is not None:
        engine.submit(command)


def _report(engine: ControlEngine) -> None:
    """Print each result as it completes, since the worker runs off-thread."""
    def show(result):
        marker = "ok " if result.ok else "ERR"
        print(f"    [{marker}] {result.command}: {result.detail or result.error}")
    engine.on_result = show


# ------------------------------------------------------------------- macOS

def _run_macos(engine: ControlEngine, targets_only: bool = False) -> int:
    try:
        import Quartz
    except ImportError:
        print("  global hotkeys need pyobjc-framework-Quartz", file=sys.stderr)
        return 1

    if not targets_only:
        _report(engine)
    state = {}

    def callback(proxy, event_type, event, refcon):
        # macOS disables a tap that takes too long; re-enable and carry on.
        if event_type in (Quartz.kCGEventTapDisabledByTimeout,
                          Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(state["tap"], True)
            return event

        flags = Quartz.CGEventGetFlags(event)
        keycode = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGKeyboardEventKeycode)

        # ctrl+shift+arrows step the target, in both modes.
        if (flags & Quartz.kCGEventFlagMaskControl
                and flags & Quartz.kCGEventFlagMaskShift):
            arrow = _MAC_ARROWS.get(keycode)
            if arrow is not None:
                engine.submit(TARGET_KEYS[arrow])
                return None  # swallow it

        if targets_only:
            return event

        if not (flags & Quartz.kCGEventFlagMaskControl
                and flags & Quartz.kCGEventFlagMaskAlternate):
            return event

        letter = _MAC_KEYCODES.get(keycode)
        if letter is None:
            return event

        _fire(engine, letter)
        return None  # swallow it, so the focused app never sees the chord

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,   # not listen-only: we consume matches
        Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
        callback,
        None,
    )
    if tap is None:
        print("  could not watch the keyboard. Grant Accessibility to this app in\n"
              "  System Settings > Privacy & Security > Accessibility.", file=sys.stderr)
        return 1
    state["tap"] = tap

    source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source,
                              Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)
    try:
        Quartz.CFRunLoopRun()
    except KeyboardInterrupt:
        pass
    finally:
        Quartz.CGEventTapEnable(tap, False)
        # Embedded, the engine belongs to the camera loop; leave it be.
        if not targets_only:
            engine.close()
    if not targets_only:
        print("\n  bye.")
    return 0


# ----------------------------------------------------------------- Windows

def _run_windows(engine: ControlEngine, targets_only: bool = False) -> int:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    WH_KEYBOARD_LL = 13
    WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
    VK_CONTROL, VK_MENU = 0x11, 0x12
    HELD = 0x8000

    # Every handle here is pointer-sized.  ctypes assumes a 32-bit int return
    # for anything undeclared, which silently truncates them on 64-bit Python
    # -- that is what made SetWindowsHookExW fail with error 126.
    LRESULT = ctypes.c_ssize_t
    HHOOK = wintypes.HANDLE

    class KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD),
                    ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]

    _report(engine)
    proc_type = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int,
                                   wintypes.WPARAM, wintypes.LPARAM)

    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    user32.SetWindowsHookExW.argtypes = [ctypes.c_int, proc_type,
                                         wintypes.HMODULE, wintypes.DWORD]
    user32.SetWindowsHookExW.restype = HHOOK
    user32.UnhookWindowsHookEx.argtypes = [HHOOK]
    user32.UnhookWindowsHookEx.restype = wintypes.BOOL
    user32.CallNextHookEx.argtypes = [HHOOK, ctypes.c_int,
                                      wintypes.WPARAM, wintypes.LPARAM]
    user32.CallNextHookEx.restype = LRESULT
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                   wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = ctypes.c_int  # -1 signals an error

    VK_SHIFT = 0x10
    _WIN_ARROWS = {0x25: "left", 0x27: "right"}

    def proc(code, message, data):
        if code == 0 and message in (WM_KEYDOWN, WM_SYSKEYDOWN):
            key = ctypes.cast(data, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents.vkCode

            # ctrl+shift+arrows step the target, in both modes.
            if (user32.GetAsyncKeyState(VK_CONTROL) & HELD
                    and user32.GetAsyncKeyState(VK_SHIFT) & HELD):
                arrow = _WIN_ARROWS.get(key)
                if arrow is not None:
                    engine.submit(TARGET_KEYS[arrow])
                    return 1  # swallow it

            if not targets_only and (
                    user32.GetAsyncKeyState(VK_CONTROL) & HELD
                    and user32.GetAsyncKeyState(VK_MENU) & HELD):
                letter = _WIN_KEYCODES.get(key)
                if letter is not None:
                    _fire(engine, letter)
                    return 1  # swallow it
        return user32.CallNextHookEx(None, code, message, data)

    callback = proc_type(proc)  # keep a reference, or it is garbage collected
    module = kernel32.GetModuleHandleW(None)
    hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, callback, module, 0)
    if not hook:
        # A low-level hook may also be installed with no module at all; try
        # that before giving up.
        hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, callback, None, 0)
    if not hook:
        print(f"  could not watch the keyboard (error {ctypes.get_last_error()})",
              file=sys.stderr)
        return 1

    # A low-level hook only fires while its thread pumps messages.
    message = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
    except KeyboardInterrupt:
        pass
    finally:
        user32.UnhookWindowsHookEx(hook)
        # Embedded, the engine belongs to the camera loop; leave it be.
        if not targets_only:
            engine.close()
    if not targets_only:
        print("\n  bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
