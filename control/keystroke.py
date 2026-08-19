"""Turning "cmd+shift+n" into something a backend can press.

A custom keystroke is how a taught gesture reaches an action QRUDO has
no code for -- next track, mute, a window-manager shortcut -- without a
per-app handler for each.  The user types the combo the app already
uses; this parses it into a base key plus modifier flags, and each
backend presses that.

Parsing lives here, apart from any OS, so it is unit-tested on every
platform.  The keycodes it yields are macOS virtual keycodes; the
Windows backend maps the same parsed combo to its own when that lands.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Modifier words a user might type, folded to a canonical set.  Both the
#: mac names (cmd, option) and the common ones (win, ctrl, alt) are
#: accepted, because a person types what their keyboard says.
_MODIFIER_ALIASES = {
    "cmd": "cmd", "command": "cmd", "meta": "cmd", "win": "cmd",
    "super": "cmd",
    "ctrl": "ctrl", "control": "ctrl",
    "shift": "shift",
    "alt": "alt", "opt": "alt", "option": "alt",
}

#: The base keys we can name, to a macOS virtual keycode.  Letters,
#: digits, and the handful of named keys a shortcut usually ends on.
_KEYCODES = {
    "a": 0, "b": 11, "c": 8, "d": 2, "e": 14, "f": 3, "g": 5, "h": 4,
    "i": 34, "j": 38, "k": 40, "l": 37, "m": 46, "n": 45, "o": 31,
    "p": 35, "q": 12, "r": 15, "s": 1, "t": 17, "u": 32, "v": 9,
    "w": 13, "x": 7, "y": 16, "z": 6,
    "0": 29, "1": 18, "2": 19, "3": 20, "4": 21, "5": 23, "6": 22,
    "7": 26, "8": 28, "9": 25,
    "space": 49, "return": 36, "enter": 36, "tab": 48, "escape": 53,
    "esc": 53, "left": 123, "right": 124, "down": 125, "up": 126,
    "delete": 51, "backspace": 51,
}

#: CGEventFlags masks, so a backend can set them without importing Quartz
#: to learn the numbers.
FLAG_SHIFT = 0x20000
FLAG_CONTROL = 0x40000
FLAG_ALT = 0x80000
FLAG_CMD = 0x100000

_FLAG_FOR = {"shift": FLAG_SHIFT, "ctrl": FLAG_CONTROL,
             "alt": FLAG_ALT, "cmd": FLAG_CMD}


class ComboError(ValueError):
    """A keystroke combo could not be understood."""


@dataclass(frozen=True)
class Combo:
    """A parsed shortcut: one base key and the modifiers held with it."""

    key_code: int
    flags: int
    key_name: str
    modifiers: tuple  # canonical names, sorted, for display

    def describe(self) -> str:
        parts = list(self.modifiers) + [self.key_name]
        return "+".join(parts)


def parse(combo: str) -> Combo:
    """Parse "cmd+shift+n" (any order, any spacing) into a Combo.

    The last unrecognised-as-a-modifier token is the base key, and there
    must be exactly one.  A combo with no base key, an unknown key, or a
    stray token is refused with a reason -- a gesture that fired a
    keystroke nobody could name would be worse than one that did nothing.
    """

    if not combo or not combo.strip():
        raise ComboError("empty keystroke")

    tokens = [t.strip().lower() for t in combo.replace(" ", "+").split("+")
              if t.strip()]

    if not tokens:
        raise ComboError("empty keystroke")

    modifiers = []
    keys = []

    for token in tokens:
        if token in _MODIFIER_ALIASES:
            modifiers.append(_MODIFIER_ALIASES[token])
        elif token in _KEYCODES:
            keys.append(token)
        else:
            raise ComboError(f"unknown key or modifier: {token!r}")

    if len(keys) != 1:
        raise ComboError(
            "a shortcut needs exactly one base key "
            f"(saw {len(keys)}: {', '.join(keys) or 'none'})")

    key_name = keys[0]
    canonical = tuple(sorted(set(modifiers)))
    flags = 0
    for name in canonical:
        flags |= _FLAG_FOR[name]

    return Combo(key_code=_KEYCODES[key_name], flags=flags,
                 key_name=key_name, modifiers=canonical)


def is_valid(combo: str) -> bool:
    """Whether ``parse`` would accept this, for live UI validation."""

    try:
        parse(combo)
        return True
    except ComboError:
        return False
