"""Fast, dependency-free voice intent router.

``VoiceIntentRouter`` turns a spoken transcript into exactly one supported
:class:`~control.Command`, or ``None`` when nothing supported was said.

This is the minimal "Hey Qrudo" fast path:

    Voice -> STT -> VoiceIntentRouter -> Command -> ControlEngine -> OS

Design rules (milestone constraints):
  * Pure text-in/Command-or-None-out. No microphone, sounddevice, OS calls,
    ControlEngine execution, LLM, or network calls here -- nothing with side
    effects, so it is trivially unit-testable.
  * Only maps phrases to commands the current ControlEngine/backends actually
    support. It never invents new OS capabilities.
  * Narrow by intent, not by word. A transcript is matched against explicit,
    command-scoped patterns -- never against stray tokens like "up", "down"
    or "open" -- so unrelated requests (launching apps, typing, screenshots)
    fall through to ``None``.
"""

from __future__ import annotations

import re

from control import Command

# Matches the verified command vocabulary that ControlEngine currently handles.
# TARGET_NEXT / TARGET_PREV only appear in their genuine "switch the target
# app" sense -- skipping songs/tracks is deliberately *not* mapped, because
# that is a different operation than moving where commands are aimed.
_SUPPORTED: tuple[Command, ...] = (
    Command.VOLUME_UP,
    Command.VOLUME_DOWN,
    Command.PLAY_PAUSE,
    Command.REWIND,
    Command.FORWARD,
    Command.BRIGHTNESS_UP,
    Command.BRIGHTNESS_DOWN,
    Command.TARGET_NEXT,
    Command.TARGET_PREV,
)

#: Ordered (Command, regex) pairs. The regexes run against a normalized
#: lowercase, punctuation-stripped, single-spaced transcript. Order matters
#: only where two phrases could overlap; each family below is confined to its
#: own command, so collisions stay internal to one group.
_PATTERNS: tuple[tuple[Command, str], ...] = (
    # --- VOLUME_UP -------------------------------------------------------
    (Command.VOLUME_UP, r"\bvolume up\b"),
    (Command.VOLUME_UP, r"\bincrease (?:the )?volume\b"),
    (Command.VOLUME_UP, r"\braise (?:the )?volume\b"),
    (Command.VOLUME_UP, r"\bturn (?:it|the volume) up\b"),
    (Command.VOLUME_UP, r"\bmake it louder\b"),
    (Command.VOLUME_UP, r"\bcrank (?:it|the volume) up\b"),

    # --- VOLUME_DOWN -----------------------------------------------------
    (Command.VOLUME_DOWN, r"\bvolume down\b"),
    (Command.VOLUME_DOWN, r"\b(?:decrease|lower|reduce) (?:the )?volume\b"),
    (Command.VOLUME_DOWN, r"\bturn (?:it|the volume) down\b"),
    (Command.VOLUME_DOWN, r"\bmake it quieter\b"),

    # --- PLAY_PAUSE ------------------------------------------------------
    (Command.PLAY_PAUSE, r"\bplay pause\b"),
    (Command.PLAY_PAUSE, r"\bplay (?:the )?music\b"),
    (Command.PLAY_PAUSE, r"\bpause\b"),

    # --- REWIND ----------------------------------------------------------
    (Command.REWIND, r"\brewind\b"),
    (Command.REWIND, r"\bgo back\b"),
    (Command.REWIND, r"\bskip back\b"),

    # --- FORWARD ---------------------------------------------------------
    (Command.FORWARD, r"\bfast forward\b"),
    (Command.FORWARD, r"\bskip forward\b"),
    (Command.FORWARD, r"\bforward\b"),

    # --- BRIGHTNESS_UP ---------------------------------------------------
    (Command.BRIGHTNESS_UP, r"\bbrightness up\b"),
    (Command.BRIGHTNESS_UP, r"\bincrease (?:the )?brightness\b"),
    (Command.BRIGHTNESS_UP, r"\braise (?:the )?brightness\b"),
    (Command.BRIGHTNESS_UP, r"\bmake (?:it|the screen) brighter\b"),

    # --- BRIGHTNESS_DOWN -------------------------------------------------
    (Command.BRIGHTNESS_DOWN, r"\bbrightness down\b"),
    (Command.BRIGHTNESS_DOWN, r"\b(?:decrease|lower|reduce) (?:the )?brightness\b"),
    (Command.BRIGHTNESS_DOWN, r"\bmake (?:it|the screen) darker\b"),

    # --- TARGET_NEXT (genuine "switch target app" intent only) -----------
    (Command.TARGET_NEXT, r"\bnext target\b"),
    (Command.TARGET_NEXT, r"\bnext app\b"),
    (Command.TARGET_NEXT, r"\bswitch (?:to |to the |the )?(?:next )?(?:target|app)\b"),

    # --- TARGET_PREV -----------------------------------------------------
    (Command.TARGET_PREV, r"\bprevious target\b"),
    (Command.TARGET_PREV, r"\bprevious app\b"),
    (Command.TARGET_PREV, r"\bprev(?:ious)? (?:target|app)\b"),
    (Command.TARGET_PREV, r"\bswitch (?:to |to the |the )?previous (?:target|app)\b"),
)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    ``"  Turn the Volume UP !!!  "`` -> ``"turn the volume up"``.
    """
    lowered = (text or "").lower()
    collapsed = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", lowered))
    return collapsed.strip()


class VoiceIntentRouter:
    """Maps spoken text to a supported :class:`~control.Command`.

    Pure and stateless: the same transcript always yields the same result and
    nothing outside the object is touched. Safe to call from any thread.
    """

    def classify(self, text: str) -> Command | None:
        """Return the :class:`~control.Command` for ``text``, or ``None``.

        ``None`` means either nothing was understood or the request is a
        capability SARV does not have yet -- either way the caller should
        execute nothing.
        """
        normalized = normalize(text)
        if not normalized:
            return None
        for command, pattern in _PATTERNS:
            if re.search(pattern, normalized):
                return command
        return None