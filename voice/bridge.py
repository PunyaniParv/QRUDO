"""Fast, dependency-free voice intent router.

``VoiceIntentRouter`` turns a spoken transcript into exactly one supported
:class:`~control.Command` -- or, for a phrase that names a catalog job, the
existing CUSTOM payload that command carries -- or ``None`` when nothing
supported was said.

This is the minimal "Hey Qrudo" fast path:

    Voice -> STT -> VoiceIntentRouter -> Command -> ControlEngine -> OS

Design rules (milestone constraints):
  * Pure text-in/Command-or-None-out. No microphone, sounddevice, OS calls,
    ControlEngine execution, LLM, or network calls here -- nothing with side
    effects, so it is trivially unit-testable.
  * Only maps phrases to commands the current ControlEngine/backends actually
    support. It never invents new OS capabilities.
  * Built-in phrases map to plain Commands exactly as before.  A phrase that
    names a catalog job ("next track", "open Chrome", ...) resolves through
    control/catalog -- the same table the add-a-gesture form uses -- into the
    existing serialised CUSTOM payload, so voice rides the taught-action path
    a gesture uses instead of owning a second one.  The router only maps
    phrases to job *names*; what a job does is the catalog's fact, not this
    module's.
  * Narrow by intent, not by word. A transcript is matched against explicit,
    command-scoped patterns -- never against stray tokens like "up", "down"
    or "open" -- so unrelated requests (launching apps, typing, screenshots)
    fall through to ``None``.  A catalog job that cannot be resolved is
    unhandled, never an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from control import Command
from control import actions as action_mod
from control import catalog

# Matches the verified command vocabulary that ControlEngine currently handles.
# TARGET_NEXT / TARGET_PREV only appear in their genuine "switch the target
# app" sense -- skipping songs/tracks is deliberately *not* mapped here,
# because that is a different operation than moving where commands are aimed.
# This is the built-in vocabulary; phrases that name a catalog job resolve
# separately, through _CATALOG_PHRASES, into a CUSTOM payload.
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

#: Ordered (catalog job name, regex) pairs -- the spoken names of jobs from
#: control/catalog, in the same style as _PATTERNS.  The job name is the only
#: catalog fact this module knows; what the job does (its keys, the app it
#: opens) lives in the catalog and is resolved through it.  Checked only after
#: _PATTERNS, so a phrase the built-in vocabulary already owns ("pause",
#: "volume up") keeps its plain Command and never becomes a CUSTOM payload.
_CATALOG_PHRASES: tuple[tuple[str, str], ...] = (
    ("Next track", r"\bnext track\b"),
    ("Next track", r"\bnext song\b"),
    ("Next track", r"\bskip (?:to )?(?:the )?next (?:track|song)\b"),
    ("Previous track", r"\bprevious track\b"),
    ("Previous track", r"\bprevious song\b"),
    ("Full screen", r"\bfull screen\b"),
    ("Full screen", r"\bfullscreen\b"),
    ("Mute (in the app)", r"\bmute\b"),
    ("New tab", r"\bnew tab\b"),
    ("Close tab", r"\bclose tab\b"),
    ("Open Chrome", r"\bopen chrome\b"),
    ("Open Chrome", r"\blaunch chrome\b"),
)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    ``"  Turn the Volume UP !!!  "`` -> ``"turn the volume up"``.
    """
    lowered = (text or "").lower()
    collapsed = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", lowered))
    return collapsed.strip()


@dataclass(frozen=True)
class Route:
    """One intent resolved to something ControlEngine can execute.

    ``command`` is always a real Command; ``payload`` is non-empty only
    for :attr:`~control.Command.CUSTOM`, where it is the serialised action
    chain the command pipe already carries for a taught gesture.
    """

    command: Command
    payload: str = ""


def _job_by_name(normalized: str) -> str | None:
    """The catalog job whose plain name is exactly ``normalized``, if any.

    This is the fallback that lets a job added to the catalog be spoken by
    its own name ("open spotify") without touching this module -- the
    router's job table only ever holds the natural-language shortcuts.
    """
    for name in catalog.job_names():
        if normalize(name) == normalized:
            return name
    return None


class VoiceIntentRouter:
    """Maps spoken text to a supported :class:`~control.Command`.

    Pure and stateless: the same transcript always yields the same result and
    nothing outside the object is touched. Safe to call from any thread.

    A built-in phrase yields a plain Command.  A phrase that names a catalog
    job yields :attr:`~control.Command.CUSTOM` whose payload is the job's
    serialised action chain -- the same payload a taught gesture carries --
    so voice executes through the existing custom-action path.  Anything
    else is None: either nothing was understood or the request is a
    capability QRUDO does not have yet -- either way the caller should
    execute nothing.
    """

    def route(self, text: str) -> Route | None:
        """Return the :class:`Route` for ``text``, or ``None``.

        Built-in vocabulary wins over the catalog when a phrase could be
        both, so the plain commands behave exactly as they always did;
        only phrases the built-ins do not own fall through to the catalog.
        A catalog job that cannot be resolved (unknown, malformed) is
        unhandled, never raised.
        """
        normalized = normalize(text)
        if not normalized:
            return None
        for command, pattern in _PATTERNS:
            if re.search(pattern, normalized):
                return Route(command)
        for job_name, pattern in _CATALOG_PHRASES:
            if re.search(pattern, normalized):
                return self._resolve_job(job_name)
        job_name = _job_by_name(normalized)
        if job_name is not None:
            return self._resolve_job(job_name)
        return None

    def classify(self, text: str) -> Command | None:
        """Return the :class:`~control.Command` for ``text``, or ``None``.

        The payload-free view of :meth:`route`: a plain built-in command
        comes back as the Command itself; a phrase that resolves to a
        catalog CUSTOM payload has no single Command, so it comes back as
        None -- exactly what this method returned before the catalog
        existed.  Keep using it where a bare Command is all that is wanted.
        """
        routed = self.route(text)
        if routed is None or routed.payload:
            return None
        return routed.command

    @staticmethod
    def _resolve_job(job_name: str) -> Route | None:
        """The catalog job as a Route, or None if it cannot be built.

        A builtin catalog job becomes its plain Command (the same one the
        built-in vocabulary would have produced), so it never wraps in a
        CUSTOM payload.  Anything else becomes Command.CUSTOM with the
        job's serialised action chain.  A job that fails to resolve or to
        serialise is unhandled -- the caller executes nothing.
        """
        try:
            action = catalog.resolve(job_name)
            if action is None:
                return None
            if action.get("type") == "builtin":
                return Route(Command(action["command"]))
            return Route(Command.CUSTOM, action_mod.serialize(action))
        except Exception:
            return None
