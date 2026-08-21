"""Gestures a user teaches QRUDO, kept apart from the ones it ships with.

The eight built-in gestures are reliable because each was tuned by hand
-- the finger-cliff maths, the crosstalk floors.  A shape a user records
has no such tuning, so the one rule that governs this whole module is
isolation: a custom gesture is matched only after the built-in
classifier has already said UNKNOWN, and it never reads or writes a
single built-in threshold.  If the built-ins recognise a shape, nothing
here runs that frame, and the shipped gestures stay exactly as good as
they were.

Everything a user teaches lives in its own file, qrudo_gestures.json,
which never touches qrudo_calibration.json -- so a hand-edited or
corrupt custom file cannot damage the built-in calibration.

This module holds the representation and the store.  The matcher and the
runtime hook are the next phase; they live here too but are added
separately so the isolation contract can be tested before anything uses
it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from paths import data_dir

STORE_PATH = data_dir() / "qrudo_gestures.json"

#: The fingers a signature is measured over, in the project's usual order.
FINGERS = ("index", "middle", "ring", "pinky")

#: Names a custom gesture may never take, because the built-in pipeline
#: already means something by them.  A save that collides is refused
#: rather than silently shadowing a shipped gesture.
RESERVED = {
    "FIST", "POINT", "TWO_FINGER", "OPEN_PALM", "UNKNOWN", "NONE",
    "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN",
    "PALM_UP", "PALM_DOWN",
}

#: The accept radius can never be set looser than this, whatever a
#: sloppy recording measured.  A custom gesture that matched half the
#: shapes a hand can make would fire constantly; the floor keeps the
#: match meaningfully close to what was recorded.
MIN_TOLERANCE = 0.04

#: Nor tighter than this, or natural variation in a held hand -- the
#: wobble the built-ins already tolerate -- would keep a genuine repeat
#: from ever matching.
MAX_TOLERANCE = 0.30


class CustomError(ValueError):
    """A custom gesture could not be built or saved as asked."""


@dataclass
class CustomGesture:
    """One gesture a user taught, and what it should do.

    ``signature`` is the per-finger extension the shape was recorded at,
    in the same 0-1 space as ``hand_state.finger_span`` -- so matching is
    a distance in the very numbers the built-in classifier measures, with
    no new threshold surface of its own.

    ``kind`` is "pose" for a held shape, or "move" for a shape carried in
    a direction.  ``direction`` names that direction for a move
    ("left"/"right"/"up"/"down"), empty for a pose.

    ``binding`` says what firing does: an action already in the command
    vocabulary, or a keystroke combo typed by the user.
    """

    name: str
    signature: dict            # {finger: extension}
    tolerance: float
    kind: str = "pose"
    direction: str = ""
    #: The thumb-to-index gap the shape was recorded at, in palm-lengths,
    #: or None for a gesture recorded before this existed.  It is what
    #: tells a closed hole (thumb touching, ~0) from an open C (the same
    #: fingers, apart) -- finger extension alone cannot.
    thumb_gap: float | None = None
    #: The OTHER hand's shape, for a gesture made with two hands at
    #: once -- None for the ordinary one-hand gesture.  A two-hand
    #: gesture only matches when both live hands sit within tolerance
    #: of the pair, in either left/right assignment.
    partner_signature: dict | None = None
    partner_thumb_gap: float | None = None
    #: The Karabiner-style switch: a gesture turned off stays taught
    #: and listed, but never matches -- paused, not forgotten.
    enabled: bool = True
    #: What firing does: an ordered list of actions (the source of truth).
    #: A gesture saved before chains existed carries the legacy binding
    #: fields instead, and the normaliser below turns those into a
    #: one-element ``actions`` list -- so old files keep working and new
    #: builds read one shape.
    actions: list = field(default_factory=list)
    #: Legacy, kept writable so a downgrade does not brick the store.
    binding_type: str = "action"   # "action" | "keystroke"
    command: str = ""              # when binding_type == "action"
    combo: str = ""                # when binding_type == "keystroke"
    target_app: str = ""           # "" means fall back to the global target

    def __post_init__(self):
        self.name = self.name.strip().upper()

        if not self.name:
            raise CustomError("a custom gesture needs a name")

        if self.name in RESERVED:
            raise CustomError(
                f"{self.name!r} is a built-in gesture name -- pick another")

        missing = [f for f in FINGERS if f not in self.signature]
        if missing:
            raise CustomError(f"signature is missing {', '.join(missing)}")

        # Keep only the fingers we know, as floats, so a stray key in a
        # hand-edited file cannot widen the distance metric.
        self.signature = {f: float(self.signature[f]) for f in FINGERS}

        self.tolerance = min(MAX_TOLERANCE,
                             max(MIN_TOLERANCE, float(self.tolerance)))

        # The gap joins the distance metric, so it gets the same
        # coercion as the spans do.  Stored uncast, a non-number here
        # loads without complaint, sits dormant, and then TypeErrors
        # inside distance() the first time a hand comes near this
        # shape -- which is the per-frame loop, which takes the camera
        # down with it.  Cast at load, so a bad value fails HERE, where
        # load_all drops just this entry.
        if self.thumb_gap is not None:
            self.thumb_gap = float(self.thumb_gap)

        # The second hand gets exactly the scrutiny of the first: known
        # fingers only, as floats, gap coerced -- a corrupt entry fails
        # HERE, where load_all drops just this gesture.
        if self.partner_signature is not None:
            missing = [f for f in FINGERS
                       if f not in self.partner_signature]
            if missing:
                raise CustomError(
                    f"partner signature is missing {', '.join(missing)}")

            self.partner_signature = {
                f: float(self.partner_signature[f]) for f in FINGERS}

        if self.partner_thumb_gap is not None:
            self.partner_thumb_gap = float(self.partner_thumb_gap)

        self.enabled = bool(self.enabled)

        if self.kind not in ("pose", "move"):
            raise CustomError(f"unknown kind {self.kind!r}")

        if self.kind == "move" and self.direction not in (
                "left", "right", "up", "down"):
            raise CustomError(
                "a move gesture needs a direction: left, right, up or down")

        self._resolve_actions()

    def _resolve_actions(self):
        """Settle on a validated ``actions`` list, source of truth.

        If actions were given, they win and are validated.  If not, the
        legacy binding fields are turned into a one-element chain -- an
        "action" binding into a builtin action, a "keystroke" binding
        into a keystroke action -- so a gesture saved by an older build
        keeps doing exactly what it did.
        """

        from control.actions import ActionError as _AE
        from control.actions import normalise

        if self.actions:
            try:
                self.actions = normalise(self.actions)
            except _AE as exc:
                raise CustomError(str(exc)) from exc
            return

        if self.binding_type == "keystroke":
            if not self.combo:
                raise CustomError("a keystroke gesture needs a combo")
            self.actions = [{"type": "keystroke", "combo": self.combo}]
        elif self.binding_type == "action":
            if not self.command:
                raise CustomError("an action gesture needs a command")
            self.actions = [{"type": "builtin", "command": self.command}]
        else:
            raise CustomError(f"unknown binding {self.binding_type!r}")

    def distance(self, spans: dict, live_gap: float | None = None) -> float:
        """How far a live hand sits from this gesture's shape.

        Euclidean over the four finger extensions -- the same
        measurement the built-in classifier reads, never a threshold it
        consults.  When both this gesture and the live hand carry a
        thumb gap, that difference joins the distance, so a closed hole
        and an open C -- identical in extension -- are told apart.  A
        hand nearer than ``tolerance`` is a match.
        """

        return self._one(self.signature, self.thumb_gap, spans, live_gap)

    @staticmethod
    def _one(signature, thumb_gap, spans, live_gap):
        total = sum((spans.get(f, 0.0) - signature[f]) ** 2
                    for f in FINGERS)

        if thumb_gap is not None and live_gap is not None:
            total += (live_gap - thumb_gap) ** 2

        return total ** 0.5

    def pair_distance(self, spans, live_gap, partner_spans, partner_gap):
        """How far a live PAIR of hands sits from this two-hand shape.

        Both hands must fit -- the worse of the two distances counts --
        and nobody promises which physical hand recorded which half, so
        both assignments are tried and the better one is the answer.
        """

        straight = max(
            self._one(self.signature, self.thumb_gap, spans, live_gap),
            self._one(self.partner_signature, self.partner_thumb_gap,
                      partner_spans, partner_gap))
        swapped = max(
            self._one(self.signature, self.thumb_gap,
                      partner_spans, partner_gap),
            self._one(self.partner_signature, self.partner_thumb_gap,
                      spans, live_gap))

        return min(straight, swapped)


def _known_fields():
    return {f.name for f in fields(CustomGesture)}


def load_all(path: str | Path | None = None) -> list[CustomGesture]:
    """Every taught gesture, or an empty list.

    A store that is missing, empty, or partly corrupt yields whatever
    valid gestures it holds and drops the rest -- a bad entry must never
    stop the good ones loading, and must never reach the runtime.
    """

    path = Path(path) if path else STORE_PATH

    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return []

    if not isinstance(raw, list):
        return []

    known = _known_fields()
    gestures = []

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            gestures.append(CustomGesture(
                **{k: v for k, v in entry.items() if k in known}))
        except Exception:
            # Anything.  A named tuple of expected errors once let a
            # null name slip through as AttributeError and crash the
            # whole app at startup, before any window existed -- over
            # one bad line in a store file.  The contract above is
            # absolute: this entry is dropped, the rest load.
            continue

    return gestures


def save_all(gestures: list[CustomGesture],
             path: str | Path | None = None) -> Path:
    """Write the whole set, refusing a name that collides with another.

    The write is atomic -- a temp file renamed into place -- so an
    interrupted save leaves the old set intact rather than a half file.
    """

    path = Path(path) if path else STORE_PATH

    seen = set()
    for gesture in gestures:
        if gesture.name in seen:
            raise CustomError(f"two custom gestures both named {gesture.name!r}")
        seen.add(gesture.name)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([asdict(g) for g in gestures], indent=2) + "\n"

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(path)

    return path


def add(gesture: CustomGesture, path: str | Path | None = None) -> Path:
    """Add one gesture to the store, replacing any of the same name."""

    kept = [g for g in load_all(path) if g.name != gesture.name]
    kept.append(gesture)

    return save_all(kept, path)


# ---------------------------------------------------------
# Runtime matching -- the isolated half
# ---------------------------------------------------------

#: The gestures the matcher considers, loaded once and refreshed when the
#: user changes them.  Empty by default, so a build with no custom
#: gestures pays nothing and behaves exactly as before.
_active: list[CustomGesture] = []


def load(path: str | Path | None = None) -> int:
    """(Re)load the active set.  Returns how many are now live.

    Called at startup and after every save, so a newly taught gesture
    works without a restart.
    """

    global _active
    _active = load_all(path)

    return len(_active)


def active() -> list[CustomGesture]:
    return list(_active)


def _nearest(scored) -> str | None:
    """Nearest within its tolerance wins; a dead tie rejects.

    A tie -- two shapes equally, genuinely close -- is ambiguous, and a
    gesture-control app must not guess which action a person meant.
    """

    if not scored:
        return None

    scored = sorted(scored, key=lambda pair: pair[0])
    best_distance, best = scored[0]

    if best_distance > best.tolerance:
        return None

    if len(scored) > 1:
        second_distance = scored[1][0]
        if second_distance <= best.tolerance and \
                abs(second_distance - best_distance) < 1e-6:
            return None

    return best.name


def match(spans: dict, thumb_gap: float | None = None,
          partner_spans: dict | None = None,
          partner_gap: float | None = None) -> str | None:
    """The custom gesture this hand -- or pair of hands -- matches.

    Reached only after the built-in classifier has returned UNKNOWN --
    that ordering is the isolation, and it lives at the one call site in
    gestures.py, not here.  ``thumb_gap`` -- the live thumb-to-index
    distance -- joins the distance for gestures that recorded one, so a
    closed hole and an open C are told apart.

    With a second hand live, the two-hand gestures are asked first and
    a fit among them wins outright: a pair is the more specific claim,
    and a person holding up two deliberate shapes did not mean the
    one-hand gesture that half of it resembles.
    """

    if not _active:
        return None

    if partner_spans is not None:
        paired = _nearest([
            (g.pair_distance(spans, thumb_gap, partner_spans, partner_gap),
             g)
            for g in _active
            if g.enabled and g.partner_signature is not None])

        if paired is not None:
            return paired

    return _nearest([
        (g.distance(spans, thumb_gap), g)
        for g in _active if g.enabled and g.partner_signature is None])


def by_name(name: str) -> CustomGesture | None:
    for gesture in _active:
        if gesture.name == name:
            return gesture

    return None


def set_enabled(name: str, on: bool,
                path: str | Path | None = None) -> bool:
    """Flip one gesture's switch on disk and in the live registry.

    Returns whether the gesture was found.  Off means paused, not
    forgotten: it stays listed and taught, and simply never matches.
    """

    gestures = load_all(path)
    found = False

    for gesture in gestures:
        if gesture.name == name.strip().upper():
            gesture.enabled = bool(on)
            found = True

    if found:
        save_all(gestures, path)
        load(path)

    return found


def remove(name: str, path: str | Path | None = None) -> Path:
    """Drop a gesture by name.  A name that is not there is not an error."""

    name = name.strip().upper()
    kept = [g for g in load_all(path) if g.name != name]

    return save_all(kept, path)
