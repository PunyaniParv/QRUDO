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

        if self.kind not in ("pose", "move"):
            raise CustomError(f"unknown kind {self.kind!r}")

        if self.kind == "move" and self.direction not in (
                "left", "right", "up", "down"):
            raise CustomError(
                "a move gesture needs a direction: left, right, up or down")

        if self.binding_type == "action" and not self.command:
            raise CustomError("an action gesture needs a command")

        if self.binding_type == "keystroke" and not self.combo:
            raise CustomError("a keystroke gesture needs a combo")

        if self.binding_type not in ("action", "keystroke"):
            raise CustomError(f"unknown binding {self.binding_type!r}")

    def distance(self, spans: dict) -> float:
        """How far a live hand's fingers sit from this gesture's shape.

        Euclidean over the four extensions -- the same measurement the
        built-in classifier reads, never a threshold it consults.  A
        hand nearer than ``tolerance`` is a match.
        """

        return sum((spans.get(f, 0.0) - self.signature[f]) ** 2
                   for f in FINGERS) ** 0.5


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
        except (CustomError, TypeError, ValueError):
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


def match(spans: dict) -> str | None:
    """The custom gesture this hand matches, or None.

    Reached only after the built-in classifier has returned UNKNOWN --
    that ordering is the isolation, and it lives at the one call site in
    gestures.py, not here.  Here the rule is only: nearest signature
    within its tolerance wins, and a tie between two rejects rather than
    guessing.  Poses only; a "move" gesture is matched for its shape here
    and its direction is confirmed by the existing swipe detector, the
    same machinery the built-in movements use.
    """

    if not _active:
        return None

    scored = sorted(
        ((g.distance(spans), g) for g in _active),
        key=lambda pair: pair[0])

    best_distance, best = scored[0]

    if best_distance > best.tolerance:
        return None

    # A tie -- two shapes equally, genuinely close -- is ambiguous, and a
    # gesture-control app must not guess which action a person meant.
    if len(scored) > 1:
        second_distance = scored[1][0]
        if second_distance <= best.tolerance and \
                abs(second_distance - best_distance) < 1e-6:
            return None

    return best.name


def by_name(name: str) -> CustomGesture | None:
    for gesture in _active:
        if gesture.name == name:
            return gesture

    return None


def remove(name: str, path: str | Path | None = None) -> Path:
    """Drop a gesture by name.  A name that is not there is not an error."""

    name = name.strip().upper()
    kept = [g for g in load_all(path) if g.name != name]

    return save_all(kept, path)
