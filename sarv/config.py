"""Tunable settings for the control layer.

Every magic number the control engine uses lives here, so the increments can be
changed without touching control logic (section B of the spec: "start with a
fixed increment such as 5%, then make it configurable").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "sarv_config.json"


@dataclass
class ControlConfig:
    # --- volume ------------------------------------------------------------
    volume_step: int = 5
    """Percentage points added/removed per VOLUME_UP / VOLUME_DOWN."""

    unmute_on_volume_up: bool = True
    """VOLUME_UP on a muted machine unmutes it instead of silently doing nothing."""

    # --- brightness --------------------------------------------------------
    brightness_step: int = 8
    """Percentage points added/removed per BRIGHTNESS_UP / BRIGHTNESS_DOWN."""

    # --- media / seeking ---------------------------------------------------
    seek_seconds: int = 10
    """How far REWIND / FORWARD should move, in seconds."""

    seek_step_seconds: int = 5
    """Seconds moved by one arrow-key press in the target player.

    Browser video (YouTube etc.) seeks 5s per arrow press, so the default sends
    two presses to cover ``seek_seconds``.  VLC and QuickTime use 10s -- set
    this to 10 when driving those.
    """

    seek_mode: str = "seek"
    """``"seek"`` = move within the current track (arrow keys).
    ``"track"`` = previous/next track (HID media keys)."""

    # --- safety / behaviour ------------------------------------------------
    cooldown_seconds: float = 0.6
    """Minimum gap between two accepted commands.  Gesture recognition fires
    many frames per second; without this, one hand pose becomes 30 volume
    steps.  Repeats of the same command are throttled per-command."""

    dry_run: bool = False
    """Log what would happen but never touch the OS.  Useful for demos and for
    the Vision Engine's own testing."""

    log_dir: str = "logs"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ControlConfig":
        """Load config from JSON, falling back to defaults if the file is absent.

        Unknown keys in the file are ignored rather than crashing, so an older
        build never dies on a newer config.
        """
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @property
    def seek_presses(self) -> int:
        """Number of arrow-key presses needed to cover ``seek_seconds``."""
        return max(1, round(self.seek_seconds / max(1, self.seek_step_seconds)))
