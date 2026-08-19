"""Tunable settings for the control layer.

Every magic number the control engine uses lives here, so the increments can be
changed without touching control logic (section B of the spec: "start with a
fixed increment such as 5%, then make it configurable").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from paths import data_dir

DEFAULT_CONFIG_PATH = data_dir() / "qrudo_config.json"


@dataclass
class ControlConfig:
    # --- volume ------------------------------------------------------------
    volume_step: int = 10
    """Percentage points added/removed per VOLUME_UP / VOLUME_DOWN.

    Ten rather than the five a keyboard uses, because the cost of asking
    is not the same.  Tapping a volume key again is nothing; making a
    gesture again means raising your hand, moving it, and bringing it
    back to where it started before the next one counts.  A step that
    suits a key that can be held down makes a gesture into a chore.
    """

    unmute_on_volume_up: bool = True
    """VOLUME_UP on a muted machine unmutes it instead of silently doing nothing."""

    # --- brightness --------------------------------------------------------
    brightness_step: int = 10
    """Percentage points added/removed per BRIGHTNESS_UP / BRIGHTNESS_DOWN.

    Ten, for the reason volume is ten: a gesture is expensive to repeat
    compared with the key that used to do this.
    """

    # --- media / seeking ---------------------------------------------------
    seek_seconds: int = 10
    """How far REWIND / FORWARD should move, in seconds."""

    seek_step_seconds: int = 5
    """Seconds moved by one arrow-key press in the target player.

    Browser video (YouTube etc.) seeks 5s per arrow press, so the default sends
    two presses to cover ``seek_seconds``.  VLC and QuickTime use 10s -- set
    this to 10 when driving those.
    """

    target_app: str = ""
    """The app QRUDO is controlling, e.g. "Google Chrome" or "Spotify".

    Worth setting.  Without it, PLAY_PAUSE is the keyboard's media key, and
    macOS answers a media key with nothing playing by opening Music -- so
    the first gesture of a demo launches the wrong app.  Naming the app
    sends play/pause to it directly instead.

    It also covers seeking, which otherwise only reaches whichever window
    has keyboard focus.

    macOS matches the app's name; Windows matches any part of a window
    title, so "YouTube" works.  If the app is not running, both fall back
    to the old behaviour rather than failing.
    """

    seek_target_app: str = ""
    """Older name for ``target_app``, still honoured.  Use ``target_app``.

    Seeking has no system-wide key, so it is sent as arrow keys -- and arrow
    keys normally land in the focused window, which is why seeking from the
    simulator hits the terminal instead of the video.  Naming the app here
    routes them to it directly, so you can seek without switching windows.

    macOS: the app's name, e.g. "Google Chrome" (substring match).
    Windows: any part of the window title, e.g. "YouTube".
    Empty means "send to whatever has focus", the old behaviour.
    """

    browser_play_key: str = "media"
    """Which key plays and pauses inside a browser.

    "media" -- the default -- is the system's own now-playing key.  It
    reaches whichever tab actually holds the playing media, in any
    browser, wherever that tab sits, and it is genuinely play/pause on
    YouTube Music rather than next-track.  Three complaints came from
    the old default and this cures them: a fist worked only when the
    video was the front tab, it changed tracks on YouTube Music, and it
    silently did nothing when a chat box was in front.

    Its one hazard is the empty case: sent with nothing playing and no
    app owning the now-playing role, macOS answers it by opening Music.
    So the backend uses it only when that cannot happen -- when
    something is playing, or when QRUDO is the one that paused it -- and
    falls back to the letter for the first press on a fresh, silent
    video.  Audio-playing and role-owning are different questions, and
    that fallback is where the difference is handled rather than
    guessed.

    "k" forces YouTube's letter shortcut (the old default); it lands in
    the front tab and is typed literally if the focus is a search box.
    "space" is the other letter, and scrolls the page when the player
    is not selected.
    """

    browser_seek_keys: str = "arrows"
    """Which keys seek inside a browser.

    "arrows" is left and right, which YouTube reads as five seconds each.
    "jl" is j and l, ten seconds each, which some players use instead.

    Sites differ, and a key that does not seek somewhere usually does
    something else there -- so if seeking works on one site and not
    another, this is the setting, not the gesture.

    For music, ``seek_mode`` set to "track" skips whole tracks instead,
    which is generally what rewind and forward mean there.
    """

    browser_video_titles: str = "youtube"
    """Which front-tab titles the play/pause letter may be sent to.

    Comma-separated substrings, matched case-insensitively against the
    browser's front window title.  The letter can only land in the front
    tab, so a front tab that is not the video means the letter is at best
    useless and at worst typed into a chat box -- a k went into ChatGPT
    exactly that way.  Watching video somewhere other than YouTube, add
    the site: "youtube, vimeo".
    """

    seek_mode: str = "seek"
    """``"seek"`` = move within the current track (arrow keys).
    ``"track"`` = previous/next track (HID media keys)."""

    # --- safety / behaviour ------------------------------------------------
    gesture_cooldown_seconds: float = 1.0
    """How long after any gesture before another can be seen at all.

    One movement crosses many frames, and without this each of them would
    be a command -- one raise would be five volume steps.  It is not the
    only thing preventing that, and not the main one: a raise must also
    come back to where it started before another counts, a turn must go
    quiet, and a held pose must be dropped and made again.  Those do not
    depend on time, so they hold however long the movement takes.

    Measured with the volume gesture repeated as fast as a hand can make
    it: at 0.6 a command lands every 1.5s, because the movement and its
    return take that long anyway.  At 1.0 it is every 1.8s.  The 0.3s is
    paid for margin: those other guards are tuned to the six movements
    that exist, and a gesture added to the tables later brings whatever
    settling it brings -- so the default leans conservative, and this is
    the setting to lower if repeats feel slow.
    """

    cooldown_seconds: float = 0.6
    """Minimum gap between two accepted commands.  Gesture recognition fires
    many frames per second; without this, one hand pose becomes 30 volume
    steps.  Repeats of the same command are throttled per-command."""

    windows_persistent_powershell: bool = True
    """Windows only.  Keep one PowerShell alive for brightness instead of
    launching one per command (~1.4s each).  Set false if it ever misbehaves;
    the slower path is always available as a fallback."""

    dry_run: bool = False
    """Log what would happen but never touch the OS.  Useful for demos and for
    the Vision Engine's own testing."""

    voice_enabled: bool = False
    """Whether the app also listens for voice commands, beside the camera.

    Needs the voice requirements (requirements-voice.txt): faster-whisper,
    openwakeword and sounddevice.  Voice is a second input to the same
    engine -- cooldown, dry-run, logging and the reliability report apply
    to it exactly as they do to a gesture -- and it is best-effort: a
    machine without the requirements, a microphone or a wake-word model
    prints why and runs camera-only, exactly as before."""

    show_preview: bool = True
    """Whether the application window shows the camera's view.  The gestures
    never needed it -- the engine reads frames whether or not anyone watches
    them -- so this is purely about what a person wants to look at."""

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
    def app(self) -> str:
        """The app to aim commands at, under either config name."""

        return (self.target_app or self.seek_target_app).strip()

    @property
    def seek_presses(self) -> int:
        """Number of arrow-key presses needed to cover ``seek_seconds``."""
        return max(1, round(self.seek_seconds / max(1, self.seek_step_seconds)))
