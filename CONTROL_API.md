# SARV Control Engine — interface for the Vision Engine

This is everything the vision side needs to know. You never import `cv2` or
`mediapipe` into the control layer, and the control layer never imports them
into you.

## The contract

```python
from control import Command, ControlEngine

engine = ControlEngine()          # build once, at startup

# ...inside your frame loop, when a gesture is recognised:
result = engine.execute(Command.VOLUME_UP)
```

That's it. Rules the control side guarantees:

- **`execute()` never raises.** A dead audio device, a revoked permission, a
  timeout — all come back as a `CommandResult`, so a failed OS action can't
  crash the camera loop.
- **`execute()` is fast** (a few ms to ~200 ms). See below if that's too slow
  for your frame loop.
- **`Command.NONE` is a safe no-op.** Emit it every frame where no gesture is
  recognised — no null checks needed on your side.
- **Repeats are debounced for you.** Holding a gesture for 30 frames does not
  fire 30 volume steps; anything inside the cooldown window (default 0.6 s)
  comes back as `THROTTLED`. If you want your own debounce instead, set
  `cooldown_seconds: 0` in the config.

## The seven commands

| Command | macOS | Windows |
|---|---|---|
| `VOLUME_UP` | +5 % (unmutes if muted) | +4 % (2 key presses of 2 %) |
| `VOLUME_DOWN` | −5 % | −4 % |
| `PLAY_PAUSE` | play/pause media key — works in any app | same |
| `REWIND` | seek back ~10 s | same |
| `FORWARD` | seek forward ~10 s | same |
| `BRIGHTNESS_UP` | +8 % on the built-in display | +8 % via WMI, laptop screens only |
| `BRIGHTNESS_DOWN` | −8 % | −8 % |
| `NONE` | nothing | nothing |

Windows moves the volume in fixed 2 % notches, so a 5 % setting becomes 4 %.
Everything above the backend — the interface, the debouncing, the logging — is
identical on both, so you write your gesture code once.

Windows also cannot read the current volume back (that needs an extra package),
so if you start `--selftest` at exactly 0 % or 100 % volume, it can finish one
notch off. Brightness reads back correctly on both platforms.

Command names are also plain strings — `Command.VOLUME_UP == "VOLUME_UP"` — so
they survive JSON, sockets, or a log file if we later split the two engines
into separate processes.

## Which window a command lands in

Two of the seven are sensitive to keyboard focus:

| Commands | Where they go |
|---|---|
| `VOLUME_UP/DOWN`, `PLAY_PAUSE`, `BRIGHTNESS_UP/DOWN` | system-wide — focus is irrelevant |
| `REWIND`, `FORWARD` | the **focused window**, because seeking has no system-wide key and we send arrow keys |

That's why seeking from `--simulate` appears to do nothing: you are typing into
the terminal, so the terminal is focused and swallows the arrows.

**Name the app you are controlling.** In `sarv_config.json`:

```json
{ "target_app": "Google Chrome" }
```

This is worth setting for a second reason: `PLAY_PAUSE` otherwise sends the
keyboard's media key, and macOS answers a media key with nothing playing by
**opening Music** — so the first gesture of a demo launches the wrong app. With
a target named, play/pause goes to that app instead: a scriptable player like
Spotify is told directly, and a browser is sent the keyboard's own play/pause
key, which reaches whatever is actually playing.

That key is sent only when something genuinely is playing, or when SARV was
what paused it — CoreAudio is asked. It is a message to the system rather than
to a player, and the system answers one with nothing playing by opening Music.
With nothing to pause, PLAY_PAUSE says so instead.

`browser_play_key` can still be set to `"k"` or `"space"` to force a site
shortcut, but a site shortcut lands wherever the keyboard focus is — a search
box gets a literal `k`.

macOS matches the app name; Windows matches any part of the window title (so
`"YouTube"` works). The keys are then delivered to that app whether or not it
has focus — no window switching, and the simulator stops counting down. If the
app isn't running, seeking quietly falls back to the focused window.

Without it, seek only reaches the focused window, so either click your video
first or use the countdown:

```bash
python main.py --command FORWARD --delay 5
```

For the real app, setting `seek_target_app` is the robust choice: it keeps
seeking working even if your camera preview window takes focus.

### If seeking works on one site and not another

It is the site, not the browser. Seeking has no system-wide key, so it is a
keyboard shortcut sent to the page — and pages do not agree on which one.
Arrow keys move five seconds on YouTube; YouTube Music ignores them and uses
`j` and `l` for ten seconds each.

```json
{ "browser_seek_keys": "jl" }
```

`"arrows"` is the default. A key that does not seek on a given site is rarely
idle there — it usually does something else — which is why this is a setting
rather than something to guess at.

For music, rewind and forward usually mean the previous and next track rather
than a jump inside one. That is a different setting, and it needs no shortcut
at all:

```json
{ "seek_mode": "track" }
```

## If 200 ms is too slow for your frame loop

`VOLUME_UP`/`VOLUME_DOWN` cost ~200 ms because macOS makes us shell out to
`osascript`. At 30 fps that's six dropped frames. If the preview stutters, use
the non-blocking path instead:

```python
engine = ControlEngine(on_result=lambda r: overlay.show(r.detail))
engine.submit(Command.VOLUME_UP)   # returns immediately
...
engine.close()                     # on shutdown
```

Commands run one at a time in submission order on a worker thread. If they pile
up faster than they can run (more than 4 waiting), the newest are dropped and a
warning is logged — a stale volume nudge is worth less than a responsive app.

## The result object

```python
result.command      # "VOLUME_UP"
result.status       # "OK" | "NOOP" | "THROTTLED" | "UNSUPPORTED" | "ERROR"
result.ok           # True for OK and NOOP
result.detail       # "volume 60% -> 65%"
result.error        # None, or the reason it failed
result.duration_ms
result.timestamp    # UTC ISO-8601
```

Useful for the on-screen overlay: draw `result.detail` when `result.ok`, and
`result.error` in red when it isn't. Ignore `THROTTLED` in the UI — it's normal.

## Setup on your machine

**Windows** — nothing extra to install; the backend uses only `ctypes` and
PowerShell, both built in.

```powershell
pip install -r requirements.txt
python main.py --check
```

**macOS** — needs pyobjc for the media keys. (Do *not* run this on Windows;
pyobjc is macOS-only and will fail to build.)

```bash
pip install -r requirements.txt
pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa
python main.py --check
```

`--check` tells you what works on your laptop. On macOS, if it warns about
Accessibility, grant it in System Settings → Privacy & Security → Accessibility
for whatever launches SARV (Terminal / iTerm / VS Code); volume and brightness
work without it, play/pause and seeking don't. Windows needs no such permission.

Then run `python main.py --selftest` — it fires all seven commands and puts your
machine back the way it was.

## Configuration

Defaults live in `control/config.py`. To override, drop a `sarv_config.json` next
to `main.py`:

```json
{
  "volume_step": 10,
  "brightness_step": 5,
  "seek_seconds": 10,
  "seek_step_seconds": 5,
  "target_app": "Google Chrome",
  "browser_play_key": "media",
  "browser_seek_keys": "arrows",
  "cooldown_seconds": 0.6,
  "dry_run": false
}
```

Or in code: `ControlEngine(config=ControlConfig(volume_step=10))`.

`volume_step` is 10 rather than the 5 a keyboard uses. Tapping a volume key
again is nothing; making a gesture again means raising your hand, moving it,
and bringing it back before the next one counts — so the step is bigger
because asking is more expensive.

`dry_run: true` logs every command without touching the machine — use it while
tuning gesture recognition so a shaky hand doesn't blind you with a brightness
change.

## Testing without a camera

```bash
python main.py --check       # what works on this machine
python main.py --selftest    # run all seven, then restore the machine
python main.py --simulate    # keyboard: U D P L R B N
python main.py --tune        # camera, gesture numbers shown, nothing done
python main.py --hotkeys     # ctrl+alt+U/D/P/L/R/B/N from any app
```

`--hotkeys` is demo insurance, not part of the product: it drives the same
seven commands from anywhere on the machine, so if gesture recognition
misbehaves in front of an audience you can fall back to the keyboard with the
video still fullscreen. Run it in a second terminal and minimise it.

Chords need ctrl+alt so a bare `u` still types a u everywhere. Matched chords
are swallowed and never reach the app you are in.

**The vision half does not need this.** Your camera loop reads frames whether or
not SARV has focus, so nothing about gestures depends on capturing keystrokes.

## Logs

- `logs/sarv.log` — human readable.
- `logs/commands.jsonl` — one JSON object per command; load it with
  `pandas.read_json("logs/commands.jsonl", lines=True)` if we want accuracy
  numbers for the demo.
