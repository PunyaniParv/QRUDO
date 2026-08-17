# QRUDO Control Engine — interface for the Vision Engine

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
| `TARGET_NEXT` / `TARGET_PREV` | step which app the targeted commands land in | same |
| `NONE` | nothing | nothing |

The two `TARGET_*` commands touch no OS control: they move the aim of the
others.  They cycle through "auto" and every candidate app the machine can
see, and the result's `detail` says where the aim landed (`target ->
Spotify`) so the overlay always shows it.  They arrive by three routes:
pointing at the camera, `ctrl+shift+←/→` from any app (both platforms,
active while the camera runs), and `[` / `]` in the simulator.

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

**Name the app you are controlling.** In `qrudo_config.json`:

```json
{ "target_app": "Google Chrome" }
```

This is worth setting for a second reason: `PLAY_PAUSE` otherwise sends the
keyboard's media key, and macOS answers a media key with nothing playing by
**opening Music** — so the first gesture of a demo launches the wrong app. With
a target named, play/pause goes to that app instead: a scriptable player like
Spotify is told directly, and a browser is sent the keyboard's own play/pause
key, which reaches whatever is actually playing.

That key is sent only when something genuinely is playing, or when QRUDO was
what paused it — CoreAudio is asked. It is a message to the system rather than
to a player, and the system answers one with nothing playing by opening Music.
With nothing to pause, PLAY_PAUSE says so instead.

`browser_play_key` is `"k"` by default — the site shortcut, sent to the browser
and nowhere else. A shortcut is a letter, and a letter lands in the browser's
**front tab**, wherever its keyboard focus is. So before sending it QRUDO asks
two questions through Accessibility: is the focus something that takes typing
(a chat box, a search field — refused with a message), and does the front tab's
title look like the video (`browser_video_titles`, default `"youtube"`). A
front tab that is not the video refuses too, because the letter can go nowhere
else — a `k` once went into a ChatGPT composer exactly that way, with the
browser in the background reporting no focus at all. Watching video on another
site, add it: `"browser_video_titles": "youtube, vimeo"`.

`"media"` sends the keyboard's own play/pause key instead. It is not the
default because it is a message to the system rather than to a player: with
the now-playing role unclaimed, macOS answers it by opening Music, which then
holds that role and takes every media key afterwards.

macOS matches the app name; Windows matches any part of the window title (so
`"YouTube"` works). The keys are then delivered to that app whether or not it
has focus — no window switching, and the simulator stops counting down. If the
app isn't running, seeking quietly falls back to the focused window.

**Or let it choose.** Leave `target_app` unset (or set it to `"auto"`) and a
resolver keeps it pointed at the right app, refreshed every couple of
seconds: the focused app wins if it is a candidate, then a player that says
it is playing (Spotify and Music can be asked on macOS), then whatever the
config prefers. Naming an app pins it and disables all guessing — the old
contract, one config line away. Point at the camera or press
`ctrl+shift+←/→` to step the target by hand; the switch shows on screen
before any command follows it.

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
result.source       # "gesture" | "hotkey" | "simulator" | "selftest" | "cli"
```

`source` is set by whoever calls `execute()`/`submit()` — pass
`source="gesture"` from the camera loop. It changes nothing at runtime; it
travels into the event log so `--report` can judge the camera by the
camera's own commands alone.

Useful for the on-screen overlay: draw `result.detail` when `result.ok`, and
`result.error` in red when it isn't. Ignore `THROTTLED` in the UI — it's normal.

## Setup on your machine

None, usually.  `python3 main.py` (or `py main.py` on Windows) notices a
machine that has never seen QRUDO, builds `.venv/` beside `main.py`, and
installs requirements.txt into it -- once per device, see `bootstrap.py`.
The pyobjc frameworks ride along on macOS via platform markers in
requirements.txt, so one file installs both halves on both platforms.

Manual setup still works, for anyone who wants their own environment --
bootstrap leaves any interpreter that can already import everything
alone:

```bash
pip install -r requirements.txt
python main.py --check
```

`--check` tells you what works on your laptop. On macOS, if it warns about
Accessibility, grant it in System Settings → Privacy & Security → Accessibility
for whatever launches QRUDO (Terminal / iTerm / VS Code); volume and brightness
work without it, play/pause and seeking don't. Windows needs no such permission.

Then run `python main.py --selftest` — it fires all seven commands and puts your
machine back the way it was.

## Configuration

Defaults live in `control/config.py`. To override, drop a `qrudo_config.json` next
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
not QRUDO has focus, so nothing about gestures depends on capturing keystrokes.

## Logs

- `logs/qrudo.log` — human readable.
- `logs/commands.jsonl` — one JSON object per command; load it with
  `pandas.read_json("logs/commands.jsonl", lines=True)` if we want accuracy
  numbers for the demo.

## The reliability report

```bash
python main.py --report
```

Reads `commands.jsonl` back and prints the number that decides whether
people keep QRUDO installed: how often a gesture fired that nobody meant.
Nobody files a bug for a wrong volume nudge — they take it back, and the
taking-back is in the log as a command followed within 3 s by its
opposite (PLAY_PAUSE counts as its own opposite: a toggle nobody meant
gets toggled straight back). Each pair is a suspected misfire; the report
leads with the rate, then per-command counts.

Two rules keep the number honest: only commands that arrived by camera
are judged (`source` tells the routes apart — hotkeys, the simulator,
the selftest and `--command` are deliberate by construction), and a
repeat of the same command reads as leaning on the gesture on purpose,
so two notches up and one down blames only the second notch.

The heuristic errs both ways — a genuine up-then-down counts, an
un-undone misfire is missed — but it errs the same way on every device
and every day, which is what makes the trend worth watching before
launch.
