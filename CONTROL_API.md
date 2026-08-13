# SARV Control Engine — interface for the Vision Engine

This is everything the vision side needs to know. You never import `cv2` or
`mediapipe` into the control layer, and the control layer never imports them
into you.

## The contract

```python
from sarv import Command, ControlEngine

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

| Command | Effect on macOS |
|---|---|
| `VOLUME_UP` | +5 % (unmutes if muted) |
| `VOLUME_DOWN` | −5 % |
| `PLAY_PAUSE` | play/pause media key — works in any app |
| `REWIND` | seek back ~10 s |
| `FORWARD` | seek forward ~10 s |
| `BRIGHTNESS_UP` | +8 % on the built-in display |
| `BRIGHTNESS_DOWN` | −8 % |
| `NONE` | nothing |

Command names are also plain strings — `Command.VOLUME_UP == "VOLUME_UP"` — so
they survive JSON, sockets, or a log file if we later split the two engines
into separate processes.

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

```bash
pip install -r requirements.txt
pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa
python main.py --check
```

`--check` tells you what works on your laptop. If it warns about Accessibility,
grant it in System Settings → Privacy & Security → Accessibility for whatever
launches SARV (Terminal / iTerm / VS Code). Volume and brightness work without
it; play/pause and seeking don't.

## Configuration

Defaults live in `sarv/config.py`. To override, drop a `sarv_config.json` next
to `main.py`:

```json
{
  "volume_step": 10,
  "brightness_step": 5,
  "seek_seconds": 10,
  "seek_step_seconds": 5,
  "cooldown_seconds": 0.6,
  "dry_run": false
}
```

Or in code: `ControlEngine(config=ControlConfig(volume_step=10))`.

`dry_run: true` logs every command without touching the machine — use it while
tuning gesture recognition so a shaky hand doesn't blind you with a brightness
change.

## Testing without a camera

```bash
python main.py --check       # what works on this machine
python main.py --selftest    # run all seven, then restore the machine
python main.py --simulate    # keyboard: U D P L R B N
```

## Logs

- `logs/sarv.log` — human readable.
- `logs/commands.jsonl` — one JSON object per command; load it with
  `pandas.read_json("logs/commands.jsonl", lines=True)` if we want accuracy
  numbers for the demo.
