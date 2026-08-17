# QRUDO

Control your computer with hand gestures.

## Run it

```bash
git clone <this repo>
cd QRUDO
python3 main.py        # macOS -- or double-click qrudo.command
py main.py             # Windows -- or double-click qrudo.bat
```

That is the whole install.  The first start on a new device builds
QRUDO's own Python environment in `.venv/` and fills it from
requirements.txt -- a few hundred MB, once.  Every start after that is
immediate.  No pip commands, no venv to remember, nothing to activate:
any Python 3 on the machine is enough to light the fuse.

macOS asks for Camera and Accessibility permission on first run; grant
both, for whatever launched QRUDO (Terminal, iTerm, VS Code).  `python3
main.py --check` says what this machine can control and what is still
missing.

The first run also spends a minute measuring your hand, so the gestures
suit your camera and where you stand.

After you have used it a while, `python3 main.py --report` reads the
command log back and reports how often a gesture fired that you then
took straight back -- the misfire rate, measured from real use.

## More

- `CONTROL_API.md` -- the contract between the vision half and the
  control half, the seven commands, configuration, and every no-camera
  test mode (`--check`, `--selftest`, `--simulate`, `--hotkeys`).
