"""Make QRUDO start on a machine that has never seen it.

On a fresh device `python main.py` used to die with `No module named
'cv2'`, and the cure -- create a venv, install requirements.txt into it,
remember to launch through it forever after -- was a ritual someone had
to know.  This module is that ritual, performed automatically: main.py
calls ensure() before anything imports the heavy dependencies.

The promises:

- If the interpreter that launched us can already import everything,
  nothing happens.  Anyone managing their own environment keeps it.
- Otherwise QRUDO hands off to its own venv at .venv/, building it and
  filling it from requirements.txt first if it has to.  One slow first
  run per device; every start after that is immediate.
- requirements.txt is fingerprinted into the venv, so pulling a version
  with new dependencies re-installs on the next start instead of
  crashing halfway into the camera loop.
- The handoff happens at most once per start (QRUDO_BOOTSTRAPPED guards
  it), so a broken install explains itself rather than looping.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / "requirements.sha256"
GUARD = "QRUDO_BOOTSTRAPPED"

# Checked with find_spec, never imported: asking costs milliseconds,
# importing mediapipe costs seconds, and this runs on every start.
MODULES = ["cv2", "mediapipe", "numpy"]
if sys.platform == "darwin":
    # Media keys, seeking and hotkeys.  Volume and brightness would
    # manage without it, but one environment that can do everything
    # beats remembering which mode needs which half.
    MODULES.append("Quartz")


def ensure():
    """Return with the dependencies importable, or hand off to .venv."""
    # A packaged app carries everything inside itself; there is nothing
    # to install and no interpreter to hand off to.  A missing import
    # there is a packaging bug, and the loud ImportError names it.
    if getattr(sys, "frozen", False):
        return

    if not missing():
        return

    if os.environ.get(GUARD):
        _explain_and_exit()

    py = venv_python()

    if not py.exists():
        print(_first_run_notice())
        _run([sys.executable, "-m", "venv", str(VENV)],
             "could not create the environment")

    if _stamp_stale():
        _run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
             "could not install the requirements")
        STAMP.write_text(fingerprint())

    _handoff(py)


def missing():
    return [name for name in MODULES
            if importlib.util.find_spec(name) is None]


def venv_python():
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def fingerprint():
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def _stamp_stale():
    try:
        return STAMP.read_text().strip() != fingerprint()
    except OSError:
        return True


def _run(cmd, what):
    # Output is left on the terminal on purpose: a first install takes
    # minutes, and pip's progress is the only sign it is alive.
    code = subprocess.run(cmd).returncode
    if code != 0:
        print(f"\n  ! {what} (exit {code}); the output above says why."
              f"\n    After fixing it, delete {VENV}"
              f"\n    and start QRUDO again to rebuild.", file=sys.stderr)
        raise SystemExit(1)


def _handoff(py):
    env = dict(os.environ)
    env[GUARD] = "1"
    argv = [str(py), str(ROOT / "main.py"), *sys.argv[1:]]

    if os.name == "nt":
        # exec on Windows does not replace the process -- the parent
        # returns and the console prompt lands on top of QRUDO's output.
        # A child process keeps the terminal honest.
        raise SystemExit(subprocess.run(argv, env=env).returncode)

    os.execve(str(py), argv, env)


def _explain_and_exit():
    names = ", ".join(missing())
    print(f"  ! the environment at {VENV}"
          f"\n    is missing {names} even after installing"
          f"\n    requirements.txt.  Delete that folder and start QRUDO"
          f"\n    again to rebuild it; if this message comes back, the"
          f"\n    pip output above it holds the reason.", file=sys.stderr)
    raise SystemExit(1)


def _first_run_notice():
    return "\n".join([
        "",
        "  First run on this device: QRUDO is building its own Python",
        "  environment in .venv/ and installing what it needs -- a few",
        "  hundred MB, a few minutes on a normal connection.",
        "",
        "  This happens once.  Every start after this one is immediate.",
        "",
    ])
