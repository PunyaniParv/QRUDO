#!/bin/bash
# Double-click me to start QRUDO on a Mac.
#
# main.py builds its own environment on a new device, so all this has
# to do is find any Python 3 and get out of the way.
cd "$(dirname "$0")" || exit 1
exec python3 main.py "$@"
