# -*- mode: python ; coding: utf-8 -*-
"""How QRUDO becomes an app.

One spec for both platforms, driven by packaging/build_mac.sh and
packaging/build_windows.bat from the repository root, so macOS and
Windows cannot drift apart.  The bundle carries everything -- Python,
MediaPipe, OpenCV, the hand model -- which is what makes the install
"drag to Applications" instead of a ritual, and what makes the
permission prompts say QRUDO instead of Terminal.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all

# Relative paths in a spec resolve against the spec, which lives in
# packaging/ -- everything here is said from the repository root.
ROOT = os.path.dirname(SPECPATH) if os.path.basename(SPECPATH) == "packaging" \
    else SPECPATH

# The hand model, where hand_tracker expects it: models/ beside the code.
datas = [(os.path.join(ROOT, "models", "hand_landmarker.task"), "models")]
binaries = []
hidden = []

# MediaPipe finds hands through graph files it loads at runtime; they
# are data, not imports, so nothing static ever sees them and a bare
# build leaves them behind.
for package in ("mediapipe",):
    collected_datas, collected_binaries, collected_hidden = collect_all(package)
    datas += collected_datas
    binaries += collected_binaries
    hidden += collected_hidden

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    datas=datas,
    binaries=binaries,
    hiddenimports=hidden,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="QRUDO",
    # A product opens its camera window, not a terminal.  The CLI modes
    # still answer when the binary is run from a shell on macOS; on
    # Windows a windowed build is silent, so --report and --simulate
    # belong to the repository way of running there.
    console=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name="QRUDO")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="QRUDO.app",
        bundle_identifier="com.qrudo.app",
        info_plist={
            # This sentence is the camera permission dialog.
            "NSCameraUsageDescription":
                "QRUDO watches your hand through the camera so your "
                "gestures can control this computer.",
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.utilities",
        },
    )
