"""The face of the product, pinned byte for byte.

The icon took a full day to settle: the gold Q on soft charcoal, as a
layered Icon Composer document for macOS, a legacy icns fallback, a
Windows ico, and the window's own image.  These files ARE the brand
now, on both platforms, and nothing regenerates them as a side effect
-- the generation scripts are gone from the build on purpose.

So any change to these bytes fails here, loudly, on macOS and Windows
CI alike.  A deliberate rebrand is welcome: regenerate the artifacts,
then update the hashes below in the same commit, so the change is a
decision with a diff -- never an accident with a build.
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: sha256 of every identity artifact, recorded 2026-08-19 when the
#: icon was declared final.  logo.png and the layered icon's mark are
#: intentionally the same image -- one face, several carriers.
IDENTITY = {
    "assets/logo.png":
        "c686eb053841fb8de28ce0da008f96bf124c13d2f21663612048ba1310d5ce50",
    "assets/qrudo.icns":
        "38423ed805bb1267102ea5f772e0473df82b35aa19ed2933cdcef1da68d1844d",
    "assets/qrudo.ico":
        "3a40ba274a1500c8796b4e980ddac06fe628108f13d9aca3a35b0a02787fdc4d",
    "packaging/AppIcon.icon/icon.json":
        "14b83a989a38025d7c5ac8f46a5a96643d4f5e770b6e17f2a5ba9eed5fbf2564",
    "packaging/AppIcon.icon/Assets/mark.png":
        "c686eb053841fb8de28ce0da008f96bf124c13d2f21663612048ba1310d5ce50",
}


class TestTheFaceDoesNotDrift(unittest.TestCase):
    def test_every_identity_artifact_is_exactly_itself(self):
        for rel, expected in IDENTITY.items():
            with self.subTest(file=rel):
                path = ROOT / rel

                self.assertTrue(path.exists(),
                                f"{rel} is missing -- the identity "
                                f"artifacts must ship with the repo")

                actual = hashlib.sha256(path.read_bytes()).hexdigest()

                self.assertEqual(actual, expected,
                                 f"{rel} changed.  If this is a "
                                 f"deliberate rebrand, update the hash "
                                 f"in the same commit; if not, restore "
                                 f"the file -- the icon is final.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
