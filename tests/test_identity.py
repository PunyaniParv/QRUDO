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
        "a5f97ea3a23928ef7985fec83c80d472d73d09c6cb07468de39d43ef1da0e05e",
    "assets/qrudo.icns":
        "beda563a0ba7eb4034843edb97a95f6a8f2cbfede79ce5ebe71cb2a6527cc37c",
    "assets/qrudo.ico":
        "458015a51c637dbe304398fc232b3341dff50bf2799d7ab5f2bf58b8c614207d",
    "packaging/AppIcon.icon/icon.json":
        "14b83a989a38025d7c5ac8f46a5a96643d4f5e770b6e17f2a5ba9eed5fbf2564",
    "packaging/AppIcon.icon/Assets/mark.png":
        "a5f97ea3a23928ef7985fec83c80d472d73d09c6cb07468de39d43ef1da0e05e",
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
