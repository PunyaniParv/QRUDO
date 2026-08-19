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
        "ed378fa4fd3630a3f4afec3d94a621852bd7a5fe2d80f2ab9d0ef869bd2e9593",
    "assets/qrudo.icns":
        "f51f498f9ecd9243aa45306533b68104fc0fcc40890fdd14dddda1ac5cacbdc4",
    "assets/qrudo.ico":
        "680ee2818b0123c49c0683bba8258a14f1874f64c8d71aec5f432c36629ca7c4",
    "packaging/AppIcon.icon/icon.json":
        "14b83a989a38025d7c5ac8f46a5a96643d4f5e770b6e17f2a5ba9eed5fbf2564",
    "packaging/AppIcon.icon/Assets/mark.png":
        "ed378fa4fd3630a3f4afec3d94a621852bd7a5fe2d80f2ab9d0ef869bd2e9593",
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
