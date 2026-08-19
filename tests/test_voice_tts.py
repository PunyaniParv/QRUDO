"""Tests for the offline TTS layer (voice/tts.py).

These tests never produce sound: ``subprocess.run`` is faked, so no
PowerShell process is spawned and nothing plays through a speaker.
"""

from __future__ import annotations

import base64
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import tts


class SpeakEmptyCase(unittest.TestCase):
    def test_empty_text_is_a_noop(self):
        with mock.patch.object(tts.subprocess, "run") as run:
            tts.speak("")
        run.assert_not_called()


class BackendAvailabilityCase(unittest.TestCase):
    def test_non_windows_raises_clear_error(self):
        with mock.patch.object(tts.os, "name", "posix"):
            with self.assertRaises(tts.TTSError) as ctx:
                tts.speak("Hey, how are you?")
        self.assertIn("Windows", str(ctx.exception))

    def test_missing_powershell_raises_clear_error(self):
        with mock.patch.object(tts.os, "name", "nt"), mock.patch.object(
            tts.shutil, "which", return_value=None
        ):
            with self.assertRaises(tts.TTSError) as ctx:
                tts.speak("Hey, how are you?")
        self.assertIn("powershell", str(ctx.exception))


@unittest.skipUnless(sys.platform == "win32",
                     "PowerShell TTS is Windows-only; speak() refuses "
                     "elsewhere by design, so there is nothing to invoke")
class PowerShellInvocationCase(unittest.TestCase):
    def _fake_run(self):
        run = mock.patch.object(tts.subprocess, "run").start()
        run.return_value = subprocess.CompletedProcess([], 0)
        self.addCleanup(mock.patch.stopall)
        return run

    def _script(self, text):
        run = self._fake_run()
        tts.speak(text)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "powershell")
        self.assertIn("-NoProfile", cmd)
        return cmd[cmd.index("-Command") + 1]

    def test_speak_invokes_powershell_with_base64_text(self):
        run = self._fake_run()
        tts.speak("Hey, how are you?")
        args, kwargs = run.call_args
        script = args[0][args[0].index("-Command") + 1]
        expected_b64 = base64.b64encode("Hey, how are you?".encode()).decode()
        self.assertIn(expected_b64, script)
        self.assertIn("System.Speech.Synthesis.SpeechSynthesizer", script)
        self.assertEqual(kwargs["check"], True)
        self.assertEqual(kwargs["timeout"], tts._PROCESS_TIMEOUT_SECONDS)

    def test_special_characters_are_base64_escaped(self):
        text = 'It"s $100 "aloud", ok?\nnext line'
        script = self._script(text)
        expected_b64 = base64.b64encode(text.encode()).decode()
        self.assertIn(expected_b64, script)
        for chunk in ('$100', '"aloud"', "\n"):
            self.assertNotIn(chunk, script)

    def test_called_process_error_becomes_tts_error(self):
        run = mock.patch.object(tts.subprocess, "run").start()
        run.side_effect = subprocess.CalledProcessError(1, ["powershell"])
        self.addCleanup(mock.patch.stopall)
        with self.assertRaises(tts.TTSError):
            tts.speak("Hey, how are you?")

    def test_timeout_becomes_tts_error(self):
        run = mock.patch.object(tts.subprocess, "run").start()
        run.side_effect = subprocess.TimeoutExpired(["powershell"], 60)
        self.addCleanup(mock.patch.stopall)
        with self.assertRaises(tts.TTSError):
            tts.speak("Hey, how are you?")


if __name__ == "__main__":
    unittest.main(verbosity=2)