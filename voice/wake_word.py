"""
Wake word detection using Picovoice Porcupine.

Blocks until "Hey Qrudo" is heard, then returns. Runs fully offline once
the .ppn model is downloaded — no audio is sent anywhere.
"""

import struct

import pvporcupine

from voice.config import CONFIG
from voice.device import MicrophoneStream


class WakeWordListener:
    def __init__(self):
        if not CONFIG.picovoice_access_key:
            raise RuntimeError(
                "PICOVOICE_ACCESS_KEY is not set. Get a free key at "
                "https://console.picovoice.ai and set it as an environment variable."
            )

        self._porcupine = pvporcupine.create(
            access_key=CONFIG.picovoice_access_key,
            keyword_paths=[CONFIG.wake_word_path],
            sensitivities=[CONFIG.wake_word_sensitivity],
        )
        self._frame_length = self._porcupine.frame_length          # samples per frame Porcupine expects
        self._sample_rate = self._porcupine.sample_rate            # should match CONFIG.sample_rate (16000)

    def wait_for_wake_word(self) -> None:
        """Blocks until the wake word is detected. Returns immediately after."""
        with MicrophoneStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._frame_length,
        ) as stream:
            while True:
                frame, _overflowed = stream.read(self._frame_length)
                pcm = struct.unpack_from(
                    "h" * self._frame_length, frame.tobytes()
                )
                keyword_index = self._porcupine.process(pcm)
                if keyword_index >= 0:
                    return

    def close(self) -> None:
        self._porcupine.delete()