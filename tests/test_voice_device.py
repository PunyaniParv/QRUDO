"""Tests for QRUDO microphone selection (voice/device.py).

These are pure tests: they replace ``sounddevice`` with a fake module so no
microphone hardware, Bluetooth device, or USB device is required. They prove
the device-selection rules that the wake-word listener and audio capture rely
on:

  * ``microphone_device = None``  -> the *current* OS/sounddevice default
                                     input device (``sd.default.device[0]``),
                                     never a hard-coded or stale index/name.
  * explicit ``int``              -> that exact device index.
  * explicit ``str``              -> case-insensitive device-name substring.
  * configured device unavailable -> fall back to the current OS default.
  * a Bluetooth-style input device exposed by sounddevice is selectable
    exactly like any other input device.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import device  # noqa: E402


def _mk_device(index, name, hostapi, in_ch=2, out_ch=0, rate=44100.0):
    return {
        "name": name,
        "index": index,
        "hostapi": hostapi,
        "max_input_channels": in_ch,
        "max_output_channels": out_ch,
        "default_samplerate": rate,
    }


def _build_fake(default_input):
    """A representative Windows-like sounddevice backend.

    Built-in, Bluetooth and OS-default devices are present, and the built-in
    array appears on two host APIs (MME + DirectSound) so a bare name query can
    also be ambiguous, just like the real backend.
    """
    hostapis = [
        {"name": "MME", "default_input_device": 1},
        {"name": "Windows DirectSound", "default_input_device": 4},
        {"name": "Windows WASAPI", "default_input_device": 6},
    ]
    devices = [
        _mk_device(0, "Microphone (Built-in Array)", 0),
        _mk_device(1, "Microphone Array (Intel Smart)", 0),
        _mk_device(2, "Headset (OnePlus Bullets Bluetooth)", 0),
        _mk_device(3, "Speakers (Realtek)", 0, in_ch=0, out_ch=2),
        _mk_device(4, "Microphone Array (Intel Smart)", 1),
        _mk_device(5, "Headset (WH-CH520)", 1, in_ch=1),
        _mk_device(6, "Microphone Array (Intel Smart)", 2),
    ]
    return _FakeSD(hostapis, devices, default_input)


class _FakeSD:
    """Minimal stand-in for the ``sounddevice`` module (device-only surface)."""

    def __init__(self, hostapis, devices, default_input):
        self._hostapis = hostapis
        self._devices = devices
        self.default = SimpleNamespace(device=[default_input, None])

    def query_hostapis(self):
        return list(self._hostapis)

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            return list(self._devices)
        if device is None and kind == "input":
            return self._devices[self.default.device[0]]
        candidates = []
        hostapi_names = [ha["name"].lower() for ha in self._hostapis]
        for info in self._devices:
            if kind and info["max_input_channels"] < 1:
                continue
            candidates.append(
                (info["index"], info["name"], hostapi_names[info["hostapi"]])
            )
        if isinstance(device, int):
            for idx, _, _ in candidates:
                if idx == device:
                    return self._devices[device]
            raise IndexError(f"device {device} is not an input device / not found")
        # String: case-insensitive, space-separated substrings matched in
        # order, exactly like sounddevice's `query_devices('<name>')`.
        query_string = device.lower()
        substrings = query_string.split()
        matches = []
        for idx, dev_name, ha_name in candidates:
            full = (dev_name + ", " + ha_name).lower()
            pos = 0
            for sub in substrings:
                pos = full.find(sub, pos)
                if pos < 0:
                    break
                pos += len(sub)
            else:
                matches.append(idx)
        if not matches:
            raise ValueError(f"no input device matching {device!r}")
        if len(matches) > 1:
            raise ValueError(f"multiple input devices matching {device!r}: {matches}")
        return self._devices[matches[0]]
class DefaultDeviceCase(unittest.TestCase):
    """microphone_device = None must track the current OS default, live."""

    def test_default_follows_sd_default_device_zero(self):
        fake = _build_fake(default_input=1)
        with mock.patch.object(device, "sd", fake):
            spec, name = device._resolve(None)
        self.assertEqual(spec, 1)
        self.assertEqual(spec, fake.default.device[0])
        self.assertEqual(name, fake._devices[1]["name"])

    def test_default_follows_a_changed_os_default_not_a_stale_device(self):
        # The OS default input changes (device unplugged / selection changed);
        # QRUDO must follow it and NOT keep a previously used device.
        fake = _build_fake(default_input=1)
        with mock.patch.object(device, "sd", fake):
            self.assertEqual(device._resolve(None)[0], 1)
        # Now the OS default moves to the Bluetooth headset's index.
        fake.default.device[0] = 5
        with mock.patch.object(device, "sd", fake):
            self.assertEqual(device._resolve(None)[0], 5)
            self.assertEqual(device._resolve(None)[1], "Headset (WH-CH520)")

    def test_no_default_reports_none_cleanly(self):
        fake = _build_fake(default_input=-1)
        with mock.patch.object(device, "sd", fake):
            self.assertEqual(
                device._default_input(), (None, "OS default (none available)")
            )


class ExplicitDeviceCase(unittest.TestCase):
    def test_explicit_integer_index(self):
        fake = _build_fake(default_input=1)
        with mock.patch.object(device, "sd", fake):
            spec, name = device._resolve(2)
        self.assertEqual((spec, name), (2, "Headset (OnePlus Bullets Bluetooth)"))

    def test_explicit_name_substring_case_insensitive(self):
        fake = _build_fake(default_input=1)
        with mock.patch.object(device, "sd", fake):
            spec, name = device._resolve("oneplus")
        self.assertEqual((spec, name), (2, "Headset (OnePlus Bullets Bluetooth)"))
        with mock.patch.object(device, "sd", fake):
            spec2, _ = device._resolve("  ONEplus  ")  # trimmed + uppercased
        self.assertEqual(spec2, 2)


class FallbackCase(unittest.TestCase):
    def test_unavailable_integer_falls_back_to_current_os_default(self):
        fake = _build_fake(default_input=1)
        with mock.patch.object(device, "sd", fake):
            spec, name = device._resolve(99)
        self.assertEqual((spec, name), (1, "Microphone Array (Intel Smart)"))

    def test_unavailable_name_falls_back_to_current_os_default(self):
        fake = _build_fake(default_input=1)
        with mock.patch.object(device, "sd", fake):
            spec, name = device._resolve("Old Cami Cam mic")
        self.assertEqual((spec, name), (1, "Microphone Array (Intel Smart)"))

    def test_fallback_uses_live_default_not_a_saved_device(self):
        fake = _build_fake(default_input=5)
        with mock.patch.object(device, "sd", fake):
            spec, name = device._resolve("does not exist")
        self.assertEqual((spec, name), (5, "Headset (WH-CH520)"))
class BluetoothSupportCase(unittest.TestCase):
    """Bluetooth-style inputs are plain input-capable devices, nothing special."""

    def test_bt_headset_selectable_by_unique_name(self):
        fake = _build_fake(default_input=1)
        with mock.patch.object(device, "sd", fake):
            spec, name = device._resolve("WH-CH520")
        self.assertEqual(spec, 5)
        self.assertEqual(name, "Headset (WH-CH520)")

    def test_bt_headset_usable_as_os_default(self):
        fake = _build_fake(default_input=5)
        with mock.patch.object(device, "sd", fake):
            spec, name = device._resolve(None)
        self.assertEqual((spec, name), (5, "Headset (WH-CH520)"))

    def test_disconnected_bt_falls_back_to_os_default(self):
        # The BT headset is no longer exposed by sounddevice; QRUDO must not
        # crash and must fall back to the current OS default input.
        fake = _build_fake(default_input=1)
        fake._devices = [d for d in fake._devices if d["name"] != "Headset (WH-CH520)"]
        with mock.patch.object(device, "sd", fake):
            spec, name = device._resolve("WH-CH520")  # disconnected / gone
        self.assertEqual((spec, name), (1, "Microphone Array (Intel Smart)"))


class MicrophoneStreamSelectionCase(unittest.TestCase):
    def test_stream_uses_resolved_device_and_logs_exact_selection(self):
        fake = _build_fake(default_input=1)
        with mock.patch.object(device, "sd", fake), \
             mock.patch.object(device.MicrophoneStream, "_open", return_value=None), \
             mock.patch.object(
                 device, "CONFIG", SimpleNamespace(microphone_device=None)
             ):
            with self.assertLogs("sarv.voice.device", level="INFO") as cm:
                stream = device.MicrophoneStream(
                    samplerate=16000, channels=1, dtype="int16", blocksize=1280
                )
                stream.__enter__()
        self.assertEqual(stream.device_spec, 1)
        self.assertEqual(stream.device_name, fake._devices[1]["name"])
        joined = "\n".join(cm.output)
        self.assertIn("audio input selected", joined)
        self.assertIn("index=1", joined)
        self.assertIn("Microphone Array (Intel Smart)", joined)
        self.assertIn("hostapi=MME", joined)
        self.assertIn("channels=1", joined)
        self.assertIn("samplerate=16000", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)