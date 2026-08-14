from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt


SAMPLE_RATE = 48_000
TEST_SECONDS = 5
TONE_FREQUENCY = 18_000
AMPLITUDE = 0.08
CHANNELS = 1


def make_test_signal(seconds: float) -> np.ndarray:
    samples = int(seconds * SAMPLE_RATE)
    t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE

    fade_samples = int(0.05 * SAMPLE_RATE)
    envelope = np.ones(samples, dtype=np.float32)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    signal = AMPLITUDE * np.sin(2 * np.pi * TONE_FREQUENCY * t)
    return (signal * envelope).astype(np.float32)


def bandpass(signal: np.ndarray) -> np.ndarray:
    low = 16_000 / (SAMPLE_RATE / 2)
    high = 20_000 / (SAMPLE_RATE / 2)
    sos = butter(6, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def main() -> None:
    print("Available audio devices:\n")
    print(sd.query_devices())

    print("\nDefault device:")
    print(sd.default.device)

    input("\nPress Enter to begin. Keep your hand away during the first test...")

    signal = make_test_signal(TEST_SECONDS)

    print("Recording baseline. Keep still...")
    baseline = sd.playrec(
        signal,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocking=True,
    ).flatten()

    time.sleep(1)

    input("Now place your hand approximately 1 metre away and move it slowly left/right. Press Enter...")

    print("Recording hand movement...")
    movement = sd.playrec(
        signal,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocking=True,
    ).flatten()

    baseline_filtered = bandpass(baseline)
    movement_filtered = bandpass(movement)

    baseline_rms = float(np.sqrt(np.mean(baseline_filtered ** 2)))
    movement_rms = float(np.sqrt(np.mean(movement_filtered ** 2)))

    print(f"\nBaseline filtered RMS: {baseline_rms:.8f}")
    print(f"Movement filtered RMS: {movement_rms:.8f}")

    output_dir = Path("acoustic_test_output")
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "baseline.npy", baseline_filtered)
    np.save(output_dir / "movement.npy", movement_filtered)

    plt.figure(figsize=(12, 5))
    plt.plot(baseline_filtered, alpha=0.7, label="Baseline")
    plt.plot(movement_filtered, alpha=0.7, label="Hand movement")
    plt.title("SARV acoustic test signal")
    plt.xlabel("Audio samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150)
    plt.show()

    print(f"\nSaved temporary analysis files in: {output_dir.resolve()}")
    print("This first result only checks whether the acoustic signal changes.")
    print("It is not yet a gesture classifier.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nAudio test failed: {exc}")
        print("\nCheck microphone permission, speaker selection, and audio device settings.")
        sys.exit(1)