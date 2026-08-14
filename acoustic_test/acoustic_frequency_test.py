import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# ============================================================
# SARV Acoustic Frequency Response Test
# ============================================================

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

SAMPLE_RATE = 44100
INPUT_CHANNELS = 4

FREQUENCIES = [
    8000,
    10000,
    12000,
    14000,
    16000,
    18000,
    20000,
]

TONE_DURATION = 1.0
AMPLITUDE = 0.05

PRE_ROLL = 0.25
POST_ROLL = 0.25


def create_tone(frequency):
    """Create a smooth sine tone."""

    samples = int(TONE_DURATION * SAMPLE_RATE)

    t = np.arange(samples) / SAMPLE_RATE

    tone = np.sin(
        2 * np.pi * frequency * t
    )

    # Fade in/out to avoid clicks.
    fade_samples = int(0.05 * SAMPLE_RATE)

    fade = np.ones(samples)

    fade[:fade_samples] = np.linspace(
        0,
        1,
        fade_samples
    )

    fade[-fade_samples:] = np.linspace(
        1,
        0,
        fade_samples
    )

    tone *= fade
    tone *= AMPLITUDE

    return tone.astype(np.float32)


def measure_frequency(frequency):
    """Play one frequency and measure the microphone response."""

    print()
    print(f"Testing {frequency / 1000:.1f} kHz...")

    tone = create_tone(frequency)

    pre_samples = int(
        PRE_ROLL * SAMPLE_RATE
    )

    post_samples = int(
        POST_ROLL * SAMPLE_RATE
    )

    signal = np.concatenate([
        np.zeros(
            pre_samples,
            dtype=np.float32
        ),
        tone,
        np.zeros(
            post_samples,
            dtype=np.float32
        )
    ])

    # One output channel.
    output = signal.reshape(-1, 1)

    print("  Recording...")

    recording = sd.playrec(
        output,
        samplerate=SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        dtype="float32",
        blocking=True
    )

    # Extract only the section where the tone is playing.
    start = pre_samples
    end = start + len(tone)

    active = recording[start:end]

    # --------------------------------------------------------
    # Band-pass filter
    # --------------------------------------------------------

    nyquist = SAMPLE_RATE / 2

    bandwidth = 500

    low = (frequency - bandwidth) / nyquist
    high = (frequency + bandwidth) / nyquist

    # Keep the filter safely inside the valid range.
    low = max(low, 0.001)
    high = min(high, 0.999)

    if low >= high:
        raise ValueError(
            f"Invalid filter range for {frequency} Hz: "
            f"low={low}, high={high}"
        )

    sos = butter(
        4,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        active,
        axis=0
    )

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    channel_rms = np.sqrt(
        np.mean(
            np.square(filtered),
            axis=0
        )
    )

    print(
        "  RMS:",
        " | ".join(
            f"CH{i + 1}={value:.8e}"
            for i, value in enumerate(channel_rms)
        )
    )

    return channel_rms


def main():

    print("=" * 60)
    print("SARV ACOUSTIC FREQUENCY RESPONSE TEST")
    print("=" * 60)

    print()
    print("Input device :", INPUT_DEVICE)
    print("Output device:", OUTPUT_DEVICE)
    print("Sample rate  :", SAMPLE_RATE)
    print()

    print("IMPORTANT:")
    print(
        "Keep your hand away from the laptop."
    )
    print(
        "Keep the room reasonably quiet."
    )
    print(
        "The laptop speakers will emit test tones."
    )
    print()

    input(
        "Press ENTER to start the frequency sweep..."
    )

    results = []

    # --------------------------------------------------------
    # Frequency sweep
    # --------------------------------------------------------

    for frequency in FREQUENCIES:

        rms_values = measure_frequency(
            frequency
        )

        results.append(
            rms_values
        )

        # Short pause between frequencies.
        sd.sleep(300)

    results = np.array(results)

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for i, frequency in enumerate(FREQUENCIES):

        print(
            f"{frequency / 1000:5.1f} kHz : "
            + " | ".join(
                f"CH{ch + 1}={results[i, ch]:.8e}"
                for ch in range(INPUT_CHANNELS)
            )
        )

    # --------------------------------------------------------
    # Average response
    # --------------------------------------------------------

    average_response = np.mean(
        results,
        axis=1
    )

    print()
    print("Average response:")

    for frequency, response in zip(
        FREQUENCIES,
        average_response
    ):

        print(
            f"{frequency / 1000:5.1f} kHz -> "
            f"{response:.8e}"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        np.array(FREQUENCIES) / 1000,
        average_response,
        marker="o"
    )

    plt.xlabel(
        "Frequency (kHz)"
    )

    plt.ylabel(
        "Received filtered RMS"
    )

    plt.title(
        "SARV Laptop Speaker → Microphone Acoustic Response"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()