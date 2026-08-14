import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# ============================================================
# SARV Acoustic Distance Test v1
# ============================================================

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

SAMPLE_RATE = 44100
INPUT_CHANNELS = 4

CENTER_FREQUENCY = 8000
BANDWIDTH = 1000

CHIRP_DURATION = 0.20
CHIRP_INTERVAL = 0.50
NUMBER_OF_CHIRPS = 8

AMPLITUDE = 0.05

PRE_ROLL = 0.50
POST_ROLL = 0.50

DISTANCES = [
    30,
    60,
    100
]


# ============================================================
# Signal generation
# ============================================================

def create_chirp():

    samples = int(
        CHIRP_DURATION * SAMPLE_RATE
    )

    t = np.arange(samples) / SAMPLE_RATE

    f0 = CENTER_FREQUENCY - BANDWIDTH / 2
    f1 = CENTER_FREQUENCY + BANDWIDTH / 2

    phase = 2 * np.pi * (
        f0 * t
        + ((f1 - f0) / (2 * CHIRP_DURATION))
        * t**2
    )

    signal = np.sin(phase)

    window = np.hanning(len(signal))

    signal *= window
    signal *= AMPLITUDE

    return signal.astype(np.float32)


def create_test_signal():

    chirp = create_chirp()

    period_samples = int(
        CHIRP_INTERVAL * SAMPLE_RATE
    )

    silence_samples = (
        period_samples - len(chirp)
    )

    silence = np.zeros(
        silence_samples,
        dtype=np.float32
    )

    period = np.concatenate([
        chirp,
        silence
    ])

    signal = np.tile(
        period,
        NUMBER_OF_CHIRPS
    )

    return signal.astype(np.float32)


# ============================================================
# Filter
# ============================================================

def bandpass_filter(signal):

    nyquist = SAMPLE_RATE / 2

    low = (
        CENTER_FREQUENCY
        - BANDWIDTH / 2
    ) / nyquist

    high = (
        CENTER_FREQUENCY
        + BANDWIDTH / 2
    ) / nyquist

    sos = butter(
        6,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal,
        axis=0
    )


# ============================================================
# Measurement
# ============================================================

def run_measurement(label):

    print()
    print("=" * 60)
    print(label)
    print("=" * 60)

    signal = create_test_signal()

    pre_samples = int(
        PRE_ROLL * SAMPLE_RATE
    )

    post_samples = int(
        POST_ROLL * SAMPLE_RATE
    )

    playback = np.concatenate([
        np.zeros(
            pre_samples,
            dtype=np.float32
        ),
        signal,
        np.zeros(
            post_samples,
            dtype=np.float32
        )
    ])

    output = playback.reshape(-1, 1)

    print(
        f"Duration: "
        f"{len(playback) / SAMPLE_RATE:.2f} seconds"
    )

    print("Starting in 2 seconds...")

    sd.sleep(2000)

    print("Recording...")

    recording = sd.playrec(
        output,
        samplerate=SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        device=(
            INPUT_DEVICE,
            OUTPUT_DEVICE
        ),
        dtype="float32",
        blocking=True
    )

    print("Recording complete.")

    return recording


# ============================================================
# Analysis
# ============================================================

def calculate_rms(recording):

    filtered = bandpass_filter(
        recording
    )

    rms = np.sqrt(
        np.mean(
            np.square(filtered),
            axis=0
        )
    )

    return rms


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("SARV ACOUSTIC DISTANCE TEST v1")
    print("=" * 60)

    print()
    print("Input device :", INPUT_DEVICE)
    print("Output device:", OUTPUT_DEVICE)
    print("Sample rate  :", SAMPLE_RATE)
    print("Frequency    :", CENTER_FREQUENCY, "Hz")
    print()

    print("Keep the OnePlus neckband disconnected.")
    print("Fan may remain ON.")
    print()

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print("BASELINE")

    print()
    print("Keep your hand away from the laptop.")
    print("Remain still.")

    input(
        "Press ENTER to measure baseline..."
    )

    baseline_recording = run_measurement(
        "BASELINE — NO HAND"
    )

    baseline_rms = calculate_rms(
        baseline_recording
    )

    print()
    print("Baseline RMS:")

    for i, value in enumerate(
        baseline_rms
    ):

        print(
            f"  CH{i + 1}: "
            f"{value:.8e}"
        )

    # --------------------------------------------------------
    # DISTANCE TESTS
    # --------------------------------------------------------

    results = {}

    for distance in DISTANCES:

        print()
        print("#" * 60)
        print(
            f"DISTANCE: {distance} cm"
        )
        print("#" * 60)

        # ----------------------------------------------------
        # STATIONARY
        # ----------------------------------------------------

        print()
        print(
            f"Place your hand approximately "
            f"{distance} cm away."
        )

        print(
            "Keep your hand completely still."
        )

        input(
            "Press ENTER when ready..."
        )

        stationary_recording = run_measurement(
            f"STATIONARY HAND — {distance} cm"
        )

        stationary_rms = calculate_rms(
            stationary_recording
        )

        # ----------------------------------------------------
        # MOVEMENT
        # ----------------------------------------------------

        print()
        print(
            f"Keep your hand approximately "
            f"{distance} cm away."
        )

        print(
            "Slowly move:"
        )

        print(
            "LEFT → RIGHT → LEFT"
        )

        input(
            "Press ENTER when ready..."
        )

        movement_recording = run_measurement(
            f"MOVING HAND — {distance} cm"
        )

        movement_rms = calculate_rms(
            movement_recording
        )

        results[distance] = {
            "stationary": stationary_rms,
            "movement": movement_rms
        }

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for distance in DISTANCES:

        stationary = results[
            distance
        ]["stationary"]

        movement = results[
            distance
        ]["movement"]

        print()
        print(
            f"===== {distance} cm ====="
        )

        for ch in range(
            INPUT_CHANNELS
        ):

            stationary_ratio = (
                stationary[ch]
                / max(
                    baseline_rms[ch],
                    1e-12
                )
            )

            movement_ratio = (
                movement[ch]
                / max(
                    baseline_rms[ch],
                    1e-12
                )
            )

            print(
                f"CH{ch + 1}: "
                f"stationary={stationary[ch]:.8e} "
                f"({stationary_ratio:.2f}x) | "
                f"movement={movement[ch]:.8e} "
                f"({movement_ratio:.2f}x)"
            )

    # --------------------------------------------------------
    # Plot CH1 / CH2
    # --------------------------------------------------------

    distances = np.array(
        DISTANCES,
        dtype=float
    )

    for ch in [0, 1]:

        stationary_ratios = []

        movement_ratios = []

        for distance in DISTANCES:

            stationary = results[
                distance
            ]["stationary"][ch]

            movement = results[
                distance
            ]["movement"][ch]

            stationary_ratios.append(
                stationary
                / max(
                    baseline_rms[ch],
                    1e-12
                )
            )

            movement_ratios.append(
                movement
                / max(
                    baseline_rms[ch],
                    1e-12
                )
            )

        plt.figure(
            figsize=(9, 6)
        )

        plt.plot(
            distances,
            stationary_ratios,
            marker="o",
            label="Stationary hand"
        )

        plt.plot(
            distances,
            movement_ratios,
            marker="o",
            label="Moving hand"
        )

        plt.xlabel(
            "Distance from laptop (cm)"
        )

        plt.ylabel(
            "Signal / baseline"
        )

        plt.title(
            f"SARV Acoustic Response — "
            f"Microphone Channel {ch + 1}"
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()