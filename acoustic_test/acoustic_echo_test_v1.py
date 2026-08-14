import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate


# ============================================================
# SARV ACOUSTIC ECHO / TIME-OF-FLIGHT TEST v1
# ============================================================

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

SAMPLE_RATE = 44100
INPUT_CHANNELS = 4

# Acoustic signal
F_START = 7500
F_END = 8500

CHIRP_DURATION = 0.020
AMPLITUDE = 0.05

# Record before/after transmission
PRE_ROLL = 0.050
POST_ROLL = 0.080

# Distances to investigate
DISTANCES_CM = [30, 60, 100]


# ============================================================
# Create transmitted chirp
# ============================================================

def create_chirp():

    samples = int(
        CHIRP_DURATION * SAMPLE_RATE
    )

    t = np.arange(samples) / SAMPLE_RATE

    signal = chirp(
        t,
        f0=F_START,
        f1=F_END,
        t1=CHIRP_DURATION,
        method="linear"
    )

    # Smooth edges
    window = np.hanning(
        samples
    )

    signal *= window
    signal *= AMPLITUDE

    return signal.astype(
        np.float32
    )


# ============================================================
# Record one measurement
# ============================================================

def record_measurement():

    transmitted = create_chirp()

    pre_samples = int(
        PRE_ROLL * SAMPLE_RATE
    )

    post_samples = int(
        POST_ROLL * SAMPLE_RATE
    )

    output_signal = np.concatenate([
        np.zeros(
            pre_samples,
            dtype=np.float32
        ),

        transmitted,

        np.zeros(
            post_samples,
            dtype=np.float32
        )
    ])

    output = output_signal.reshape(
        -1,
        1
    )

    print(
        "Recording..."
    )

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

    print(
        "Recording complete."
    )

    return recording, transmitted, pre_samples


# ============================================================
# Matched filtering
# ============================================================

def matched_filter(
    recording,
    transmitted
):

    results = []

    for channel in range(
        INPUT_CHANNELS
    ):

        signal = recording[
            :,
            channel
        ]

        correlation = correlate(
            signal,
            transmitted,
            mode="full",
            method="fft"
        )

        results.append(
            correlation
        )

    return np.array(
        results
    )


# ============================================================
# Find peaks around expected delays
# ============================================================

def analyze_echo(
    correlation,
    distance_cm,
    pre_samples
):

    # Expected round-trip delay:
    #
    # distance × 2 / speed of sound
    #
    speed_of_sound = 343.0

    distance_m = (
        distance_cm / 100.0
    )

    expected_delay = (
        2.0
        * distance_m
        / speed_of_sound
    )

    expected_samples = int(
        expected_delay
        * SAMPLE_RATE
    )

    # Correlation lag corresponding
    # to the transmitted chirp.
    #
    # Because correlate(mode="full")
    # produces lags from -(M-1) to N-1,
    # the relevant lag is shifted by
    # len(transmitted)-1.
    #
    # We examine a window around the
    # expected echo delay.

    return (
        expected_delay,
        expected_samples
    )


# ============================================================
# Normalize correlation
# ============================================================

def normalize_correlation(
    correlation
):

    normalized = []

    for channel in correlation:

        maximum = np.max(
            np.abs(channel)
        )

        if maximum == 0:

            normalized.append(
                channel
            )

        else:

            normalized.append(
                channel / maximum
            )

    return np.array(
        normalized
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print(
        "SARV ACOUSTIC ECHO / TIME-OF-FLIGHT TEST v1"
    )
    print("=" * 60)

    print()
    print(
        "Input device :",
        INPUT_DEVICE
    )

    print(
        "Output device:",
        OUTPUT_DEVICE
    )

    print(
        "Sample rate  :",
        SAMPLE_RATE
    )

    print(
        f"Chirp        : "
        f"{F_START}–{F_END} Hz"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "OnePlus neckband should be DISCONNECTED."
    )

    print(
        "Fan may remain ON."
    )

    print(
        "Do not change the speaker volume."
    )

    print()

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print("=" * 60)
    print("BASELINE — NO HAND")
    print("=" * 60)

    print()
    print(
        "Keep your hand away from the laptop."
    )

    input(
        "Press ENTER to record baseline..."
    )

    baseline_recording, transmitted, pre_samples = (
        record_measurement()
    )

    baseline_correlation = matched_filter(
        baseline_recording,
        transmitted
    )

    # --------------------------------------------------------
    # HAND TESTS
    # --------------------------------------------------------

    hand_results = {}

    for distance in DISTANCES_CM:

        print()
        print("=" * 60)
        print(
            f"HAND TEST — {distance} cm"
        )
        print("=" * 60)

        print()

        print(
            f"Place your hand approximately "
            f"{distance} cm in front of the laptop."
        )

        print(
            "Keep your hand still."
        )

        input(
            "Press ENTER to record..."
        )

        recording, _, _ = record_measurement()

        correlation = matched_filter(
            recording,
            transmitted
        )

        hand_results[
            distance
        ] = correlation

    # --------------------------------------------------------
    # Expected delays
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EXPECTED ECHO DELAYS")
    print("=" * 60)

    for distance in DISTANCES_CM:

        expected_delay, expected_samples = (
            analyze_echo(
                baseline_correlation,
                distance,
                pre_samples
            )
        )

        print(
            f"{distance:3d} cm:"
            f" {expected_delay * 1000:.2f} ms"
            f" ≈ {expected_samples} samples"
        )

    # --------------------------------------------------------
    # Plot correlation around expected region
    # --------------------------------------------------------

    # We need the lag axis.
    correlation_length = (
        baseline_correlation.shape[1]
    )

    lags = np.arange(
        correlation_length
    )

    # Convert correlation index
    # into approximate time relative
    # to the transmitted signal.
    #
    # correlate(full) starts at
    # -(chirp_length - 1).

    chirp_samples = len(
        transmitted
    )

    lag_samples = (
        lags
        - (chirp_samples - 1)
    )

    lag_time_ms = (
        lag_samples
        / SAMPLE_RATE
        * 1000
    )

    # Only show a useful region.
    plot_min_ms = -2
    plot_max_ms = 15

    mask = (
        (lag_time_ms >= plot_min_ms)
        &
        (lag_time_ms <= plot_max_ms)
    )

    # --------------------------------------------------------
    # Plot baseline
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    for channel in [0, 1]:

        correlation = (
            baseline_correlation[
                channel
            ]
        )

        plt.plot(
            lag_time_ms[mask],
            np.abs(
                correlation[mask]
            ),
            label=f"Baseline CH{channel + 1}"
        )

    plt.xlabel(
        "Relative delay (ms)"
    )

    plt.ylabel(
        "Correlation magnitude"
    )

    plt.title(
        "SARV Baseline Acoustic Correlation"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # Plot each distance
    # --------------------------------------------------------

    for distance in DISTANCES_CM:

        plt.figure(
            figsize=(12, 6)
        )

        correlation = hand_results[
            distance
        ]

        for channel in [0, 1]:

            plt.plot(
                lag_time_ms[mask],
                np.abs(
                    correlation[
                        channel
                    ][mask]
                ),
                label=f"CH{channel + 1}"
            )

        expected_delay, _ = analyze_echo(
            correlation,
            distance,
            pre_samples
        )

        plt.axvline(
            expected_delay * 1000,
            linestyle="--",
            label=(
                f"Expected echo "
                f"≈ {expected_delay * 1000:.2f} ms"
            )
        )

        plt.xlabel(
            "Relative delay (ms)"
        )

        plt.ylabel(
            "Correlation magnitude"
        )

        plt.title(
            f"SARV Acoustic Echo — "
            f"{distance} cm"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    print()
    print(
        "Look at the plots for peaks near:"
    )

    print(
        "30 cm  → ~1.75 ms"
    )

    print(
        "60 cm  → ~3.50 ms"
    )

    print(
        "100 cm → ~5.83 ms"
    )

    print()

    print(
        "Send me the terminal output and"
    )

    print(
        "screenshots of the plots."
    )

    plt.show()


if __name__ == "__main__":
    main()