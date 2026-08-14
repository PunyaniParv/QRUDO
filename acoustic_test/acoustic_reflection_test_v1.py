import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# ============================================================
# SARV Acoustic Reflection Test v1
# ============================================================

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

SAMPLE_RATE = 44100
INPUT_CHANNELS = 4

# Based on the frequency-response test:
CENTER_FREQUENCY = 8000
BANDWIDTH = 1000

# Test signal
CHIRP_DURATION = 0.20
CHIRP_INTERVAL = 0.50
NUMBER_OF_CHIRPS = 8

AMPLITUDE = 0.05

PRE_ROLL = 0.50
POST_ROLL = 0.50


# ============================================================
# Signal generation
# ============================================================

def create_chirp():
    """
    Create a short 7.5-8.5 kHz chirp.
    """

    duration_samples = int(
        CHIRP_DURATION * SAMPLE_RATE
    )

    t = np.arange(
        duration_samples
    ) / SAMPLE_RATE

    # Linear frequency sweep.
    f0 = CENTER_FREQUENCY - BANDWIDTH / 2
    f1 = CENTER_FREQUENCY + BANDWIDTH / 2

    phase = 2 * np.pi * (
        f0 * t
        + ((f1 - f0) / (2 * CHIRP_DURATION))
        * t**2
    )

    signal = np.sin(phase)

    # Smooth start/end.
    window = np.hanning(
        len(signal)
    )

    signal *= window
    signal *= AMPLITUDE

    return signal.astype(
        np.float32
    )


def create_test_signal():
    """
    Create repeated chirps separated by silence.
    """

    chirp_signal = create_chirp()

    period_samples = int(
        CHIRP_INTERVAL * SAMPLE_RATE
    )

    silence_samples = (
        period_samples
        - len(chirp_signal)
    )

    silence = np.zeros(
        silence_samples,
        dtype=np.float32
    )

    one_period = np.concatenate([
        chirp_signal,
        silence
    ])

    signal = np.tile(
        one_period,
        NUMBER_OF_CHIRPS
    )

    return signal.astype(
        np.float32
    )


# ============================================================
# Filtering
# ============================================================

def bandpass_filter(signal):
    """
    Keep the acoustic test band.
    """

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

    test_signal = create_test_signal()

    pre_samples = int(
        PRE_ROLL * SAMPLE_RATE
    )

    post_samples = int(
        POST_ROLL * SAMPLE_RATE
    )

    playback_signal = np.concatenate([
        np.zeros(
            pre_samples,
            dtype=np.float32
        ),
        test_signal,
        np.zeros(
            post_samples,
            dtype=np.float32
        )
    ])

    output = playback_signal.reshape(
        -1,
        1
    )

    total_duration = (
        len(playback_signal)
        / SAMPLE_RATE
    )

    print(
        f"Duration: {total_duration:.2f} seconds"
    )

    print(
        "Starting in 2 seconds..."
    )

    sd.sleep(2000)

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

    return recording


# ============================================================
# Analysis
# ============================================================

def analyze(recording):

    filtered = bandpass_filter(
        recording
    )

    # Overall RMS for each microphone.
    channel_rms = np.sqrt(
        np.mean(
            np.square(filtered),
            axis=0
        )
    )

    return filtered, channel_rms


def calculate_time_rms(filtered):

    # 50 ms windows.
    window_size = int(
        0.05 * SAMPLE_RATE
    )

    hop_size = int(
        0.025 * SAMPLE_RATE
    )

    values = []
    times = []

    for start in range(
        0,
        len(filtered) - window_size,
        hop_size
    ):

        end = (
            start
            + window_size
        )

        window = filtered[
            start:end
        ]

        # Average energy across microphones.
        value = np.sqrt(
            np.mean(
                np.square(window)
            )
        )

        values.append(
            value
        )

        times.append(
            start / SAMPLE_RATE
        )

    return (
        np.array(times),
        np.array(values)
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("SARV ACOUSTIC REFLECTION TEST v1")
    print("=" * 60)

    print()
    print(
        f"Input device : {INPUT_DEVICE}"
    )

    print(
        f"Output device: {OUTPUT_DEVICE}"
    )

    print(
        f"Sample rate  : {SAMPLE_RATE} Hz"
    )

    print(
        f"Test frequency: "
        f"{CENTER_FREQUENCY} Hz"
    )

    print()

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print(
        "TEST 1 — NO HAND"
    )

    print()

    print(
        "Keep your hand away from the laptop."
    )

    print(
        "Keep everything else unchanged."
    )

    input(
        "Press ENTER to begin..."
    )

    baseline = run_measurement(
        "BASELINE — NO HAND"
    )

    baseline_filtered, baseline_rms = (
        analyze(baseline)
    )

    # --------------------------------------------------------
    # STATIONARY HAND
    # --------------------------------------------------------

    print()
    print(
        "TEST 2 — STATIONARY HAND"
    )

    print()

    print(
        "Place your hand approximately"
    )

    print(
        "1 metre in front of the laptop."
    )

    print(
        "Keep your hand still."
    )

    input(
        "Press ENTER when ready..."
    )

    stationary = run_measurement(
        "STATIONARY HAND — ~1 METRE"
    )

    stationary_filtered, stationary_rms = (
        analyze(stationary)
    )

    # --------------------------------------------------------
    # MOVING HAND
    # --------------------------------------------------------

    print()
    print(
        "TEST 3 — MOVING HAND"
    )

    print()

    print(
        "Place your hand approximately"
    )

    print(
        "1 metre in front of the laptop."
    )

    print()

    print(
        "During the recording, slowly move:"
    )

    print(
        "LEFT → RIGHT → LEFT"
    )

    print()

    input(
        "Press ENTER when ready..."
    )

    movement = run_measurement(
        "MOVING HAND — ~1 METRE"
    )

    movement_filtered, movement_rms = (
        analyze(movement)
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print()

    for i in range(
        INPUT_CHANNELS
    ):

        print(
            f"Channel {i + 1}:"
        )

        print(
            f"  Baseline   : "
            f"{baseline_rms[i]:.8e}"
        )

        print(
            f"  Stationary : "
            f"{stationary_rms[i]:.8e}"
        )

        print(
            f"  Movement   : "
            f"{movement_rms[i]:.8e}"
        )

        stationary_ratio = (
            stationary_rms[i]
            / max(
                baseline_rms[i],
                1e-12
            )
        )

        movement_ratio = (
            movement_rms[i]
            / max(
                baseline_rms[i],
                1e-12
            )
        )

        print(
            f"  Stationary/Baseline: "
            f"{stationary_ratio:.2f}x"
        )

        print(
            f"  Movement/Baseline  : "
            f"{movement_ratio:.2f}x"
        )

        print()

    # --------------------------------------------------------
    # TIME-VARYING ANALYSIS
    # --------------------------------------------------------

    baseline_time, baseline_energy = (
        calculate_time_rms(
            baseline_filtered
        )
    )

    stationary_time, stationary_energy = (
        calculate_time_rms(
            stationary_filtered
        )
    )

    movement_time, movement_energy = (
        calculate_time_rms(
            movement_filtered
        )
    )

    # --------------------------------------------------------
    # GRAPH 1 — OVERALL TIME RESPONSE
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        baseline_time,
        baseline_energy,
        label="No hand"
    )

    plt.plot(
        stationary_time,
        stationary_energy,
        label="Stationary hand"
    )

    plt.plot(
        movement_time,
        movement_energy,
        label="Moving hand"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Filtered acoustic energy"
    )

    plt.title(
        "SARV 8 kHz Acoustic Reflection Test"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()