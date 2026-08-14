import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, chirp

# ============================================================
# SARV Acoustic Feasibility Test v1
# ============================================================

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

SAMPLE_RATE = 44100

# Experimental acoustic band
F_START = 18000
F_END = 20000

# Keep signal relatively short and low amplitude
CHIRP_DURATION = 0.20

# Silence between chirps
REPEAT_INTERVAL = 0.50

# Number of chirps during each measurement
NUM_CHIRPS = 8

# Recording margin before/after the test
PRE_ROLL = 0.50
POST_ROLL = 0.50

# Output volume
AMPLITUDE = 0.08

# Microphone channels
INPUT_CHANNELS = 4

# We initially use stereo output.
OUTPUT_CHANNELS = 2


def create_chirp():
    """Create one 18-20 kHz ultrasonic/high-frequency chirp."""
    t = np.arange(
        int(CHIRP_DURATION * SAMPLE_RATE)
    ) / SAMPLE_RATE

    signal = chirp(
        t,
        f0=F_START,
        f1=F_END,
        t1=CHIRP_DURATION,
        method="linear"
    )

    # Smooth edges to reduce clicks/transients.
    window = np.hanning(len(signal))

    signal = signal * window
    signal *= AMPLITUDE

    return signal.astype(np.float32)


def create_test_signal():
    """Create repeated chirps separated by silence."""
    chirp_signal = create_chirp()

    interval_samples = int(REPEAT_INTERVAL * SAMPLE_RATE)

    gap = np.zeros(
        interval_samples - len(chirp_signal),
        dtype=np.float32
    )

    single_period = np.concatenate([
        chirp_signal,
        gap
    ])

    signal = np.tile(single_period, NUM_CHIRPS)

    return signal.astype(np.float32)


def bandpass_filter(signal):
    """Keep only the experimental 18-20 kHz region."""

    low = F_START / (SAMPLE_RATE / 2)
    high = F_END / (SAMPLE_RATE / 2)

    sos = butter(
        6,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(sos, signal, axis=0)


def rms(signal):
    """Calculate RMS energy."""
    return np.sqrt(np.mean(np.square(signal)))


def run_measurement(label):
    print()
    print("=" * 60)
    print(label)
    print("=" * 60)

    test_signal = create_test_signal()

    total_duration = (
        PRE_ROLL
        + len(test_signal) / SAMPLE_RATE
        + POST_ROLL
    )

    total_samples = int(total_duration * SAMPLE_RATE)

    print(f"Recording duration: {total_duration:.2f} seconds")
    print("Starting in 2 seconds...")

    sd.sleep(2000)

    print("Recording...")

    # Four microphone channels are recorded.
    recording = sd.playrec(
        np.pad(
            test_signal,
            (
                int(PRE_ROLL * SAMPLE_RATE),
                int(POST_ROLL * SAMPLE_RATE)
            )
        ).reshape(-1, 1),
        samplerate=SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        dtype="float32",
        blocking=True
    )

    print("Recording complete.")

    return recording


def analyze(recording, label):
    """Analyze the received microphone signal."""

    filtered = bandpass_filter(recording)

    # Calculate RMS for each microphone.
    channel_rms = np.sqrt(
        np.mean(np.square(filtered), axis=0)
    )

    print()
    print(f"{label} filtered RMS:")

    for i, value in enumerate(channel_rms):
        print(f"  Microphone channel {i + 1}: {value:.8f}")

    return filtered, channel_rms


def plot_result(baseline_filtered, movement_filtered):
    """Plot the filtered microphone response."""

    baseline_rms = np.sqrt(
        np.mean(np.square(baseline_filtered), axis=1)
    )

    movement_rms = np.sqrt(
        np.mean(np.square(movement_filtered), axis=1)
    )

    time_baseline = np.arange(
        len(baseline_rms)
    ) / SAMPLE_RATE

    time_movement = np.arange(
        len(movement_rms)
    ) / SAMPLE_RATE

    plt.figure(figsize=(12, 6))

    plt.plot(
        time_baseline,
        baseline_rms,
        label="Baseline"
    )

    plt.plot(
        time_movement,
        movement_rms,
        label="Hand movement"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Filtered RMS")
    plt.title("SARV Acoustic Reflection Test — 18–20 kHz")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():

    print("=" * 60)
    print("SARV ACOUSTIC FEASIBILITY TEST v1")
    print("=" * 60)

    print()
    print(f"Input device : {INPUT_DEVICE}")
    print(f"Output device: {OUTPUT_DEVICE}")
    print(f"Sample rate  : {SAMPLE_RATE} Hz")
    print(f"Frequency    : {F_START}-{F_END} Hz")
    print(f"Input channels: {INPUT_CHANNELS}")
    print()

    print("Checking devices...")

    input_info = sd.query_devices(INPUT_DEVICE)
    output_info = sd.query_devices(OUTPUT_DEVICE)

    print()
    print("INPUT:")
    print(input_info)

    print()
    print("OUTPUT:")
    print(output_info)

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print()
    print("BASELINE TEST")
    print()
    print(
        "Keep your hand away from the laptop."
    )
    print(
        "Remain still during the measurement."
    )

    input("Press ENTER to begin baseline test...")

    baseline = run_measurement(
        "BASELINE — NO HAND MOVEMENT"
    )

    baseline_filtered, baseline_channel_rms = analyze(
        baseline,
        "Baseline"
    )

    # --------------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------------

    print()
    print("HAND MOVEMENT TEST")
    print()
    print(
        "Move your hand slowly left → right"
    )
    print(
        "approximately 1 metre in front of the laptop."
    )
    print()
    print(
        "Do not touch the laptop."
    )

    input("Press ENTER when you are ready...")

    movement = run_measurement(
        "MOVEMENT — HAND AT APPROXIMATELY 1 METRE"
    )

    movement_filtered, movement_channel_rms = analyze(
        movement,
        "Movement"
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print()

    for i in range(INPUT_CHANNELS):

        baseline_value = baseline_channel_rms[i]
        movement_value = movement_channel_rms[i]

        ratio = movement_value / max(
            baseline_value,
            1e-12
        )

        print(
            f"Channel {i + 1}: "
            f"baseline={baseline_value:.8f}, "
            f"movement={movement_value:.8f}, "
            f"ratio={ratio:.2f}x"
        )

    print()
    print(
        "A ratio substantially above 1 may indicate "
        "additional acoustic energy during movement."
    )

    print()
    print(
        "IMPORTANT: This is only a feasibility experiment."
    )
    print(
        "A higher RMS does NOT yet prove that the signal "
        "comes from the hand."
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    plot_result(
        baseline_filtered,
        movement_filtered
    )


if __name__ == "__main__":
    main()