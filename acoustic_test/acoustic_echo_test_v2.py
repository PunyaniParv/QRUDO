import numpy as np
import sounddevice as sd
from scipy.signal import chirp, correlate, find_peaks
import time


# ============================================================
# SARV - ACOUSTIC ECHO TEST V2
# ============================================================

# -----------------------------
# AUDIO SETTINGS
# -----------------------------

SAMPLE_RATE = 44100

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

CHANNELS = 1

# Chirp settings
CHIRP_START = 7500
CHIRP_END = 8500
CHIRP_DURATION = 0.10       # 100 ms

# Recording duration
RECORD_DURATION = 0.30      # 300 ms

# Number of repeated measurements
NUM_TESTS = 50

# Time between measurements
WAIT_TIME = 0.15


# -----------------------------
# ECHO SETTINGS
# -----------------------------

SPEED_OF_SOUND = 343.0      # m/s

# Minimum delay after direct signal
# before we consider something an echo.
MIN_ECHO_DELAY = 0.0015     # 1.5 ms

# Maximum delay we care about
MAX_ECHO_DELAY = 0.030      # 30 ms

# Peak detection threshold
PEAK_THRESHOLD = 0.20


# ============================================================
# GENERATE TRANSMITTED CHIRP
# ============================================================

chirp_samples = int(CHIRP_DURATION * SAMPLE_RATE)

t = np.linspace(
    0,
    CHIRP_DURATION,
    chirp_samples,
    endpoint=False
)

tx_signal = chirp(
    t,
    f0=CHIRP_START,
    f1=CHIRP_END,
    t1=CHIRP_DURATION,
    method="linear"
)

# Apply fade in/out to reduce clicks
fade_length = int(0.01 * SAMPLE_RATE)

fade = np.linspace(0, 1, fade_length)

tx_signal[:fade_length] *= fade
tx_signal[-fade_length:] *= fade[::-1]

# Reduce amplitude to avoid speaker clipping
tx_signal *= 0.35


# ============================================================
# DISPLAY SETTINGS
# ============================================================

print()
print("=" * 65)
print("SARV ACOUSTIC ECHO TEST V2")
print("=" * 65)

print(f"Input device  : {INPUT_DEVICE}")
print(f"Output device : {OUTPUT_DEVICE}")
print(f"Sample rate   : {SAMPLE_RATE}")
print(f"Chirp         : {CHIRP_START} - {CHIRP_END} Hz")
print(f"Chirp duration: {CHIRP_DURATION * 1000:.0f} ms")
print()

print("IMPORTANT:")
print("Keep the laptop in a fixed position.")
print("First test WITHOUT your hand.")
print("Then test WITH your hand.")
print()

input("Press ENTER to start...")
print()


# ============================================================
# SINGLE MEASUREMENT
# ============================================================

def perform_measurement():

    # Create recording buffer
    total_samples = int(RECORD_DURATION * SAMPLE_RATE)

    # Put chirp at the beginning of the playback buffer
    playback = np.zeros(total_samples, dtype=np.float32)
    playback[:len(tx_signal)] = tx_signal

    # Play and record simultaneously
    recording = sd.playrec(
        playback,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        blocking=True
    )

    # Convert to mono
    recording = recording[:, 0]

    # Remove DC component
    recording = recording - np.mean(recording)

    # --------------------------------------------------------
    # CROSS CORRELATION
    # --------------------------------------------------------

    correlation = correlate(
        recording,
        tx_signal,
        mode="full"
    )

    # Only positive lags are useful
    tx_len = len(tx_signal)

    correlation = correlation[tx_len - 1:]

    # Normalize
    max_corr = np.max(np.abs(correlation))

    if max_corr == 0:
        return None

    correlation_normalized = np.abs(correlation) / max_corr

    # --------------------------------------------------------
    # FIND PEAKS
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        correlation_normalized,
        height=PEAK_THRESHOLD,
        distance=int(0.0005 * SAMPLE_RATE)
    )

    if len(peaks) == 0:
        return None

    # Sort peaks by strength
    peak_strengths = correlation_normalized[peaks]

    sorted_indices = np.argsort(peak_strengths)[::-1]

    peaks = peaks[sorted_indices]
    peak_strengths = peak_strengths[sorted_indices]

    # --------------------------------------------------------
    # DIRECT PATH
    # --------------------------------------------------------

    direct_peak = peaks[0]

    direct_delay = direct_peak / SAMPLE_RATE

    # --------------------------------------------------------
    # SEARCH FOR DELAYED ECHO
    # --------------------------------------------------------

    echo_peak = None
    echo_strength = None

    min_echo_samples = int(
        MIN_ECHO_DELAY * SAMPLE_RATE
    )

    max_echo_samples = int(
        MAX_ECHO_DELAY * SAMPLE_RATE
    )

    for peak, strength in zip(peaks, peak_strengths):

        delay_from_direct = peak - direct_peak

        if (
            delay_from_direct >= min_echo_samples
            and
            delay_from_direct <= max_echo_samples
        ):
            echo_peak = peak
            echo_strength = strength
            break

    # --------------------------------------------------------
    # CALCULATE DISTANCE
    # --------------------------------------------------------

    distance = None
    echo_delay = None

    if echo_peak is not None:

        echo_delay_samples = echo_peak - direct_peak

        echo_delay = echo_delay_samples / SAMPLE_RATE

        # Approximate distance assuming speaker and microphone
        # are close together.
        #
        # Sound travels:
        #
        # speaker -> hand -> microphone
        #
        # approximately 2 * distance.

        distance = (
            SPEED_OF_SOUND * echo_delay
        ) / 2.0

    return {
        "direct_delay": direct_delay,
        "direct_strength": peak_strengths[0],
        "echo_delay": echo_delay,
        "echo_strength": echo_strength,
        "distance": distance,
        "peaks": list(
            zip(
                peaks[:10],
                peak_strengths[:10]
            )
        )
    }


# ============================================================
# MAIN LOOP
# ============================================================

try:

    for test_number in range(1, NUM_TESTS + 1):

        result = perform_measurement()

        print("-" * 65)

        if result is None:

            print(
                f"[{test_number:02d}] "
                "No significant correlation detected."
            )

            time.sleep(WAIT_TIME)
            continue

        print(
            f"[{test_number:02d}] "
            f"Direct: "
            f"{result['direct_delay'] * 1000:7.2f} ms "
            f"| Strength: "
            f"{result['direct_strength']:.3f}"
        )

        if result["echo_delay"] is not None:

            print(
                f"      ECHO:   "
                f"{result['echo_delay'] * 1000:7.2f} ms "
                f"| Strength: "
                f"{result['echo_strength']:.3f}"
            )

            print(
                f"      Approx. distance: "
                f"{result['distance'] * 100:.1f} cm"
            )

        else:

            print(
                "      ECHO:   "
                "No delayed reflection detected"
            )

        # Show the strongest peaks
        print("      Peaks:", end=" ")

        for peak, strength in result["peaks"][:5]:

            delay_ms = peak / SAMPLE_RATE * 1000

            print(
                f"{delay_ms:.2f}ms/{strength:.2f}",
                end="  "
            )

        print()

        time.sleep(WAIT_TIME)


except KeyboardInterrupt:

    print()
    print("Test stopped by user.")


except Exception as e:

    print()
    print("=" * 65)
    print("ERROR")
    print("=" * 65)
    print(e)


finally:

    sd.stop()

    print()
    print("=" * 65)
    print("SARV V2 TEST FINISHED")
    print("=" * 65)