import numpy as np
import sounddevice as sd
from scipy.signal import chirp, correlate
import time


# ============================================================
# SARV - ACOUSTIC ECHO TEST V2.1
# Hand Presence / Motion Detection Experiment
# ============================================================


# ------------------------------------------------------------
# AUDIO SETTINGS
# ------------------------------------------------------------

SAMPLE_RATE = 44100

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

CHANNELS = 1

CHIRP_START = 7500
CHIRP_END = 8500

CHIRP_DURATION = 0.10       # 100 ms
RECORD_DURATION = 0.30      # 300 ms

SIGNAL_AMPLITUDE = 0.35


# ------------------------------------------------------------
# EXPERIMENT SETTINGS
# ------------------------------------------------------------

NUM_SAMPLES = 100

WAIT_TIME = 0.08


# ------------------------------------------------------------
# CREATE CHIRP
# ------------------------------------------------------------

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


# Fade in/out to avoid clicks

fade_length = int(0.01 * SAMPLE_RATE)

fade = np.linspace(
    0,
    1,
    fade_length
)

tx_signal[:fade_length] *= fade
tx_signal[-fade_length:] *= fade[::-1]

tx_signal *= SIGNAL_AMPLITUDE


# ============================================================
# MEASUREMENT FUNCTION
# ============================================================

def measure():

    total_samples = int(
        RECORD_DURATION * SAMPLE_RATE
    )

    playback = np.zeros(
        total_samples,
        dtype=np.float32
    )

    playback[:len(tx_signal)] = tx_signal

    # --------------------------------------------------------
    # PLAY + RECORD
    # --------------------------------------------------------

    recording = sd.playrec(
        playback,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        blocking=True
    )

    recording = recording[:, 0]

    # Remove DC offset

    recording -= np.mean(recording)

    # --------------------------------------------------------
    # BASIC SIGNAL ENERGY
    # --------------------------------------------------------

    rms = np.sqrt(
        np.mean(recording ** 2)
    )

    peak_amplitude = np.max(
        np.abs(recording)
    )

    # --------------------------------------------------------
    # CROSS CORRELATION
    # --------------------------------------------------------

    correlation = correlate(
        recording,
        tx_signal,
        mode="full"
    )

    correlation = correlation[
        len(tx_signal) - 1:
    ]

    correlation_abs = np.abs(correlation)

    max_correlation = np.max(
        correlation_abs
    )

    if max_correlation == 0:

        return {
            "rms": rms,
            "peak": peak_amplitude,
            "correlation": 0,
            "energy": 0
        }

    # --------------------------------------------------------
    # NORMALIZED CORRELATION
    # --------------------------------------------------------

    normalized = (
        correlation_abs /
        max_correlation
    )

    # --------------------------------------------------------
    # EARLY RESPONSE
    # --------------------------------------------------------

    early_samples = int(
        0.015 * SAMPLE_RATE
    )

    early_region = normalized[
        :early_samples
    ]

    early_energy = np.mean(
        early_region ** 2
    )

    # --------------------------------------------------------
    # LATE RESPONSE
    # --------------------------------------------------------

    late_start = int(
        0.003 * SAMPLE_RATE
    )

    late_end = int(
        0.030 * SAMPLE_RATE
    )

    late_region = normalized[
        late_start:late_end
    ]

    late_energy = np.mean(
        late_region ** 2
    )

    # Maximum delayed response

    delayed_peak = np.max(
        late_region
    )

    # --------------------------------------------------------
    # RESPONSE RATIO
    # --------------------------------------------------------

    if early_energy > 0:

        response_ratio = (
            late_energy /
            early_energy
        )

    else:

        response_ratio = 0


    return {
        "rms": rms,
        "peak": peak_amplitude,
        "correlation": max_correlation,
        "energy": late_energy,
        "delayed_peak": delayed_peak,
        "ratio": response_ratio
    }


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 70)
print("SARV ACOUSTIC ECHO TEST V2.1")
print("HAND PRESENCE / MOTION EXPERIMENT")
print("=" * 70)

print()
print(f"Input device  : {INPUT_DEVICE}")
print(f"Output device : {OUTPUT_DEVICE}")
print(f"Sample rate   : {SAMPLE_RATE}")
print(f"Chirp         : {CHIRP_START} - {CHIRP_END} Hz")
print(f"Chirp duration: {CHIRP_DURATION * 1000:.0f} ms")
print()

print("This experiment does NOT estimate distance.")
print("It measures acoustic changes caused by your environment/hand.")
print()


# ============================================================
# EXPERIMENT
# ============================================================

try:

    # --------------------------------------------------------
    # STAGE 1 - BASELINE
    # --------------------------------------------------------

    print("=" * 70)
    print("STAGE 1 - BASELINE")
    print("=" * 70)

    print()
    print("Remove your hand from in front of the laptop.")
    print("Keep the laptop completely still.")
    print()

    input("Press ENTER to begin baseline...")

    baseline = []

    print()
    print("Collecting baseline...")
    print()

    for i in range(NUM_SAMPLES):

        result = measure()

        baseline.append(result)

        print(
            f"\rSample {i + 1:03d}/{NUM_SAMPLES} | "
            f"RMS {result['rms']:.5f} | "
            f"Corr {result['correlation']:.3f} | "
            f"Delayed {result['delayed_peak']:.3f} | "
            f"Ratio {result['ratio']:.3f}",
            end=""
        )

        time.sleep(WAIT_TIME)

    print()
    print()

    # --------------------------------------------------------
    # CALCULATE BASELINE
    # --------------------------------------------------------

    baseline_rms = np.mean(
        [x["rms"] for x in baseline]
    )

    baseline_corr = np.mean(
        [x["correlation"] for x in baseline]
    )

    baseline_delayed = np.mean(
        [x["delayed_peak"] for x in baseline]
    )

    baseline_ratio = np.mean(
        [x["ratio"] for x in baseline]
    )

    print("BASELINE COMPLETE")
    print()

    print(
        f"Average RMS          : {baseline_rms:.6f}"
    )

    print(
        f"Average correlation  : {baseline_corr:.4f}"
    )

    print(
        f"Average delayed peak : {baseline_delayed:.4f}"
    )

    print(
        f"Average response     : {baseline_ratio:.4f}"
    )

    print()


    # --------------------------------------------------------
    # STAGE 2 - HAND PRESENT
    # --------------------------------------------------------

    print("=" * 70)
    print("STAGE 2 - STATIONARY HAND")
    print("=" * 70)

    print()
    print("Place your hand approximately")
    print("20-30 cm in front of the laptop.")
    print()
    print("Keep your hand COMPLETELY STILL.")
    print()

    input("Press ENTER when ready...")

    hand = []

    print()
    print("Collecting hand measurements...")
    print()

    for i in range(NUM_SAMPLES):

        result = measure()

        hand.append(result)

        print(
            f"\rSample {i + 1:03d}/{NUM_SAMPLES} | "
            f"RMS {result['rms']:.5f} | "
            f"Corr {result['correlation']:.3f} | "
            f"Delayed {result['delayed_peak']:.3f} | "
            f"Ratio {result['ratio']:.3f}",
            end=""
        )

        time.sleep(WAIT_TIME)

    print()
    print()

    # --------------------------------------------------------
    # HAND AVERAGES
    # --------------------------------------------------------

    hand_rms = np.mean(
        [x["rms"] for x in hand]
    )

    hand_corr = np.mean(
        [x["correlation"] for x in hand]
    )

    hand_delayed = np.mean(
        [x["delayed_peak"] for x in hand]
    )

    hand_ratio = np.mean(
        [x["ratio"] for x in hand]
    )


    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    print("=" * 70)
    print("BASELINE vs HAND")
    print("=" * 70)

    print()

    print(
        f"RMS:"
        f"       {baseline_rms:.6f}"
        f"  ->  {hand_rms:.6f}"
    )

    print(
        f"Correlation:"
        f" {baseline_corr:.4f}"
        f"  ->  {hand_corr:.4f}"
    )

    print(
        f"Delayed peak:"
        f"  {baseline_delayed:.4f}"
        f"  ->  {hand_delayed:.4f}"
    )

    print(
        f"Response ratio:"
        f" {baseline_ratio:.4f}"
        f"  ->  {hand_ratio:.4f}"
    )

    print()

    # --------------------------------------------------------
    # PERCENT CHANGES
    # --------------------------------------------------------

    def percent_change(a, b):

        if abs(a) < 1e-12:
            return 0

        return ((b - a) / abs(a)) * 100


    print(
        f"RMS change:"
        f"       {percent_change(baseline_rms, hand_rms):+.2f}%"
    )

    print(
        f"Correlation change:"
        f" {percent_change(baseline_corr, hand_corr):+.2f}%"
    )

    print(
        f"Delayed peak change:"
        f"  {percent_change(baseline_delayed, hand_delayed):+.2f}%"
    )

    print(
        f"Response change:"
        f"       {percent_change(baseline_ratio, hand_ratio):+.2f}%"
    )

    print()


    # --------------------------------------------------------
    # STAGE 3 - MOVING HAND
    # --------------------------------------------------------

    print("=" * 70)
    print("STAGE 3 - MOVING HAND")
    print("=" * 70)

    print()
    print("Move your hand slowly:")
    print()
    print("        FAR")
    print("         ↓")
    print("        💻")
    print("         ↑")
    print("        NEAR")
    print()
    print("Then move it back away.")
    print()
    print("Press ENTER to start...")
    input()

    print()
    print("TRACKING...")
    print()

    for i in range(150):

        result = measure()

        # Difference from baseline

        rms_change = percent_change(
            baseline_rms,
            result["rms"]
        )

        delayed_change = percent_change(
            baseline_delayed,
            result["delayed_peak"]
        )

        ratio_change = percent_change(
            baseline_ratio,
            result["ratio"]
        )

        print(
            f"\r{i + 1:03d} | "
            f"RMS Δ {rms_change:+7.1f}% | "
            f"Delayed Δ {delayed_change:+7.1f}% | "
            f"Ratio Δ {ratio_change:+7.1f}%",
            end=""
        )

        time.sleep(0.05)

    print()
    print()

    print("=" * 70)
    print("V2.1 EXPERIMENT COMPLETE")
    print("=" * 70)


except KeyboardInterrupt:

    print()
    print()
    print("Experiment stopped by user.")


except Exception as e:

    print()
    print()
    print("=" * 70)
    print("ERROR")
    print("=" * 70)
    print(e)


finally:

    sd.stop()