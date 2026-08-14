import numpy as np
import sounddevice as sd
from scipy.signal import chirp, correlate
import time
import csv
import os


# ============================================================
# SARV - ACOUSTIC ECHO TEST V2.2
# CONTROLLED HAND POSITION + MOTION EXPERIMENT
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

BASELINE_SAMPLES = 60

POSITION_SAMPLES = 60

MOTION_SAMPLES = 150

WAIT_TIME = 0.08

RESULT_FILE = "v2_2_results.csv"


# ------------------------------------------------------------
# TEST POSITIONS
# ------------------------------------------------------------

POSITIONS_CM = [20, 40, 60]


# ============================================================
# CREATE TRANSMITTED CHIRP
# ============================================================

chirp_samples = int(
    CHIRP_DURATION * SAMPLE_RATE
)

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


# Fade in / fade out

fade_length = int(
    0.01 * SAMPLE_RATE
)

fade = np.linspace(
    0,
    1,
    fade_length
)

tx_signal[:fade_length] *= fade
tx_signal[-fade_length:] *= fade[::-1]

tx_signal *= SIGNAL_AMPLITUDE


# ============================================================
# MEASUREMENT
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

    # Remove DC

    recording -= np.mean(recording)


    # ========================================================
    # FEATURE 1 - RAW RMS
    # ========================================================

    rms = np.sqrt(
        np.mean(recording ** 2)
    )


    # ========================================================
    # FEATURE 2 - RAW PEAK
    # ========================================================

    raw_peak = np.max(
        np.abs(recording)
    )


    # ========================================================
    # FEATURE 3 - CROSS CORRELATION
    # ========================================================

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


    if max_correlation <= 1e-12:

        return {
            "rms": rms,
            "raw_peak": raw_peak,
            "correlation": 0.0,
            "early_energy": 0.0,
            "late_energy": 0.0,
            "late_peak": 0.0,
            "response_ratio": 0.0,
            "late_to_total": 0.0
        }


    # ========================================================
    # NORMALIZED CORRELATION
    # ========================================================

    normalized = (
        correlation_abs /
        max_correlation
    )


    # ========================================================
    # FEATURE 4 - EARLY CORRELATION ENERGY
    # ========================================================

    early_end = int(
        0.003 * SAMPLE_RATE
    )

    early_region = normalized[
        :early_end
    ]

    early_energy = np.mean(
        early_region ** 2
    )


    # ========================================================
    # FEATURE 5 - LATE CORRELATION ENERGY
    # ========================================================

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


    # ========================================================
    # FEATURE 6 - LATE PEAK
    # ========================================================

    late_peak = np.max(
        late_region
    )


    # ========================================================
    # FEATURE 7 - RESPONSE RATIO
    # ========================================================

    if early_energy > 1e-12:

        response_ratio = (
            late_energy /
            early_energy
        )

    else:

        response_ratio = 0.0


    # ========================================================
    # FEATURE 8 - LATE / TOTAL ENERGY
    # ========================================================

    total_energy = np.mean(
        normalized ** 2
    )

    if total_energy > 1e-12:

        late_to_total = (
            late_energy /
            total_energy
        )

    else:

        late_to_total = 0.0


    return {
        "rms": rms,
        "raw_peak": raw_peak,
        "correlation": max_correlation,
        "early_energy": early_energy,
        "late_energy": late_energy,
        "late_peak": late_peak,
        "response_ratio": response_ratio,
        "late_to_total": late_to_total
    }


# ============================================================
# CSV SETUP
# ============================================================

csv_file = open(
    RESULT_FILE,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "timestamp",
    "stage",
    "position_cm",
    "sample",
    "rms",
    "raw_peak",
    "correlation",
    "early_energy",
    "late_energy",
    "late_peak",
    "response_ratio",
    "late_to_total"
])


# ============================================================
# CSV LOGGING
# ============================================================

def save_result(
    stage,
    position,
    sample_number,
    result
):

    csv_writer.writerow([
        time.time(),
        stage,
        position,
        sample_number,
        result["rms"],
        result["raw_peak"],
        result["correlation"],
        result["early_energy"],
        result["late_energy"],
        result["late_peak"],
        result["response_ratio"],
        result["late_to_total"]
    ])

    csv_file.flush()


# ============================================================
# COLLECT MEASUREMENTS
# ============================================================

def collect_measurements(
    stage,
    position,
    number_of_samples,
    baseline=None
):

    results = []


    for i in range(
        number_of_samples
    ):

        result = measure()

        results.append(result)

        save_result(
            stage,
            position,
            i + 1,
            result
        )


        # ----------------------------------------------------
        # BASELINE-RELATIVE VALUES
        # ----------------------------------------------------

        if baseline is not None:

            rms_change = percent_change(
                baseline["rms"],
                result["rms"]
            )

            corr_change = percent_change(
                baseline["correlation"],
                result["correlation"]
            )

            late_change = percent_change(
                baseline["late_peak"],
                result["late_peak"]
            )

            ratio_change = percent_change(
                baseline["response_ratio"],
                result["response_ratio"]
            )

        else:

            rms_change = 0
            corr_change = 0
            late_change = 0
            ratio_change = 0


        # ----------------------------------------------------
        # PRINT EVERY 5 SAMPLES
        # ----------------------------------------------------

        if (
            (i + 1) % 5 == 0
            or
            i == 0
            or
            i == number_of_samples - 1
        ):

            print(
                f"Sample {i + 1:03d}/{number_of_samples} | "
                f"RMS {result['rms']:.6f} "
                f"({rms_change:+6.1f}%) | "
                f"Corr {result['correlation']:.4f} "
                f"({corr_change:+6.1f}%) | "
                f"Late {result['late_peak']:.4f} "
                f"({late_change:+6.1f}%) | "
                f"Ratio {result['response_ratio']:.4f} "
                f"({ratio_change:+6.1f}%)"
            )


        time.sleep(
            WAIT_TIME
        )


    return results


# ============================================================
# STATISTICS
# ============================================================

def average_results(results):

    keys = [
        "rms",
        "raw_peak",
        "correlation",
        "early_energy",
        "late_energy",
        "late_peak",
        "response_ratio",
        "late_to_total"
    ]

    averages = {}

    for key in keys:

        values = [
            result[key]
            for result in results
        ]

        averages[key] = np.mean(
            values
        )

    return averages


def standard_deviations(results):

    keys = [
        "rms",
        "raw_peak",
        "correlation",
        "early_energy",
        "late_energy",
        "late_peak",
        "response_ratio",
        "late_to_total"
    ]

    deviations = {}

    for key in keys:

        values = [
            result[key]
            for result in results
        ]

        deviations[key] = np.std(
            values
        )

    return deviations


def percent_change(
    baseline,
    current
):

    if abs(baseline) < 1e-12:

        return 0.0

    return (
        (current - baseline)
        /
        abs(baseline)
    ) * 100


# ============================================================
# DISPLAY STATISTICS
# ============================================================

def print_statistics(
    title,
    averages,
    deviations
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print()

    print(
        f"RMS:"
        f"              {averages['rms']:.6f}"
        f" ± {deviations['rms']:.6f}"
    )

    print(
        f"Raw peak:"
        f"          {averages['raw_peak']:.6f}"
        f" ± {deviations['raw_peak']:.6f}"
    )

    print(
        f"Correlation:"
        f"       {averages['correlation']:.6f}"
        f" ± {deviations['correlation']:.6f}"
    )

    print(
        f"Early energy:"
        f"      {averages['early_energy']:.6f}"
        f" ± {deviations['early_energy']:.6f}"
    )

    print(
        f"Late energy:"
        f"       {averages['late_energy']:.6f}"
        f" ± {deviations['late_energy']:.6f}"
    )

    print(
        f"Late peak:"
        f"         {averages['late_peak']:.6f}"
        f" ± {deviations['late_peak']:.6f}"
    )

    print(
        f"Response ratio:"
        f"    {averages['response_ratio']:.6f}"
        f" ± {deviations['response_ratio']:.6f}"
    )

    print(
        f"Late / total:"
        f"      {averages['late_to_total']:.6f}"
        f" ± {deviations['late_to_total']:.6f}"
    )


def print_comparison(
    baseline,
    position,
    averages
):

    print()

    print(
        f"{position:>3} cm | "
        f"RMS "
        f"{percent_change(baseline['rms'], averages['rms']):+7.2f}% | "
        f"Corr "
        f"{percent_change(baseline['correlation'], averages['correlation']):+7.2f}% | "
        f"Late "
        f"{percent_change(baseline['late_peak'], averages['late_peak']):+7.2f}% | "
        f"Ratio "
        f"{percent_change(baseline['response_ratio'], averages['response_ratio']):+7.2f}%"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

print()
print("=" * 70)
print("SARV ACOUSTIC ECHO TEST V2.2")
print("CONTROLLED HAND POSITION + MOTION EXPERIMENT")
print("=" * 70)

print()

print(f"Input device  : {INPUT_DEVICE}")
print(f"Output device : {OUTPUT_DEVICE}")
print(f"Sample rate   : {SAMPLE_RATE}")
print(
    f"Chirp         : "
    f"{CHIRP_START} - {CHIRP_END} Hz"
)

print()

print(
    "Results will be saved to:"
)
print(
    os.path.abspath(RESULT_FILE)
)

print()

try:

    # ========================================================
    # STAGE 1 - BASELINE
    # ========================================================

    print("=" * 70)
    print("STAGE 1 - BASELINE")
    print("=" * 70)

    print()
    print(
        "Remove your hand from the area in front of the laptop."
    )

    print(
        "Keep the laptop and your body as still as possible."
    )

    input(
        "\nPress ENTER to begin baseline..."
    )

    print()
    print("Collecting baseline...")
    print()

    baseline_results = collect_measurements(
        "baseline",
        0,
        BASELINE_SAMPLES
    )

    baseline_avg = average_results(
        baseline_results
    )

    baseline_std = standard_deviations(
        baseline_results
    )

    print_statistics(
        "BASELINE STATISTICS",
        baseline_avg,
        baseline_std
    )


    # ========================================================
    # STAGE 2 - CONTROLLED POSITIONS
    # ========================================================

    position_results = {}

    position_averages = {}


    for position in POSITIONS_CM:

        print()
        print("=" * 70)
        print(
            f"STAGE 2 - HAND AT {position} CM"
        )
        print("=" * 70)

        print()

        print(
            f"Place your hand approximately {position} cm "
            "in front of the laptop."
        )

        print(
            "Keep your hand completely STILL."
        )

        print()

        input(
            "Press ENTER when ready..."
        )

        print()
        print(
            f"Collecting {position} cm measurements..."
        )

        print()

        results = collect_measurements(
            "position",
            position,
            POSITION_SAMPLES,
            baseline_avg
        )

        position_results[position] = results

        averages = average_results(
            results
        )

        deviations = standard_deviations(
            results
        )

        position_averages[position] = averages

        print_statistics(
            f"{position} CM STATISTICS",
            averages,
            deviations
        )

        print_comparison(
            baseline_avg,
            position,
            averages
        )


    # ========================================================
    # POSITION COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("POSITION COMPARISON")
    print("=" * 70)

    print()

    print(
        "Position | RMS Δ | Correlation Δ | Late Peak Δ | Ratio Δ"
    )

    print("-" * 70)

    for position in POSITIONS_CM:

        averages = position_averages[position]

        print(
            f"{position:>3} cm     | "
            f"{percent_change(baseline_avg['rms'], averages['rms']):+7.2f}% | "
            f"{percent_change(baseline_avg['correlation'], averages['correlation']):+13.2f}% | "
            f"{percent_change(baseline_avg['late_peak'], averages['late_peak']):+11.2f}% | "
            f"{percent_change(baseline_avg['response_ratio'], averages['response_ratio']):+8.2f}%"
        )


    # ========================================================
    # STAGE 3 - CONTINUOUS MOTION
    # ========================================================

    print()
    print("=" * 70)
    print("STAGE 3 - CONTINUOUS HAND MOTION")
    print("=" * 70)

    print()

    print(
        "Move your hand slowly:"
    )

    print()

    print(
        "60 cm  →  50 cm  →  40 cm  →  30 cm  →  20 cm"
    )

    print()

    print(
        "Then move it back:"
    )

    print()

    print(
        "20 cm  →  30 cm  →  40 cm  →  50 cm  →  60 cm"
    )

    print()

    print(
        "Take approximately 15-20 seconds."
    )

    print(
        "The exact distance does NOT need to be perfect."
    )

    print()

    input(
        "Press ENTER to start motion tracking..."
    )

    print()

    print(
        "Tracking... (printing every 5 samples)"
    )

    print()

    motion_results = []

    for i in range(
        MOTION_SAMPLES
    ):

        result = measure()

        motion_results.append(
            result
        )

        save_result(
            "motion",
            -1,
            i + 1,
            result
        )


        # Baseline-relative changes

        rms_change = percent_change(
            baseline_avg["rms"],
            result["rms"]
        )

        corr_change = percent_change(
            baseline_avg["correlation"],
            result["correlation"]
        )

        late_change = percent_change(
            baseline_avg["late_peak"],
            result["late_peak"]
        )

        ratio_change = percent_change(
            baseline_avg["response_ratio"],
            result["response_ratio"]
        )


        if (
            (i + 1) % 5 == 0
            or
            i == 0
        ):

            print(
                f"{i + 1:03d} | "
                f"RMS {rms_change:+7.1f}% | "
                f"Corr {corr_change:+7.1f}% | "
                f"Late {late_change:+7.1f}% | "
                f"Ratio {ratio_change:+7.1f}%"
            )

        time.sleep(
            0.05
        )


    # ========================================================
    # MOTION STATISTICS
    # ========================================================

    motion_avg = average_results(
        motion_results
    )

    motion_std = standard_deviations(
        motion_results
    )

    print_statistics(
        "MOTION STATISTICS",
        motion_avg,
        motion_std
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("V2.2 EXPERIMENT COMPLETE")
    print("=" * 70)

    print()

    print(
        "CSV saved to:"
    )

    print(
        os.path.abspath(RESULT_FILE)
    )

    print()

    print(
        "The important data are:"
    )

    print(
        "1. Baseline statistics"
    )

    print(
        "2. 20 / 40 / 60 cm statistics"
    )

    print(
        "3. Motion measurements"
    )

    print()

    print(
        "Do NOT interpret the numbers as distance yet."
    )

    print(
        "The next step is to determine whether the acoustic"
    )

    print(
        "features change consistently with hand position."
    )

    print()


except KeyboardInterrupt:

    print()
    print()
    print(
        "Experiment stopped by user."
    )


except Exception as e:

    print()
    print("=" * 70)
    print("ERROR")
    print("=" * 70)
    print(e)


finally:

    sd.stop()

    csv_file.close()

    print()
    print(
        "Audio stopped."
    )