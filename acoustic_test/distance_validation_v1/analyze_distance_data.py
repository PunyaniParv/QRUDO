import os
import csv
import sys
import warnings

import numpy as np

from scipy.io import wavfile
from scipy.signal import (
    butter,
    sosfiltfilt,
    correlate,
    find_peaks
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The console on Windows may be cp1252; keep output ASCII-safe.
try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )
except Exception:
    pass

warnings.filterwarnings("ignore")

# ================================================================
# SARV CONTROLLED-DISTANCE REFLECTION VALIDATION v1
# ANALYSIS TOOL
#
# This reuses the EXACT signal-processing chain from
#   acoustic_gesture_analyzer_v8.py
# (bandpass 6-11kHz, normalized matched filter, candidate peak
#  extraction with sub-sample interpolation).
#
# It does NOT trust the strongest peak blindly. It extracts ALL
# significant candidate peaks per chirp and compares them against:
#   - the no-hand baseline (static/environmental reflections)
#   - the expected hand round-trip delay at each known distance
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_ROOT = os.path.join(
    BASE_DIR,
    "audio"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "metadata.csv"
)

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "analysis"
)

PLOTS_DIR = os.path.join(
    ANALYSIS_DIR,
    "plots"
)

RESULTS_DIR = os.path.join(
    ANALYSIS_DIR,
    "results"
)

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ================================================================
# AUDIO CONFIGURATION — PRESERVED FROM V5/V8
# ================================================================

SAMPLE_RATE = 44100

CHIRP_LOW = 7500
CHIRP_HIGH = 8500
CHIRP_DURATION = 0.100

# Analysis parameters preserved from V8.
FILTER_LOW = 6000
FILTER_HIGH = 11000

ECHO_MIN_DELAY_MS = 0.60
ECHO_MAX_DELAY_MS = 18.0

PEAK_DISTANCE_MS = 0.25
PEAK_RELATIVE_THRESHOLD = 0.18
MAX_CANDIDATE_PEAKS = 12

SUBSAMPLE_INTERPOLATION = True

# Chirp timing preserved from V8.
FIRST_CHIRP_TIME = 0.15
CHIRP_INTERVAL = 0.250
EXPECTED_CHIRPS = 7

# Physics.
SPEED_OF_SOUND = 343.0

# Tolerance window for "peak near expected hand delay".
DETECTION_TOLERANCE_MS = 0.50

# Expected delay, per distance.
CONDITION_DISTANCE = {
    "baseline": 0,
    "10cm": 10,
    "20cm": 20,
    "30cm": 30,
    "40cm": 40,
    "50cm": 50
}

CONDITION_ORDER = [
    "baseline",
    "10cm",
    "20cm",
    "30cm",
    "40cm",
    "50cm"
]


def expected_delay_ms(dist_cm):
    # round_trip = 2*d / speed_of_sound
    if dist_cm <= 0:
        return None
    return (
        2.0 * dist_cm / 100.0
        / SPEED_OF_SOUND
        * 1000.0
    )


# ================================================================
# SIGNAL CHAIN — IDENTICAL ALGORITHMS TO V8
# ================================================================

def generate_reference_chirp():

    n = int(
        CHIRP_DURATION * SAMPLE_RATE
    )

    t = np.arange(n) / SAMPLE_RATE

    k = (
        CHIRP_HIGH - CHIRP_LOW
    ) / CHIRP_DURATION

    phase = 2.0 * np.pi * (
        CHIRP_LOW * t
        + 0.5 * k * t * t
    )

    signal = np.sin(phase)

    signal *= np.hanning(n)

    signal = signal.astype(np.float64)

    norm = np.linalg.norm(signal)

    if norm > 0:
        signal /= norm

    return signal


REFERENCE_CHIRP = generate_reference_chirp()


def bandpass(signal):

    nyquist = SAMPLE_RATE / 2.0

    low = FILTER_LOW / nyquist
    high = FILTER_HIGH / nyquist

    sos = butter(
        6,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )


def matched_filter(signal, reference):

    reference = reference - np.mean(reference)
    signal = signal - np.mean(signal)

    corr = correlate(
        signal,
        reference,
        mode="valid",
        method="fft"
    )

    ref_norm = np.linalg.norm(reference)
    ref_len = len(reference)

    squared = signal ** 2
    cum = np.concatenate([[0.0], np.cumsum(squared)])

    window_energy = (
        cum[ref_len:] - cum[:-ref_len]
    )

    denominator = (
        np.sqrt(np.maximum(window_energy, 1e-18))
        * ref_norm
        + 1e-12
    )

    return corr / denominator


def refine_peak_subsample(values, index):

    index = int(index)

    if not SUBSAMPLE_INTERPOLATION:
        return float(index)

    if index <= 0:
        return float(index)

    if index >= len(values) - 1:
        return float(index)

    y1 = float(values[index - 1])
    y2 = float(values[index])
    y3 = float(values[index + 1])

    denom = (y1 - 2.0 * y2 + y3)

    if abs(denom) < 1e-12:
        return float(index)

    offset = (0.5 * (y1 - y3) / denom)
    offset = np.clip(offset, -0.5, 0.5)

    return float(index + offset)


def extract_candidate_peaks(mf, expected_index):

    min_offset = int(
        ECHO_MIN_DELAY_MS / 1000.0 * SAMPLE_RATE
    )

    max_offset = int(
        ECHO_MAX_DELAY_MS / 1000.0 * SAMPLE_RATE
    )

    start = max(0, int(expected_index + min_offset))
    end = min(len(mf), int(expected_index + max_offset))

    if end <= start + 3:
        return []

    region = np.abs(mf[start:end])

    if len(region) < 5:
        return []

    maximum = float(np.max(region))

    if maximum <= 0:
        return []

    min_distance = max(
        1,
        int(PEAK_DISTANCE_MS / 1000.0 * SAMPLE_RATE)
    )

    threshold = maximum * PEAK_RELATIVE_THRESHOLD

    peaks, properties = find_peaks(
        region,
        height=threshold,
        distance=min_distance
    )

    if len(peaks) == 0:
        strongest = int(np.argmax(region))
        peaks = np.array([strongest])

    candidates = []

    for peak in peaks:

        absolute_index = start + int(peak)

        refined = refine_peak_subsample(
            np.abs(mf),
            absolute_index
        )

        delay_samples = refined - expected_index

        delay_ms = delay_samples / SAMPLE_RATE * 1000.0

        strength = float(region[peak])

        candidates.append(
            {
                "delay_ms": float(delay_ms),
                "strength": strength
            }
        )

    candidates.sort(
        key=lambda x: x["strength"],
        reverse=True
    )

    return candidates[:MAX_CANDIDATE_PEAKS]


def expected_chirp_times():

    return np.array(
        [
            FIRST_CHIRP_TIME + i * CHIRP_INTERVAL
            for i in range(EXPECTED_CHIRPS)
        ],
        dtype=np.float64
    )


# ================================================================
# LOAD AUDIO (same as V8)
# ================================================================

def load_audio(path):

    sr, data = wavfile.read(path)

    original_dtype = data.dtype

    data = data.astype(np.float64)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        scale = max(abs(info.min), info.max)
        if scale > 0:
            data /= scale

    return data, sr


# ================================================================
# PER-RECORDING ANALYSIS
# ================================================================

def analyze_recording(audio_path):

    audio, sr = load_audio(audio_path)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sr} Hz"
        )

    filtered = bandpass(audio)

    mf = matched_filter(filtered, REFERENCE_CHIRP)

    # Background level for SNR: median |mf| over the whole response.
    background = float(
        np.median(np.abs(mf))
    ) + 1e-12

    chirp_times = expected_chirp_times()

    reference_length = len(REFERENCE_CHIRP)

    all_peaks = []

    for chirp_index, chirp_time in enumerate(chirp_times):

        sample_center = int(chirp_time * SAMPLE_RATE)

        expected_index = (
            sample_center - reference_length // 2
        )

        candidates = extract_candidate_peaks(
            mf,
            expected_index
        )

        snr_db = 20.0 * np.log10(
            float(np.max(np.abs(mf))) / background
        )

        for candidate in candidates:

            all_peaks.append(
                {
                    "chirp_index": chirp_index,
                    "chirp_time": chirp_time,
                    "delay_ms": candidate["delay_ms"],
                    "strength": candidate["strength"],
                    "snr_db": snr_db
                }
            )

    return all_peaks, background


# ================================================================
# LOAD METADATA
# ================================================================

def load_metadata():

    records = []

    if not os.path.exists(METADATA_FILE):
        return records

    with open(
        METADATA_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            records.append(row)

    return records


# ================================================================
# MAIN
# ================================================================

print("=" * 78)
print("SARV CONTROLLED-DISTANCE REFLECTION VALIDATION v1")
print("ANALYSIS TOOL")
print("=" * 78)

print()
print("Audio root:")
print(AUDIO_ROOT)

print()
print("Analysis output:")
print(ANALYSIS_DIR)

metadata = load_metadata()

print()
print(f"Metadata records: {len(metadata)}")

if not metadata:

    print()
    print("ERROR: No metadata found.")
    print("Run record_distance_data.py first.")
    sys.exit(1)


# ----------------------------------------------------------------
# Group recordings by condition
# ----------------------------------------------------------------

recordings_by_condition = {}

for row in metadata:

    cond = row.get("condition", "").strip().lower()

    if cond not in CONDITION_DISTANCE:
        continue

    filename = row.get("wav_file", "").strip()

    if not filename:
        continue

    path = os.path.join(
        AUDIO_ROOT,
        cond,
        filename
    )

    if not os.path.exists(path):
        continue

    recordings_by_condition.setdefault(cond, []).append(
        {
            "condition": cond,
            "distance_cm": CONDITION_DISTANCE[cond],
            "filename": filename,
            "path": path
        }
    )

# ----------------------------------------------------------------
# Analyze every recording
# ----------------------------------------------------------------

print()
print("=" * 78)
print("PER-RECORDING PEAK ANALYSIS")
print("=" * 78)

peak_rows = []       # one row per peak per chirp
recording_summaries = []  # one row per recording

for cond in CONDITION_ORDER:

    recs = recordings_by_condition.get(cond, [])

    print()
    print(f"[{cond.upper()}] recordings: {len(recs)}")

    for rec in recs:

        try:

            peaks, background = analyze_recording(rec["path"])

            rec_expected = expected_delay_ms(rec["distance_cm"])

            near_expected = []

            for p in peaks:

                is_near = (
                    rec_expected is not None
                    and abs(p["delay_ms"] - rec_expected)
                    <= DETECTION_TOLERANCE_MS
                )

                peak_rows.append(
                    {
                        "condition": cond,
                        "distance_cm": rec["distance_cm"],
                        "wav_file": rec["filename"],
                        "chirp_index": p["chirp_index"],
                        "delay_ms": p["delay_ms"],
                        "strength": p["strength"],
                        "snr_db": p["snr_db"],
                        "near_expected": int(is_near)
                    }
                )

                if is_near:
                    near_expected.append(p)

            n_chirps = EXPECTED_CHIRPS
            n_peaks = len(peaks)

            detection_rate = (
                len(near_expected) / n_chirps
            )

            near_delays = (
                np.array(
                    [p["delay_ms"] for p in near_expected]
                )
                if near_expected
                else np.array([])
            )

            near_strengths = (
                np.array(
                    [p["strength"] for p in near_expected]
                )
                if near_expected
                else np.array([])
            )

            recording_summaries.append(
                {
                    "condition": cond,
                    "distance_cm": rec["distance_cm"],
                    "wav_file": rec["filename"],
                    "n_chirps": n_chirps,
                    "n_candidate_peaks": n_peaks,
                    "n_peaks_near_expected": len(near_expected),
                    "detection_rate": detection_rate,
                    "median_delay_ms": (
                        float(np.median(near_delays))
                        if len(near_delays)
                        else ""
                    ),
                    "delay_std_ms": (
                        float(np.std(near_delays))
                        if len(near_delays)
                        else ""
                    ),
                    "delay_range_ms": (
                        float(np.max(near_delays) - np.min(near_delays))
                        if len(near_delays)
                        else ""
                    ),
                    "median_strength": (
                        float(np.median(near_strengths))
                        if len(near_strengths)
                        else ""
                    ),
                    "background": background
                }
            )

            print(
                f"  {rec['filename']}: "
                f"peaks={n_peaks} "
                f"near_expected={len(near_expected)}/{n_chirps} "
                f"detection={detection_rate:.2f}"
            )

        except Exception as e:

            print(
                f"  ERROR {rec['filename']}: {e}"
            )


# ----------------------------------------------------------------
# Save raw peak rows
# ----------------------------------------------------------------

peak_csv = os.path.join(
    RESULTS_DIR,
    "distance_peak_report.csv"
)

peak_fields = [
    "condition",
    "distance_cm",
    "wav_file",
    "chirp_index",
    "delay_ms",
    "strength",
    "snr_db",
    "near_expected"
]

with open(
    peak_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=peak_fields
    )

    writer.writeheader()

    for row in peak_rows:
        writer.writerow(row)

print()
print("Peak report saved:")
print(peak_csv)


# ----------------------------------------------------------------
# Save recording summaries
# ----------------------------------------------------------------

summary_csv = os.path.join(
    RESULTS_DIR,
    "distance_recording_summary.csv"
)

summary_fields = [
    "condition",
    "distance_cm",
    "wav_file",
    "n_chirps",
    "n_candidate_peaks",
    "n_peaks_near_expected",
    "detection_rate",
    "median_delay_ms",
    "delay_std_ms",
    "delay_range_ms",
    "median_strength",
    "background"
]

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fields
    )

    writer.writeheader()

    for row in recording_summaries:
        writer.writerow(row)

print()
print("Recording summary saved:")
print(summary_csv)


# ================================================================
# CONDITION AGGREGATION (TASK 6)
# ================================================================

print()
print("=" * 78)
print("CONDITION-LEVEL METRICS")
print("=" * 78)

condition_metrics = []

for cond in CONDITION_ORDER:

    recs = [
        s
        for s in recording_summaries
        if s["condition"] == cond
    ]

    if not recs:
        continue

    expected = expected_delay_ms(CONDITION_DISTANCE[cond])

    detection_rates = np.array(
        [
            r["detection_rate"]
            for r in recs
        ],
        dtype=np.float64
    )

    medians = np.array(
        [
            float(r["median_delay_ms"])
            for r in recs
            if r["median_delay_ms"] != ""
        ],
        dtype=np.float64
    )

    stds = np.array(
        [
            float(r["delay_std_ms"])
            for r in recs
            if r["delay_std_ms"] != ""
        ],
        dtype=np.float64
    )

    ranges = np.array(
        [
            float(r["delay_range_ms"])
            for r in recs
            if r["delay_range_ms"] != ""
        ],
        dtype=np.float64
    )

    strengths = np.array(
        [
            float(r["median_strength"])
            for r in recs
            if r["median_strength"] != ""
        ],
        dtype=np.float64
    )

    n_peaks_all = np.array(
        [
            r["n_candidate_peaks"]
            for r in recs
        ],
        dtype=np.float64
    )

    # For detection, count per-recording fraction of chirps with a
    # peak within tolerance of expected.
    metrics = {
        "condition": cond,
        "distance_cm": CONDITION_DISTANCE[cond],
        "expected_delay_ms": (
            expected if expected is not None else ""
        ),
        "recordings": len(recs),
        "chirps": len(recs) * EXPECTED_CHIRPS,
        "detection_rate_mean": (
            float(np.mean(detection_rates))
            if len(detection_rates)
            else 0.0
        ),
        "detection_rate_std": (
            float(np.std(detection_rates))
            if len(detection_rates)
            else 0.0
        ),
        "median_delay_ms": (
            float(np.median(medians))
            if len(medians)
            else ""
        ),
        "delay_std_ms_mean": (
            float(np.mean(stds))
            if len(stds)
            else ""
        ),
        "delay_range_ms_mean": (
            float(np.mean(ranges))
            if len(ranges)
            else ""
        ),
        "median_strength": (
            float(np.median(strengths))
            if len(strengths)
            else ""
        ),
        "candidate_peaks_per_chirp": (
            float(np.mean(n_peaks_all) / EXPECTED_CHIRPS)
            if len(n_peaks_all)
            else 0.0
        ),
        "baseline_has_same_delay": ""
    }

    print()
    print(
        f"[{cond.upper()}] "
        f"n={metrics['recordings']} "
        f"chirps={metrics['chirps']}"
    )

    print(
        f"  Expected delay        : "
        f"{metrics['expected_delay_ms']}"
    )

    print(
        f"  Detection rate        : "
        f"{metrics['detection_rate_mean']:.3f} "
        f"±{metrics['detection_rate_std']:.3f}"
    )

    print(
        f"  Median detected delay : "
        f"{metrics['median_delay_ms']}"
    )

    print(
        f"  Delay std             : "
        f"{metrics['delay_std_ms_mean']}"
    )

    print(
        f"  Delay range           : "
        f"{metrics['delay_range_ms_mean']}"
    )

    print(
        f"  Median strength       : "
        f"{metrics['median_strength']}"
    )

    print(
        f"  Candidate peaks/chirp : "
        f"{metrics['candidate_peaks_per_chirp']:.2f}"
    )

    condition_metrics.append(metrics)


# ----------------------------------------------------------------
# Baseline overlap analysis (TASK 8)
# ----------------------------------------------------------------

print()
print("=" * 78)
print("BASELINE / STATIC REFLECTION ANALYSIS")
print("=" * 78)

# Collect all baseline peak delays.
baseline_peaks = [
    p
    for p in peak_rows
    if p["condition"] == "baseline"
]

baseline_delays = np.array(
    [p["delay_ms"] for p in baseline_peaks],
    dtype=np.float64
)

print()
print(
    f"Baseline candidate peaks (all chirps): {len(baseline_peaks)}"
)

for cond in CONDITION_ORDER:

    if cond == "baseline":
        metrics = next(
            (
                m
                for m in condition_metrics
                if m["condition"] == "baseline"
            ),
            None
        )
        if metrics:
            metrics["baseline_has_same_delay"] = "n/a"
        continue

    expected = expected_delay_ms(CONDITION_DISTANCE[cond])

    if expected is None:
        continue

    # Does a baseline peak exist near this distance's expected delay?
    baseline_near = np.sum(
        np.abs(baseline_delays - expected) <= DETECTION_TOLERANCE_MS
    )

    has_baseline_peak = (
        int(baseline_near) > 0
    )

    metrics = next(
        (
            m
            for m in condition_metrics
            if m["condition"] == cond
        ),
        None
    )

    if metrics is not None:

        metrics["baseline_has_same_delay"] = (
            "yes"
            if has_baseline_peak
            else "no"
        )

        print(
            f"  {cond:<6} expected={expected:.3f}ms | "
            f"baseline peak at same delay: "
            f"{'YES' if has_baseline_peak else 'no'}"
        )


# ----------------------------------------------------------------
# Save condition metrics
# ----------------------------------------------------------------

metrics_csv = os.path.join(
    RESULTS_DIR,
    "distance_condition_metrics.csv"
)

metrics_fields = [
    "condition",
    "distance_cm",
    "expected_delay_ms",
    "recordings",
    "chirps",
    "detection_rate_mean",
    "detection_rate_std",
    "median_delay_ms",
    "delay_std_ms_mean",
    "delay_range_ms_mean",
    "median_strength",
    "candidate_peaks_per_chirp",
    "baseline_has_same_delay"
]

with open(
    metrics_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=metrics_fields
    )

    writer.writeheader()

    for m in condition_metrics:
        writer.writerow(m)

print()
print("Condition metrics saved:")
print(metrics_csv)


# ================================================================
# DISTANCE RESPONSE TEST (TASK 7)
# ================================================================

print()
print("=" * 78)
print("DISTANCE RESPONSE TEST")
print("=" * 78)

# Build per-recording (per condition) median detected delay vs distance.
dist_measured = []
dist_actual = []

for s in recording_summaries:

    if s["median_delay_ms"] == "":
        continue

    d_actual = s["distance_cm"]
    d_measured = float(s["median_delay_ms"])

    # Only include non-baseline.
    if d_actual <= 0:
        continue

    dist_actual.append(d_actual)
    dist_measured.append(d_measured)

dist_actual = np.array(dist_actual, dtype=np.float64)
dist_measured = np.array(dist_measured, dtype=np.float64)

print()

if len(dist_actual) >= 3:

    # Linear fit measured_delay = a + b * distance
    slope, intercept = np.polyfit(
        dist_actual,
        dist_measured,
        1
    )

    predicted = slope * dist_actual + intercept

    residuals = dist_measured - predicted

    corr = float(
        np.corrcoef(dist_actual, dist_measured)[0, 1]
    )

    rmse = float(
        np.sqrt(np.mean(residuals ** 2))
    )

    print(
        f"Linear fit (delay_ms = a + b*dist_cm):"
    )

    print(
        f"  a (intercept) = {intercept:.5f} ms"
    )

    print(
        f"  b (slope)     = {slope:.5f} ms/cm"
    )

    print(
        f"  correlation   = {corr:.4f}"
    )

    print(
        f"  residuals RMS = {rmse:.5f} ms"
    )

    print()
    print(
        "Ideal slope = 2/100 / 343 * 1000 = "
        f"{2.0/100.0/SPEED_OF_SOUND*1000.0:.5f} ms/cm"
    )

else:

    slope = 0.0
    intercept = 0.0
    corr = 0.0
    rmse = 0.0
    print("Not enough hand-distance data for a fit.")

print()
print("Per-condition aggregate:")

for d_target in [10, 20, 30, 40, 50]:

    expected = expected_delay_ms(d_target)

    sel_meas = dist_measured[
        dist_actual == d_target
    ]

    if len(sel_meas) == 0:
        continue

    print(
        f"  {d_target:>2d} cm | expected={expected:.3f}ms | "
        f"measured med={np.median(sel_meas):.3f}ms | "
        f"mean={np.mean(sel_meas):.3f}ms | "
        f"std={np.std(sel_meas):.3f}ms"
    )


# ================================================================
# PLOTS (TASK 9)
# ================================================================

print()
print("=" * 78)
print("GENERATING PLOTS")
print("=" * 78)

# Color per condition.
cond_colors = {
    "baseline": "gray",
    "10cm": "tab:blue",
    "20cm": "tab:green",
    "30cm": "tab:orange",
    "40cm": "tab:red",
    "50cm": "tab:purple"
}

# --- Plot 1: Expected delay vs measured delay (median) ---
plt.figure(figsize=(9, 6))

x_expected = []
y_measured = []

for m in condition_metrics:

    if (
        m["distance_cm"] > 0
        and m["median_delay_ms"] != ""
    ):

        x_expected.append(m["expected_delay_ms"])
        y_measured.append(m["median_delay_ms"])

if x_expected:

    plt.plot(
        x_expected,
        x_expected,
        "--",
        color="black",
        label="Ideal (measured = expected)"
    )

    plt.scatter(
        x_expected,
        y_measured,
        color="tab:red",
        zorder=3
    )

    plt.xlabel("Expected round-trip delay (ms)")
    plt.ylabel("Measured median detected delay (ms)")
    plt.title(
        "Distance Validation: Expected vs Measured Delay"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "expected_vs_measured.png"),
    dpi=150
)
plt.close()

# --- Plot 2: Measured delay distribution per distance ---
plt.figure(figsize=(10, 6))

all_hand_delays = []
all_labels = []

for cond in CONDITION_ORDER:

    if cond == "baseline":
        continue

    delays = np.array(
        [
            float(s["median_delay_ms"])
            for s in recording_summaries
            if (
                s["condition"] == cond
                and s["median_delay_ms"] != ""
            )
        ],
        dtype=np.float64
    )

    if len(delays):
        all_hand_delays.append(delays)
        all_labels.append(cond)

if all_hand_delays:

    plt.boxplot(
        all_hand_delays,
        tick_labels=all_labels
    )

    plt.xlabel("Condition")
    plt.ylabel("Median detected delay per recording (ms)")
    plt.title(
        "Measured Delay Distribution per Distance"
    )

    for cond in CONDITION_ORDER:

        expected = expected_delay_ms(CONDITION_DISTANCE[cond])

        if expected is not None:

            idx = [
                i
                for i, lab in enumerate(all_labels)
                if lab == cond
            ]

            if idx:
                plt.axhline(
                    expected,
                    linestyle=":",
                    color=cond_colors[cond],
                    alpha=0.8
                )

plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "measured_delay_distribution.png"),
    dpi=150
)
plt.close()

# --- Plot 3: Candidate peak locations per distance ---
plt.figure(figsize=(11, 6))

for cond in CONDITION_ORDER:

    cond_peaks = [
        p
        for p in peak_rows
        if p["condition"] == cond
    ]

    if not cond_peaks:
        continue

    delays = np.array(
        [p["delay_ms"] for p in cond_peaks],
        dtype=np.float64
    )

    # jitter to visualize density
    jitter = np.random.uniform(-0.15, 0.15, size=len(delays))

    plt.scatter(
        delays,
        jitter + CONDITION_ORDER.index(cond),
        s=8,
        alpha=0.4,
        color=cond_colors[cond],
        label=cond
    )

# Expected hand delays as vertical dashed lines.
for d_target in [10, 20, 30, 40, 50]:

    expected = expected_delay_ms(d_target)

    if expected is not None:

        plt.axvline(
            expected,
            linestyle=":",
            color="black",
            alpha=0.5
        )

plt.xlabel("Candidate peak delay (ms)")
plt.ylabel("Condition index (jittered)")
plt.yticks(
    range(len(CONDITION_ORDER)),
    CONDITION_ORDER
)
plt.title("All Candidate Peak Locations per Condition")
plt.grid(True, axis="x", alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "candidate_peak_locations.png"),
    dpi=150
)
plt.close()

# --- Plot 4: Baseline vs hand correlation response ---
plt.figure(figsize=(11, 6))

# For one representative baseline and one representative hand
# recording, plot the matched-filter |response| near 0-6 ms.
def load_mf(path):

    audio, _ = load_audio(path)

    filtered = bandpass(audio)

    return matched_filter(filtered, REFERENCE_CHIRP)


baseline_path = (
    recordings_by_condition["baseline"][0]["path"]
    if recordings_by_condition.get("baseline")
    else None
)

if baseline_path:

    base_mf = load_mf(baseline_path)

    times_ms = (
        np.arange(len(base_mf)) / SAMPLE_RATE * 1000.0
    )

    plt.plot(
        times_ms,
        np.abs(base_mf),
        color="gray",
        alpha=0.7,
        label="Baseline (no hand)"
    )

for cond in ["10cm", "30cm", "50cm"]:

    recs = recordings_by_condition.get(cond, [])

    if not recs:
        continue

    hand_mf = load_mf(recs[0]["path"])

    times_ms = (
        np.arange(len(hand_mf)) / SAMPLE_RATE * 1000.0
    )

    plt.plot(
        times_ms,
        np.abs(hand_mf),
        color=cond_colors[cond],
        alpha=0.5,
        label=f"{cond} (rep 1)"
    )

# Expected hand delays.
for d_target in [10, 20, 30, 40, 50]:

    expected = expected_delay_ms(d_target)

    if expected is not None:

        plt.axvline(
            expected,
            linestyle=":",
            color="black",
            alpha=0.4
        )

plt.xlim(0, 6.0)
plt.xlabel("Delay (ms)")
plt.ylabel("Matched-filter response |mf|")
plt.title("Correlation Response: Baseline vs Hand Distances")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "baseline_vs_hand_response.png"),
    dpi=150
)
plt.close()

# --- Plot 5: Detection rate vs distance ---
plt.figure(figsize=(9, 5))

detect_d = []
detect_r = []

for m in condition_metrics:

    if m["distance_cm"] > 0:

        detect_d.append(m["distance_cm"])
        detect_r.append(m["detection_rate_mean"])

if detect_d:

    plt.plot(
        detect_d,
        detect_r,
        marker="o"
    )

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Distance (cm)")
    plt.ylabel("Detection rate (fraction of chirps with peak near expected)")
    plt.title("Hand Peak Detection Rate vs Distance")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "detection_rate_vs_distance.png"),
    dpi=150
)
plt.close()

# --- Plot 6: Peak SNR vs distance ---
plt.figure(figsize=(9, 5))

snr_d = []
snr_v = []

for m in condition_metrics:

    if m["distance_cm"] > 0:

        # Use the SNR of peaks near expected found in the raw report.
        near_peaks = [
            p
            for p in peak_rows
            if (
                p["condition"] == m["condition"]
                and p["near_expected"] == 1
            )
        ]

        if near_peaks:

            snr_d.append(m["distance_cm"])

            snr_v.append(
                np.median(
                    [p["snr_db"] for p in near_peaks]
                )
            )

if snr_d:

    plt.plot(
        snr_d,
        snr_v,
        marker="o"
    )

    plt.xlabel("Distance (cm)")
    plt.ylabel("Median peak SNR (dB)")
    plt.title("Hand Peak SNR vs Distance")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "snr_vs_distance.png"),
    dpi=150
)
plt.close()

# --- Plot 7: Delay error vs distance ---
plt.figure(figsize=(9, 5))

error_d = []
error_v = []

for m in condition_metrics:

    if (
        m["distance_cm"] > 0
        and m["median_delay_ms"] != ""
    ):

        expected = m["expected_delay_ms"]
        measured = m["median_delay_ms"]

        error_d.append(m["distance_cm"])

        error_v.append(measured - expected)

if error_d:

    plt.bar(
        error_d,
        error_v,
        color="tab:red",
        alpha=0.7
    )

    plt.axhline(0.0, color="black", linestyle="--")
    plt.xlabel("Distance (cm)")
    plt.ylabel("Measured delay - expected delay (ms)")
    plt.title("Delay Error vs Distance")
    plt.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "delay_error_vs_distance.png"),
    dpi=150
)
plt.close()

print()
print("Plots saved to:")
print(PLOTS_DIR)


# ================================================================
# FINAL DIAGNOSIS (TASK 10)
# ================================================================

print()
print("=" * 78)
print("FINAL DIAGNOSIS")
print("=" * 78)

# Decision logic.
# A peak is "distance-dependent" if:
#   - detection rate is high for hand conditions
#   - measured delay increases with distance
#   - baseline does NOT have a persistent peak at those delays

hand_metrics = [
    m
    for m in condition_metrics
    if m["distance_cm"] > 0
]

if not hand_metrics:

    print()
    print("INCONCLUSIVE — No hand-distance data available.")

else:

    detection_rates = [
        m["detection_rate_mean"]
        for m in hand_metrics
    ]

    mean_hand_detection = float(np.mean(detection_rates))

    # Do measured delays trend upward with distance?
    delays_trend = []

    for m in hand_metrics:

        if m["median_delay_ms"] != "":
            delays_trend.append(
                (
                    m["distance_cm"],
                    float(m["median_delay_ms"])
                )
            )

    positive_trend = False

    if len(delays_trend) >= 3:

        d_arr = np.array(
            [x[0] for x in delays_trend]
        )

        m_arr = np.array(
            [x[1] for x in delays_trend]
        )

        r = float(
            np.corrcoef(d_arr, m_arr)[0, 1]
        )

        positive_trend = r > 0.3

    baseline_blocks = all(
        m["baseline_has_same_delay"] == "no"
        for m in hand_metrics
        if m["baseline_has_same_delay"] in ("yes", "no")
    )

    print()
    print(
        f"Mean hand detection rate   : "
        f"{mean_hand_detection:.3f}"
    )

    print(
        f"Delay increases with dist  : "
        f"{'yes' if positive_trend else 'no'}"
    )

    print(
        f"Baseline has same delays   : "
        f"{'no' if baseline_blocks else 'yes (partial/at least one)'}"
    )

    print()

    if (
        mean_hand_detection >= 0.80
        and positive_trend
        and baseline_blocks
    ):

        print("A) HAND REFLECTION VALIDATED")
        print()
        print(
            "A stable candidate peak exists near the expected "
            "delay, is distinguishable from baseline, and shifts "
            "predictably with known distance."
        )

    elif mean_hand_detection < 0.50 and not positive_trend:

        print("B) HAND REFLECTION NOT VALIDATED")
        print()
        print(
            "No stable distance-dependent peak can be demonstrated "
            "with the current setup."
        )

    else:

        print("C) INCONCLUSIVE")
        print()
        print(
            "Some evidence exists, but the current experiment "
            "cannot reliably distinguish the hand reflection "
            "from environmental/static peaks."
        )


# ================================================================
# COMPLETE
# ================================================================

print()
print("=" * 78)
print("DISTANCE VALIDATION ANALYSIS COMPLETE")
print("=" * 78)

print()
print("Results:")
print(RESULTS_DIR)

print()
print("Plots:")
print(PLOTS_DIR)

print()
print("=" * 78)