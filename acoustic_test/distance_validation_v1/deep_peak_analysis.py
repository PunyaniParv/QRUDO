"""
SARV DISTANCE VALIDATION v1 - DEEP PEAK IDENTITY ANALYSIS
=========================================================
Determines whether the distance-correlated peak is the same
physical acoustic component across distances, or whether the
algorithm switches between different peaks.

Also examines the raw WAV/correlation response around expected
delays in detail.
"""

import os
import csv
import sys
import warnings
from collections import defaultdict

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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_ROOT = os.path.join(BASE_DIR, "audio")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "analysis", "results")
INVESTIGATION_DIR = os.path.join(BASE_DIR, "investigation")
os.makedirs(INVESTIGATION_DIR, exist_ok=True)

SAMPLE_RATE = 44100
CHIRP_LOW = 7500
CHIRP_HIGH = 8500
CHIRP_DURATION = 0.100
FILTER_LOW = 6000
FILTER_HIGH = 11000
ECHO_MIN_DELAY_MS = 0.60
ECHO_MAX_DELAY_MS = 18.0
PEAK_DISTANCE_MS = 0.25
PEAK_RELATIVE_THRESHOLD = 0.18
MAX_CANDIDATE_PEAKS = 12
SUBSAMPLE_INTERPOLATION = True
FIRST_CHIRP_TIME = 0.15
CHIRP_INTERVAL = 0.250
EXPECTED_CHIRPS = 7
SPEED_OF_SOUND = 343.0
DETECTION_TOLERANCE_MS = 0.50

CONDITION_DISTANCE = {
    "baseline": 0,
    "10cm": 10,
    "20cm": 20,
    "30cm": 30,
    "40cm": 40,
    "50cm": 50
}

CONDITION_ORDER = [
    "baseline", "10cm", "20cm", "30cm", "40cm", "50cm"
]


def expected_delay_ms(dist_cm):
    if dist_cm <= 0:
        return None
    return 2.0 * dist_cm / 100.0 / SPEED_OF_SOUND * 1000.0


def generate_reference_chirp():
    n = int(CHIRP_DURATION * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    k = (CHIRP_HIGH - CHIRP_LOW) / CHIRP_DURATION
    phase = 2.0 * np.pi * (CHIRP_LOW * t + 0.5 * k * t * t)
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
    sos = butter(6, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def matched_filter(signal, reference):
    reference = reference - np.mean(reference)
    signal = signal - np.mean(signal)
    corr = correlate(signal, reference, mode="valid", method="fft")
    ref_norm = np.linalg.norm(reference)
    ref_len = len(reference)
    squared = signal ** 2
    cum = np.concatenate([[0.0], np.cumsum(squared)])
    window_energy = cum[ref_len:] - cum[:-ref_len]
    denominator = np.sqrt(np.maximum(window_energy, 1e-18)) * ref_norm + 1e-12
    return corr / denominator


def refine_peak_subsample(values, index):
    index = int(index)
    if not SUBSAMPLE_INTERPOLATION:
        return float(index)
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    y1 = float(values[index - 1])
    y2 = float(values[index])
    y3 = float(values[index + 1])
    denom = (y1 - 2.0 * y2 + y3)
    if abs(denom) < 1e-12:
        return float(index)
    offset = 0.5 * (y1 - y3) / denom
    offset = np.clip(offset, -0.5, 0.5)
    return float(index + offset)


def extract_candidate_peaks(mf, expected_index):
    min_offset = int(ECHO_MIN_DELAY_MS / 1000.0 * SAMPLE_RATE)
    max_offset = int(ECHO_MAX_DELAY_MS / 1000.0 * SAMPLE_RATE)
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
    min_distance = max(1, int(PEAK_DISTANCE_MS / 1000.0 * SAMPLE_RATE))
    threshold = maximum * PEAK_RELATIVE_THRESHOLD
    peaks, properties = find_peaks(region, height=threshold, distance=min_distance)
    if len(peaks) == 0:
        strongest = int(np.argmax(region))
        peaks = np.array([strongest])
    candidates = []
    for peak in peaks:
        absolute_index = start + int(peak)
        refined = refine_peak_subsample(np.abs(mf), absolute_index)
        delay_samples = refined - expected_index
        delay_ms = delay_samples / SAMPLE_RATE * 1000.0
        strength = float(region[peak])
        candidates.append({"delay_ms": float(delay_ms), "strength": strength})
    candidates.sort(key=lambda x: x["strength"], reverse=True)
    return candidates[:MAX_CANDIDATE_PEAKS]


def expected_chirp_times():
    return np.array(
        [FIRST_CHIRP_TIME + i * CHIRP_INTERVAL for i in range(EXPECTED_CHIRPS)],
        dtype=np.float64
    )


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


def load_metadata():
    records = []
    if not os.path.exists(METADATA_FILE):
        return records
    with open(METADATA_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def load_peak_report():
    path = os.path.join(RESULTS_DIR, "distance_peak_report.csv")
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "condition": row["condition"],
                "distance_cm": int(row["distance_cm"]),
                "wav_file": row["wav_file"],
                "chirp_index": int(row["chirp_index"]),
                "delay_ms": float(row["delay_ms"]),
                "strength": float(row["strength"]),
                "snr_db": float(row["snr_db"]),
                "near_expected": int(row["near_expected"])
            })
    return rows


print("=" * 78)
print("SARV DISTANCE VALIDATION v1 - DEEP PEAK IDENTITY ANALYSIS")
print("=" * 78)

metadata = load_metadata()
peak_rows = load_peak_report()

# ================================================================
# A. TRACK THE SAME PEAK ACROSS DISTANCES
# ================================================================
print("\n" + "=" * 78)
print("A. PEAK TRACKING ACROSS DISTANCES")
print("=" * 78)

# For each condition, find the delay of the strongest peak near expected.
# Then check: is this the same delay as in the baseline?
# If the hand reflection is real, the peak should be at a delay that
# is NOT present in the baseline (or at least much stronger).

print("\nFor each condition, the strongest peak near expected per chirp:")
print("(delay, strength, rank among 12 candidates)")

# Collect per-chirp strongest-near-expected peaks
peak_identity = defaultdict(list)
for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    expected = expected_delay_ms(CONDITION_DISTANCE[cond])
    if expected is None:
        continue
    by_chirp = defaultdict(list)
    for p in peak_rows:
        if p["condition"] == cond:
            by_chirp[(p["wav_file"], p["chirp_index"])].append(p)
    for (wav, chirp), peaks in sorted(by_chirp.items()):
        sorted_peaks = sorted(peaks, key=lambda x: x["strength"], reverse=True)
        near = [p for p in sorted_peaks if abs(p["delay_ms"] - expected) <= DETECTION_TOLERANCE_MS]
        if near:
            strongest_near = near[0]
            rank = sorted_peaks.index(strongest_near) + 1
            peak_identity[cond].append({
                "wav": wav,
                "chirp": chirp,
                "delay_ms": strongest_near["delay_ms"],
                "strength": strongest_near["strength"],
                "rank": rank
            })

# Check if the delay of the strongest near-expected peak is consistent
# across chirps within a condition
print("\nDelay consistency of strongest near-expected peak within condition:")
for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    peaks = peak_identity.get(cond, [])
    if not peaks:
        continue
    delays = np.array([p["delay_ms"] for p in peaks])
    print(f"  {cond:<6}: n={len(delays)} "
          f"delay_mean={np.mean(delays):.3f} "
          f"delay_std={np.std(delays):.3f} "
          f"delay_range={np.max(delays)-np.min(delays):.3f} ms")

# Check if the same delay appears in baseline
print("\nDo the strongest near-expected delays also appear in baseline?")
baseline_delays = np.array([p["delay_ms"] for p in peak_rows if p["condition"] == "baseline"])
for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    peaks = peak_identity.get(cond, [])
    if not peaks:
        continue
    delays = np.array([p["delay_ms"] for p in peaks])
    # For each unique delay, check if baseline has a peak within 0.25ms
    unique_delays = np.unique(np.round(delays, 2))
    overlap_count = 0
    for d in unique_delays:
        if np.any(np.abs(baseline_delays - d) <= 0.25):
            overlap_count += 1
    print(f"  {cond:<6}: {overlap_count}/{len(unique_delays)} unique delays "
          f"also in baseline (within 0.25ms)")

# ================================================================
# B. RAW WAV / CORRELATION RESPONSE DETAIL
# ================================================================
print("\n" + "=" * 78)
print("B. RAW CORRELATION RESPONSE DETAIL")
print("=" * 78)

# Load representative recordings and plot the full matched-filter
# response for each condition, showing all 7 chirps.
print("\nLoading representative recordings for detailed correlation plots...")

rep_paths = {}
for cond in CONDITION_ORDER:
    recs = [m for m in metadata if m["condition"].strip().lower() == cond]
    if not recs:
        continue
    rec = recs[0]
    path = os.path.join(AUDIO_ROOT, cond, rec["wav_file"])
    if os.path.exists(path):
        rep_paths[cond] = path

# Plot full matched-filter response for each condition
fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
cond_colors = {
    "baseline": "gray", "10cm": "tab:blue", "20cm": "tab:green",
    "30cm": "tab:orange", "40cm": "tab:red", "50cm": "tab:purple"
}

for idx, cond in enumerate(CONDITION_ORDER):
    if cond not in rep_paths:
        continue
    audio, sr = load_audio(rep_paths[cond])
    filtered = bandpass(audio)
    mf = matched_filter(filtered, REFERENCE_CHIRP)
    times_ms = np.arange(len(mf)) / SAMPLE_RATE * 1000.0
    axes[idx].plot(times_ms, np.abs(mf), color=cond_colors[cond], alpha=0.7)
    axes[idx].set_ylabel(f"{cond}")
    axes[idx].grid(True, alpha=0.3)
    # Mark expected delay
    expected = expected_delay_ms(CONDITION_DISTANCE[cond])
    if expected is not None:
        axes[idx].axvline(expected, color="red", linestyle="--", alpha=0.7, label=f"expected={expected:.2f}ms")
        axes[idx].legend(loc="upper right", fontsize=8)

axes[0].set_title("Full Matched-Filter Response (rep 1, all chirps)")
axes[-1].set_xlabel("Time (ms)")
plt.tight_layout()
plt.savefig(os.path.join(INVESTIGATION_DIR, "full_mf_response_all_conditions.png"), dpi=150)
plt.close()
print(f"Saved: {os.path.join(INVESTIGATION_DIR, 'full_mf_response_all_conditions.png')}")

# Zoom into 0-6ms region
fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
for idx, cond in enumerate(CONDITION_ORDER):
    if cond not in rep_paths:
        continue
    audio, sr = load_audio(rep_paths[cond])
    filtered = bandpass(audio)
    mf = matched_filter(filtered, REFERENCE_CHIRP)
    times_ms = np.arange(len(mf)) / SAMPLE_RATE * 1000.0
    axes[idx].plot(times_ms, np.abs(mf), color=cond_colors[cond], alpha=0.7)
    axes[idx].set_ylabel(f"{cond}")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(0, 6)
    expected = expected_delay_ms(CONDITION_DISTANCE[cond])
    if expected is not None:
        axes[idx].axvline(expected, color="red", linestyle="--", alpha=0.7, label=f"expected={expected:.2f}ms")
        axes[idx].legend(loc="upper right", fontsize=8)

axes[0].set_title("Matched-Filter Response 0-6ms (rep 1)")
axes[-1].set_xlabel("Delay (ms)")
plt.tight_layout()
plt.savefig(os.path.join(INVESTIGATION_DIR, "mf_response_0_6ms.png"), dpi=150)
plt.close()
print(f"Saved: {os.path.join(INVESTIGATION_DIR, 'mf_response_0_6ms.png')}")

# ================================================================
# C. PEAK RANK DISTRIBUTION
# ================================================================
print("\n" + "=" * 78)
print("C. PEAK RANK DISTRIBUTION")
print("=" * 78)

# For each condition, show the rank distribution of the strongest
# near-expected peak
print("\nRank of strongest near-expected peak (1=strongest overall):")
for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    peaks = peak_identity.get(cond, [])
    if not peaks:
        continue
    ranks = [p["rank"] for p in peaks]
    print(f"  {cond:<6}: ranks={sorted(ranks)}")

# ================================================================
# D. DELAY vs DISTANCE - PER-CHIRP ANALYSIS
# ================================================================
print("\n" + "=" * 78)
print("D. PER-CHIRP DELAY vs DISTANCE")
print("=" * 78)

# For each chirp index, compute the median delay of the strongest
# near-expected peak across recordings
print("\nPer-chirp median delay of strongest near-expected peak:")
print(f"{'chirp':>6}", end="")
for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    print(f" {cond:>8}", end="")
print()

for chirp in range(EXPECTED_CHIRPS):
    print(f"{chirp:>6}", end="")
    for cond in CONDITION_ORDER:
        if cond == "baseline":
            continue
        peaks = [p for p in peak_identity.get(cond, []) if p["chirp"] == chirp]
        if peaks:
            delays = [p["delay_ms"] for p in peaks]
            print(f" {np.median(delays):>8.3f}", end="")
        else:
            print(f" {'':>8}", end="")
    print()

# ================================================================
# E. BASELINE SUBTRACTION ANALYSIS
# ================================================================
print("\n" + "=" * 78)
print("E. BASELINE SUBTRACTION")
print("=" * 78)

# For each condition, compute the difference between the hand
# matched-filter response and the baseline matched-filter response.
# This shows what the hand actually adds.
print("\nComputing baseline-subtracted responses...")

# Load baseline MF (average of all baseline recordings)
baseline_mfs = []
for m in metadata:
    if m["condition"].strip().lower() == "baseline":
        path = os.path.join(AUDIO_ROOT, "baseline", m["wav_file"])
        if os.path.exists(path):
            audio, sr = load_audio(path)
            filtered = bandpass(audio)
            mf = matched_filter(filtered, REFERENCE_CHIRP)
            baseline_mfs.append(mf)

if baseline_mfs:
    # Truncate to common length
    min_len = min(len(mf) for mf in baseline_mfs)
    baseline_mfs = [mf[:min_len] for mf in baseline_mfs]
    baseline_avg = np.mean(np.array(baseline_mfs), axis=0)
    print(f"Baseline recordings averaged: {len(baseline_mfs)}")

    # For each hand condition, compute baseline-subtracted response
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    for idx, cond in enumerate(["10cm", "20cm", "30cm", "40cm", "50cm"]):
        if cond not in rep_paths:
            continue
        audio, sr = load_audio(rep_paths[cond])
        filtered = bandpass(audio)
        mf = matched_filter(filtered, REFERENCE_CHIRP)
        # Align lengths
        n = min(len(mf), len(baseline_avg))
        diff = np.abs(mf[:n]) - np.abs(baseline_avg[:n])
        times_ms = np.arange(len(diff)) / SAMPLE_RATE * 1000.0
        axes[idx].plot(times_ms, diff, color=cond_colors[cond], alpha=0.7)
        axes[idx].axhline(0, color="black", linestyle="--", alpha=0.5)
        axes[idx].set_ylabel(f"{cond}")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 6)
        expected = expected_delay_ms(CONDITION_DISTANCE[cond])
        if expected is not None:
            axes[idx].axvline(expected, color="red", linestyle="--", alpha=0.7,
                              label=f"expected={expected:.2f}ms")
            axes[idx].legend(loc="upper right", fontsize=8)

    axes[0].set_title("Baseline-Subtracted Matched-Filter Response (|hand| - |baseline|)")
    axes[-1].set_xlabel("Delay (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(INVESTIGATION_DIR, "baseline_subtracted_response.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(INVESTIGATION_DIR, 'baseline_subtracted_response.png')}")

    # Find the strongest positive difference peak near expected
    print("\nStrongest baseline-subtracted peak near expected delay:")
    for cond in ["10cm", "20cm", "30cm", "40cm", "50cm"]:
        if cond not in rep_paths:
            continue
        audio, sr = load_audio(rep_paths[cond])
        filtered = bandpass(audio)
        mf = matched_filter(filtered, REFERENCE_CHIRP)
        n = min(len(mf), len(baseline_avg))
        diff = np.abs(mf[:n]) - np.abs(baseline_avg[:n])
        expected = expected_delay_ms(CONDITION_DISTANCE[cond])
        if expected is None:
            continue
        # Search in window around expected
        min_s = int((expected - 1.0) / 1000.0 * SAMPLE_RATE)
        max_s = int((expected + 1.0) / 1000.0 * SAMPLE_RATE)
        min_s = max(0, min_s)
        max_s = min(len(diff), max_s)
        if max_s > min_s:
            window = diff[min_s:max_s]
            if len(window):
                peak_idx = np.argmax(window)
                peak_delay = (min_s + peak_idx) / SAMPLE_RATE * 1000.0
                peak_val = window[peak_idx]
                print(f"  {cond:<6}: expected={expected:.3f}ms "
                      f"strongest_diff_peak={peak_delay:.3f}ms "
                      f"value={peak_val:.6f}")

# ================================================================
# F. STRENGTH vs DISTANCE FOR THE TRACKED PEAK
# ================================================================
print("\n" + "=" * 78)
print("F. STRENGTH OF TRACKED PEAK vs DISTANCE")
print("=" * 78)

# If the hand reflection is real, its strength should decrease
# with distance (inverse square law).
print("\nStrength of strongest near-expected peak per condition:")
for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    peaks = peak_identity.get(cond, [])
    if not peaks:
        continue
    strengths = np.array([p["strength"] for p in peaks])
    print(f"  {cond:<6}: n={len(strengths)} "
          f"mean={np.mean(strengths):.4f} "
          f"std={np.std(strengths):.4f} "
          f"median={np.median(strengths):.4f}")

# ================================================================
# G. SUMMARY
# ================================================================
print("\n" + "=" * 78)
print("DEEP PEAK IDENTITY ANALYSIS COMPLETE")
print("=" * 78)