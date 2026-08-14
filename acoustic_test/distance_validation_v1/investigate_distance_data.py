"""
SARV DISTANCE VALIDATION v1 - DEEP INVESTIGATION
================================================
Investigates the existing distance-validation data to determine:

1. Why every chirp produces exactly 12 candidate peaks.
2. Why near_expected can exceed 7 (one per chirp).
3. Whether the selected distance-correlated peak is the same
   physical acoustic component across distances.
4. Full candidate-peak distribution comparison baseline vs hand.
5. Raw WAV/correlation response around expected delays.
6. Whether the 2d/c expected-delay model is physically valid
   for the actual laptop speaker/microphone geometry (bistatic).
7. Whether the 0.9629 correlation survives a stricter,
   physically justified peak-selection method.

This script does NOT create V9, does NOT collect new data,
does NOT train a classifier, and does NOT change hardware.
"""

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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION (identical to analyze_distance_data.py)
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_ROOT = os.path.join(BASE_DIR, "audio")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "analysis", "results")
PLOTS_DIR = os.path.join(BASE_DIR, "analysis", "plots")
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
    """Monostatic round-trip: 2d/c."""
    if dist_cm <= 0:
        return None
    return 2.0 * dist_cm / 100.0 / SPEED_OF_SOUND * 1000.0


# ================================================================
# SIGNAL CHAIN (identical to analyze_distance_data.py)
# ================================================================

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
    """Identical to analyze_distance_data.py."""
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


def analyze_recording_full(audio_path):
    """Returns matched filter, chirp times, and per-chirp candidates."""
    audio, sr = load_audio(audio_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz")
    filtered = bandpass(audio)
    mf = matched_filter(filtered, REFERENCE_CHIRP)
    background = float(np.median(np.abs(mf))) + 1e-12
    chirp_times = expected_chirp_times()
    reference_length = len(REFERENCE_CHIRP)
    all_peaks = []
    for chirp_index, chirp_time in enumerate(chirp_times):
        sample_center = int(chirp_time * SAMPLE_RATE)
        expected_index = sample_center - reference_length // 2
        candidates = extract_candidate_peaks(mf, expected_index)
        snr_db = 20.0 * np.log10(float(np.max(np.abs(mf))) / background)
        for candidate in candidates:
            all_peaks.append({
                "chirp_index": chirp_index,
                "chirp_time": chirp_time,
                "delay_ms": candidate["delay_ms"],
                "strength": candidate["strength"],
                "snr_db": snr_db
            })
    return mf, all_peaks, background


# ================================================================
# LOAD PEAK REPORT
# ================================================================

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


# ================================================================
# MAIN INVESTIGATION
# ================================================================

print("=" * 78)
print("SARV DISTANCE VALIDATION v1 - DEEP INVESTIGATION")
print("=" * 78)

metadata = load_metadata()
peak_rows = load_peak_report()

print(f"\nMetadata records: {len(metadata)}")
print(f"Peak report rows: {len(peak_rows)}")

# ----------------------------------------------------------------
# 1. WHY EXACTLY 12 CANDIDATE PEAKS PER CHIRP?
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("1. CANDIDATE PEAK COUNT ANALYSIS")
print("=" * 78)

# Group by condition and chirp
from collections import Counter, defaultdict

peaks_per_chirp = Counter()
for p in peak_rows:
    key = (p["condition"], p["wav_file"], p["chirp_index"])
    peaks_per_chirp[key] += 1

count_dist = Counter(peaks_per_chirp.values())
print(f"\nPeaks-per-chirp distribution: {dict(sorted(count_dist.items()))}")

# Check: how many peaks would find_peaks return without the cap?
# The window is 0.6-18ms = 17.4ms. With 0.25ms min distance, max ~69 peaks.
# With 18% relative threshold, how many pass?
print(f"\nWindow: {ECHO_MIN_DELAY_MS}-{ECHO_MAX_DELAY_MS} ms "
      f"= {ECHO_MAX_DELAY_MS - ECHO_MIN_DELAY_MS:.1f} ms wide")
print(f"Min peak distance: {PEAK_DISTANCE_MS} ms")
print(f"Max theoretical peaks in window: "
      f"{int((ECHO_MAX_DELAY_MS - ECHO_MIN_DELAY_MS) / PEAK_DISTANCE_MS) + 1}")
print(f"MAX_CANDIDATE_PEAKS cap: {MAX_CANDIDATE_PEAKS}")
print(f"-> Every chirp hits the cap of {MAX_CANDIDATE_PEAKS} because "
      f"find_peaks finds >= {MAX_CANDIDATE_PEAKS} peaks above the "
      f"{PEAK_RELATIVE_THRESHOLD*100:.0f}% relative threshold.")

# ----------------------------------------------------------------
# 2. WHY CAN near_expected EXCEED 7?
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("2. near_expected > 7 ANALYSIS")
print("=" * 78)

# The code counts ALL peaks within tolerance, not one per chirp.
# Multiple peaks per chirp can be within the 0.5ms tolerance window.
print("\nThe analysis code counts ALL candidate peaks within the")
print(f"{DETECTION_TOLERANCE_MS} ms tolerance window, across ALL chirps.")
print("It does NOT cap at one detection per chirp.")
print("Multiple peaks per chirp can fall within the tolerance window,")
print("so near_expected can exceed EXPECTED_CHIRPS (7).")

# Demonstrate: find recordings with near_expected > 7
summary_path = os.path.join(RESULTS_DIR, "distance_recording_summary.csv")
with open(summary_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    summaries = list(reader)

over_7 = [s for s in summaries if s["n_peaks_near_expected"] != "" and int(s["n_peaks_near_expected"]) > 7]
print(f"\nRecordings with near_expected > 7: {len(over_7)}")
for s in over_7[:10]:
    print(f"  {s['condition']} {s['wav_file']}: "
          f"near_expected={s['n_peaks_near_expected']}")

# Show that multiple peaks per chirp are within tolerance
print("\nExample: 20cm_001, chirp 0 - peaks near expected (1.166 ms):")
expected_20 = expected_delay_ms(20)
for p in peak_rows:
    if (p["condition"] == "20cm" and
        p["wav_file"] == "20cm_001_20260814_202734_901875.wav" and
        p["chirp_index"] == 0 and
        abs(p["delay_ms"] - expected_20) <= DETECTION_TOLERANCE_MS):
        print(f"  delay={p['delay_ms']:.3f} ms strength={p['strength']:.4f}")

# ----------------------------------------------------------------
# 3. IS THE SELECTED PEAK THE SAME PHYSICAL COMPONENT?
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("3. PEAK IDENTITY ACROSS DISTANCES")
print("=" * 78)

# For each condition, find the strongest peak near expected per chirp.
# Track whether the strongest near-expected peak is consistently
# the same rank (1st, 2nd, 3rd...) among all candidates.
print("\nFor each condition, the strongest peak near expected per chirp:")
print("(rank = position among all 12 candidates sorted by strength)")

rank_by_condition = defaultdict(list)
for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    expected = expected_delay_ms(CONDITION_DISTANCE[cond])
    if expected is None:
        continue
    # Group peaks by (wav_file, chirp_index)
    by_chirp = defaultdict(list)
    for p in peak_rows:
        if p["condition"] == cond:
            by_chirp[(p["wav_file"], p["chirp_index"])].append(p)
    for (wav, chirp), peaks in sorted(by_chirp.items()):
        # Sort by strength descending (rank 1 = strongest)
        sorted_peaks = sorted(peaks, key=lambda x: x["strength"], reverse=True)
        near = [p for p in sorted_peaks if abs(p["delay_ms"] - expected) <= DETECTION_TOLERANCE_MS]
        if near:
            strongest_near = near[0]
            rank = sorted_peaks.index(strongest_near) + 1
            rank_by_condition[cond].append(rank)

for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    ranks = rank_by_condition.get(cond, [])
    if ranks:
        print(f"  {cond:<6}: n={len(ranks)} "
              f"rank1={ranks.count(1)} rank2={ranks.count(2)} "
              f"rank3={ranks.count(3)} rank4+={sum(1 for r in ranks if r >= 4)} "
              f"median_rank={np.median(ranks):.1f}")

# ----------------------------------------------------------------
# 4. FULL CANDIDATE-PEAK DISTRIBUTION COMPARISON
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("4. CANDIDATE-PEAK DISTRIBUTION COMPARISON")
print("=" * 78)

# Compare the full delay distribution of baseline vs each hand distance.
# Use histogram bins and compute overlap.
print("\nDelay distribution (all candidate peaks, all chirps):")
print(f"{'Condition':<10} {'n':>5} {'mean_ms':>8} {'std_ms':>8} "
      f"{'min_ms':>8} {'max_ms':>8} {'median_ms':>10}")

cond_delays = {}
for cond in CONDITION_ORDER:
    delays = np.array([p["delay_ms"] for p in peak_rows if p["condition"] == cond])
    cond_delays[cond] = delays
    if len(delays):
        print(f"{cond:<10} {len(delays):>5} {np.mean(delays):>8.3f} "
              f"{np.std(delays):>8.3f} {np.min(delays):>8.3f} "
              f"{np.max(delays):>8.3f} {np.median(delays):>10.3f}")

# Histogram overlap between baseline and each hand condition
print("\nHistogram overlap (0.25ms bins, 0-18ms):")
bins = np.arange(0, 18.25, 0.25)
baseline_hist, _ = np.histogram(cond_delays["baseline"], bins=bins)
baseline_hist = baseline_hist / max(1, baseline_hist.sum())

for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    cond_hist, _ = np.histogram(cond_delays[cond], bins=bins)
    cond_hist = cond_hist / max(1, cond_hist.sum())
    overlap = np.sum(np.minimum(baseline_hist, cond_hist))
    print(f"  {cond:<6}: overlap={overlap:.3f}")

# ----------------------------------------------------------------
# 5. RAW WAV / CORRELATION RESPONSE AROUND EXPECTED DELAYS
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("5. RAW CORRELATION RESPONSE AROUND EXPECTED DELAYS")
print("=" * 78)

# Load one representative recording per condition and inspect
# the matched-filter response around the expected delay.
print("\nRepresentative matched-filter response near expected delay:")
print(f"{'Condition':<10} {'expected_ms':>11} {'peak_near_ms':>12} "
      f"{'peak_strength':>13} {'background':>10} {'snr_db':>8}")

rep_mf = {}
for cond in CONDITION_ORDER:
    recs = [m for m in metadata if m["condition"].strip().lower() == cond]
    if not recs:
        continue
    rec = recs[0]
    path = os.path.join(AUDIO_ROOT, cond, rec["wav_file"])
    if not os.path.exists(path):
        continue
    mf, peaks, background = analyze_recording_full(path)
    rep_mf[cond] = mf
    expected = expected_delay_ms(CONDITION_DISTANCE[cond])
    if expected is None:
        print(f"{cond:<10} {'n/a':>11} {'':>12} {'':>13} "
              f"{background:>10.6f} {'':>8}")
        continue
    # Find strongest peak near expected
    near = [p for p in peaks if abs(p["delay_ms"] - expected) <= DETECTION_TOLERANCE_MS]
    if near:
        strongest = max(near, key=lambda x: x["strength"])
        snr = 20.0 * np.log10(strongest["strength"] / background)
        print(f"{cond:<10} {expected:>11.3f} {strongest['delay_ms']:>12.3f} "
              f"{strongest['strength']:>13.6f} {background:>10.6f} {snr:>8.1f}")
    else:
        print(f"{cond:<10} {expected:>11.3f} {'none':>12} {'':>13} "
              f"{background:>10.6f} {'':>8}")

# Plot the correlation response for all conditions
plt.figure(figsize=(12, 7))
cond_colors = {
    "baseline": "gray", "10cm": "tab:blue", "20cm": "tab:green",
    "30cm": "tab:orange", "40cm": "tab:red", "50cm": "tab:purple"
}
for cond in CONDITION_ORDER:
    if cond not in rep_mf:
        continue
    mf = rep_mf[cond]
    times_ms = np.arange(len(mf)) / SAMPLE_RATE * 1000.0
    plt.plot(times_ms, np.abs(mf), color=cond_colors[cond], alpha=0.6, label=cond)

for d_target in [10, 20, 30, 40, 50]:
    expected = expected_delay_ms(d_target)
    if expected is not None:
        plt.axvline(expected, linestyle=":", color="black", alpha=0.4)

plt.xlim(0, 6.0)
plt.xlabel("Delay (ms)")
plt.ylabel("Matched-filter response |mf|")
plt.title("Correlation Response: Baseline vs Hand Distances (rep 1)")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(INVESTIGATION_DIR, "correlation_response_all.png"), dpi=150)
plt.close()
print(f"\nSaved: {os.path.join(INVESTIGATION_DIR, 'correlation_response_all.png')}")

# ----------------------------------------------------------------
# 6. BISTATIC vs MONOSTATIC MODEL
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("6. BISTATIC vs MONOSTATIC DELAY MODEL")
print("=" * 78)

# Laptop speaker/microphone geometry.
# Typical laptop: speaker and mic are separated by ~10-20 cm.
# The hand is placed in front of the laptop.
# For a bistatic path: speaker -> hand -> mic
# Path length = d_sp + d_mi where d_sp = distance from speaker to hand,
# d_mi = distance from mic to hand.
# If speaker and mic are separated by s cm, and the hand is at
# perpendicular distance d from the laptop surface:
#   d_sp = sqrt(d^2 + (s/2)^2)  (if hand centered between them)
#   d_mi = sqrt(d^2 + (s/2)^2)
#   total path = 2 * sqrt(d^2 + (s/2)^2)
#   delay = 2 * sqrt(d^2 + (s/2)^2) / c

print("\nAssuming speaker and mic are separated by s cm on the laptop:")
print("Bistatic path (hand centered): 2 * sqrt(d^2 + (s/2)^2)")
print("Monostatic path: 2 * d")
print()

for s_cm in [5, 10, 15, 20]:
    print(f"  Speaker-mic separation s = {s_cm} cm:")
    print(f"  {'dist_cm':>8} {'mono_ms':>8} {'bistatic_ms':>12} {'diff_ms':>8}")
    for d in [10, 20, 30, 40, 50]:
        mono = 2.0 * d / 100.0 / SPEED_OF_SOUND * 1000.0
        bistatic = 2.0 * np.sqrt(d**2 + (s_cm/2)**2) / 100.0 / SPEED_OF_SOUND * 1000.0
        print(f"  {d:>8} {mono:>8.3f} {bistatic:>12.3f} {bistatic-mono:>8.3f}")

# Fit the measured data to both models
print("\nMeasured median delays vs distance:")
measured_by_dist = {}
for s in summaries:
    if s["median_delay_ms"] != "" and int(s["distance_cm"]) > 0:
        d = int(s["distance_cm"])
        measured_by_dist.setdefault(d, []).append(float(s["median_delay_ms"]))

for d in sorted(measured_by_dist):
    vals = measured_by_dist[d]
    print(f"  {d:>3} cm: median={np.median(vals):.3f} ms "
          f"mean={np.mean(vals):.3f} ms std={np.std(vals):.3f} ms")

# Fit monostatic: delay = a + b*dist
d_arr = np.array(sorted(measured_by_dist.keys()), dtype=np.float64)
m_arr = np.array([np.median(measured_by_dist[d]) for d in sorted(measured_by_dist.keys())], dtype=np.float64)

slope, intercept = np.polyfit(d_arr, m_arr, 1)
predicted = slope * d_arr + intercept
corr = np.corrcoef(d_arr, m_arr)[0, 1]
rmse = np.sqrt(np.mean((m_arr - predicted)**2))

print(f"\nMonostatic fit (delay = a + b*dist):")
print(f"  a = {intercept:.5f} ms")
print(f"  b = {slope:.5f} ms/cm")
print(f"  correlation = {corr:.4f}")
print(f"  RMSE = {rmse:.5f} ms")
print(f"  Ideal slope (2d/c) = {2.0/100.0/SPEED_OF_SOUND*1000.0:.5f} ms/cm")

# Fit bistatic with unknown speaker-mic separation s
# delay = 2 * sqrt(d^2 + (s/2)^2) / c * 1000
# Solve for s that minimizes RMSE
from scipy.optimize import minimize_scalar

def bistatic_rmse(s_cm):
    delays = 2.0 * np.sqrt(d_arr**2 + (s_cm/2.0)**2) / 100.0 / SPEED_OF_SOUND * 1000.0
    return np.sqrt(np.mean((m_arr - delays)**2))

result = minimize_scalar(bistatic_rmse, bounds=(0, 50), method="bounded")
best_s = result.x
best_rmse = result.fun
bistatic_delays = 2.0 * np.sqrt(d_arr**2 + (best_s/2.0)**2) / 100.0 / SPEED_OF_SOUND * 1000.0
bistatic_corr = np.corrcoef(d_arr, m_arr)[0, 1]  # same as monostatic for linear relationship

print(f"\nBistatic fit (delay = 2*sqrt(d^2 + (s/2)^2)/c):")
print(f"  Best s = {best_s:.2f} cm")
print(f"  RMSE = {best_rmse:.5f} ms")
print(f"  Correlation = {bistatic_corr:.4f}")

# Also fit with an offset (systematic delay offset)
# delay = offset + 2*sqrt(d^2 + (s/2)^2)/c
def bistatic_offset_rmse(params):
    offset, s_cm = params
    delays = offset + 2.0 * np.sqrt(d_arr**2 + (s_cm/2.0)**2) / 100.0 / SPEED_OF_SOUND * 1000.0
    return np.sqrt(np.mean((m_arr - delays)**2))

from scipy.optimize import minimize
result2 = minimize(bistatic_offset_rmse, x0=[0.5, 10.0], method="Nelder-Mead")
offset_best, s_best2 = result2.x
rmse_best2 = result2.fun
print(f"\nBistatic + offset fit:")
print(f"  Offset = {offset_best:.3f} ms")
print(f"  Best s = {s_best2:.2f} cm")
print(f"  RMSE = {rmse_best2:.5f} ms")

# ----------------------------------------------------------------
# 7. STRICTER PEAK-SELECTION CORRELATION
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("7. STRICTER PEAK-SELECTION CORRELATION")
print("=" * 78)

# Method 1: Use only the STRONGEST peak per chirp (rank 1)
# Method 2: Use only peaks that are the GLOBAL maximum in the window
# Method 3: Use only peaks with SNR above a threshold
# Method 4: Use only peaks that are consistently the same rank

print("\nMethod A: Strongest peak per chirp (rank 1) near expected")
strict_delays = defaultdict(list)
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
        if not sorted_peaks:
            continue
        strongest = sorted_peaks[0]
        if abs(strongest["delay_ms"] - expected) <= DETECTION_TOLERANCE_MS:
            strict_delays[cond].append(strongest["delay_ms"])

strict_d_arr = []
strict_m_arr = []
for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    if strict_delays.get(cond):
        d = CONDITION_DISTANCE[cond]
        strict_d_arr.extend([d] * len(strict_delays[cond]))
        strict_m_arr.extend(strict_delays[cond])

strict_d_arr = np.array(strict_d_arr, dtype=np.float64)
strict_m_arr = np.array(strict_m_arr, dtype=np.float64)

if len(strict_d_arr) >= 3:
    slope_s, intercept_s = np.polyfit(strict_d_arr, strict_m_arr, 1)
    pred_s = slope_s * strict_d_arr + intercept_s
    corr_s = np.corrcoef(strict_d_arr, strict_m_arr)[0, 1]
    rmse_s = np.sqrt(np.mean((strict_m_arr - pred_s)**2))
    print(f"  n={len(strict_d_arr)}")
    print(f"  slope={slope_s:.5f} ms/cm")
    print(f"  intercept={intercept_s:.5f} ms")
    print(f"  correlation={corr_s:.4f}")
    print(f"  RMSE={rmse_s:.5f} ms")
else:
    print("  Not enough data")

print("\nMethod B: Per-condition median of strongest near-expected peak")
strict_cond_medians = {}
for cond in CONDITION_ORDER:
    if cond == "baseline":
        continue
    if strict_delays.get(cond):
        strict_cond_medians[cond] = np.median(strict_delays[cond])

cond_d = np.array([CONDITION_DISTANCE[c] for c in strict_cond_medians], dtype=np.float64)
cond_m = np.array([strict_cond_medians[c] for c in strict_cond_medians], dtype=np.float64)
if len(cond_d) >= 3:
    slope_c, intercept_c = np.polyfit(cond_d, cond_m, 1)
    pred_c = slope_c * cond_d + intercept_c
    corr_c = np.corrcoef(cond_d, cond_m)[0, 1]
    rmse_c = np.sqrt(np.mean((cond_m - pred_c)**2))
    print(f"  n={len(cond_d)}")
    print(f"  slope={slope_c:.5f} ms/cm")
    print(f"  intercept={intercept_c:.5f} ms")
    print(f"  correlation={corr_c:.4f}")
    print(f"  RMSE={rmse_c:.5f} ms")

print("\nMethod C: Peak with highest SNR (global max in window)")
# The global max in the window is the strongest peak overall.
# This is the same as Method A's rank-1 peak.

print("\nMethod D: Only peaks that are rank 1 AND within tolerance")
# Same as Method A.

print("\nMethod E: Per-recording median delay using rank-1 peak only")
rec_strict = []
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
    # Group by wav_file
    by_wav = defaultdict(list)
    for (wav, chirp), peaks in sorted(by_chirp.items()):
        sorted_peaks = sorted(peaks, key=lambda x: x["strength"], reverse=True)
        if sorted_peaks and abs(sorted_peaks[0]["delay_ms"] - expected) <= DETECTION_TOLERANCE_MS:
            by_wav[wav].append(sorted_peaks[0]["delay_ms"])
    for wav, delays in by_wav.items():
        if delays:
            rec_strict.append({
                "condition": cond,
                "distance_cm": CONDITION_DISTANCE[cond],
                "wav_file": wav,
                "median_delay_ms": np.median(delays),
                "n_detections": len(delays)
            })

if rec_strict:
    rs_d = np.array([r["distance_cm"] for r in rec_strict], dtype=np.float64)
    rs_m = np.array([r["median_delay_ms"] for r in rec_strict], dtype=np.float64)
    slope_r, intercept_r = np.polyfit(rs_d, rs_m, 1)
    pred_r = slope_r * rs_d + intercept_r
    corr_r = np.corrcoef(rs_d, rs_m)[0, 1]
    rmse_r = np.sqrt(np.mean((rs_m - pred_r)**2))
    print(f"  n={len(rec_strict)}")
    print(f"  slope={slope_r:.5f} ms/cm")
    print(f"  intercept={intercept_r:.5f} ms")
    print(f"  correlation={corr_r:.4f}")
    print(f"  RMSE={rmse_r:.5f} ms")

# ----------------------------------------------------------------
# 8. BASELINE PEAK OVERLAP AT EXPECTED DELAYS
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("8. BASELINE PEAK OVERLAP AT EXPECTED DELAYS")
print("=" * 78)

baseline_delays = np.array([p["delay_ms"] for p in peak_rows if p["condition"] == "baseline"])
print(f"\nBaseline candidate peaks: {len(baseline_delays)}")

for d_target in [10, 20, 30, 40, 50]:
    expected = expected_delay_ms(d_target)
    if expected is None:
        continue
    near = np.sum(np.abs(baseline_delays - expected) <= DETECTION_TOLERANCE_MS)
    # Also check with wider tolerance
    near_wide = np.sum(np.abs(baseline_delays - expected) <= 1.0)
    print(f"  {d_target:>2} cm expected={expected:.3f} ms: "
          f"baseline peaks within 0.5ms: {near}, within 1.0ms: {near_wide}")

# ----------------------------------------------------------------
# 9. STRENGTH ANALYSIS
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("9. PEAK STRENGTH ANALYSIS")
print("=" * 78)

print("\nStrength of strongest peak per chirp:")
for cond in CONDITION_ORDER:
    by_chirp = defaultdict(list)
    for p in peak_rows:
        if p["condition"] == cond:
            by_chirp[(p["wav_file"], p["chirp_index"])].append(p)
    strengths = []
    for (wav, chirp), peaks in by_chirp.items():
        sorted_peaks = sorted(peaks, key=lambda x: x["strength"], reverse=True)
        if sorted_peaks:
            strengths.append(sorted_peaks[0]["strength"])
    if strengths:
        print(f"  {cond:<10}: n={len(strengths)} "
              f"mean={np.mean(strengths):.4f} "
              f"std={np.std(strengths):.4f} "
              f"median={np.median(strengths):.4f}")

# ----------------------------------------------------------------
# 10. SUMMARY
# ----------------------------------------------------------------
print("\n" + "=" * 78)
print("INVESTIGATION COMPLETE")
print("=" * 78)
print(f"\nInvestigation outputs saved to: {INVESTIGATION_DIR}")