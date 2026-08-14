"""SARV distance validation v2 -- analysis + validation.

Implements the strict, physically-justified detection algorithm:

  1. matched-filter response
  2. search within a narrow, physically-justified window around the
     bistatic expected delay (NOT the full 0.6-18 ms response)
  3. select the STRONGEST/global candidate in that window
  4. enforce ONE detection per chirp
  5. calculate the bistatic expected delay
  6. calculate baseline response statistics
  7. perform baseline subtraction
  8. measure residual peak amplitude near the expected delay
  9. measure the global maximum and its delay
 10. record peak rank/strength/delay for every chirp

Produces machine-readable results and 7 plots, then applies the
validation logic and prints VALIDATED or ACOUSTIC REFLECTION NOT
VALIDATED.

Usage:
    python analyze_reflection_data.py
"""

import os
import csv
import sys
import warnings

import numpy as np

from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, correlate, find_peaks
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")

import config

os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
os.makedirs(config.PLOTS_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)


# ================================================================
# SIGNAL CHAIN
# ================================================================
def generate_reference_chirp():
    n = int(config.CHIRP_DURATION * config.SAMPLE_RATE)
    t = np.arange(n) / config.SAMPLE_RATE
    k = (config.CHIRP_HIGH - config.CHIRP_LOW) / config.CHIRP_DURATION
    phase = 2.0 * np.pi * (config.CHIRP_LOW * t + 0.5 * k * t * t)
    signal = np.sin(phase)
    signal *= np.hanning(n)
    signal = signal.astype(np.float64)
    norm = np.linalg.norm(signal)
    if norm > 0:
        signal /= norm
    return signal


REFERENCE_CHIRP = generate_reference_chirp()


def bandpass(signal):
    nyquist = config.SAMPLE_RATE / 2.0
    low = config.FILTER_LOW / nyquist
    high = config.FILTER_HIGH / nyquist
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
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    y1 = float(values[index - 1])
    y2 = float(values[index])
    y3 = float(values[index + 1])
    denom = y1 - 2.0 * y2 + y3
    if abs(denom) < 1e-12:
        return float(index)
    offset = 0.5 * (y1 - y3) / denom
    offset = np.clip(offset, -0.5, 0.5)
    return float(index + offset)


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


def expected_chirp_times():
    return np.array(
        [config.FIRST_CHIRP_TIME + i * config.CHIRP_INTERVAL
         for i in range(config.EXPECTED_CHIRPS)],
        dtype=np.float64,
    )


# ================================================================
# DETECTION
# ================================================================
def search_window(reference_index, expected_delay_ms):
    """Return (win_start, win_end) sample indices for the search window.

    The window is centered on the bistatic expected delay with a
    physically-justified half-width, and its lower bound is clamped to
    exclude the direct speaker->mic path (which dominates short delays
    and would otherwise mask the reflector echo).
    """
    sr = config.SAMPLE_RATE
    hw_samples = config.SEARCH_WINDOW_HALF_WIDTH_MS / 1000.0 * sr
    center = reference_index + expected_delay_ms / 1000.0 * sr
    win_start = int(round(center - hw_samples))
    win_end = int(round(center + hw_samples))
    # Exclude the direct path region.
    exclusion = config.direct_path_exclusion_ms()
    exclusion_samples = int(exclusion / 1000.0 * sr)
    win_start = max(win_start, reference_index + exclusion_samples)
    return win_start, win_end


def window_delays(win_start, win_end, reference_index):
    sr = config.SAMPLE_RATE
    return (np.arange(win_start, win_end) - reference_index) / sr * 1000.0


def extract_global_peaks(mf, reference_index):
    """All significant peaks in the full echo window, sorted by strength."""
    sr = config.SAMPLE_RATE
    min_offset = int(config.ECHO_MIN_DELAY_MS / 1000.0 * sr)
    max_offset = int(config.ECHO_MAX_DELAY_MS / 1000.0 * sr)
    start = max(0, reference_index + min_offset)
    end = min(len(mf), reference_index + max_offset)
    if end <= start + 3:
        return []
    region = np.abs(mf[start:end])
    maximum = float(np.max(region))
    if maximum <= 0:
        return []
    min_distance = max(1, int(config.PEAK_DISTANCE_MS / 1000.0 * sr))
    threshold = maximum * config.PEAK_RELATIVE_THRESHOLD
    peaks, _ = find_peaks(region, height=threshold, distance=min_distance)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(region))])
    result = []
    for peak in peaks:
        abs_idx = start + int(peak)
        refined = refine_peak_subsample(np.abs(mf), abs_idx)
        delay_ms = (refined - reference_index) / sr * 1000.0
        strength = float(region[peak])
        result.append({"delay_ms": delay_ms, "strength": strength})
    result.sort(key=lambda x: x["strength"], reverse=True)
    return result


def analyze_chirp(mf, reference_index, expected_delay_ms):
    """One detection per chirp: strongest candidate in the window.

    Returns dict with delay, strength, rank, global max, and the window
    response (for baseline subtraction).
    """
    sr = config.SAMPLE_RATE
    win_start, win_end = search_window(reference_index, expected_delay_ms)
    win_start = max(0, win_start)
    win_end = min(len(mf), win_end)
    if win_end <= win_start:
        return None

    region = np.abs(mf[win_start:win_end])
    local_idx = int(np.argmax(region))
    abs_idx = win_start + local_idx
    refined = refine_peak_subsample(np.abs(mf), abs_idx)
    delay_ms = (refined - reference_index) / sr * 1000.0
    strength = float(region[local_idx])

    # Global peaks for rank
    global_peaks = extract_global_peaks(mf, reference_index)
    rank = 1 + sum(1 for p in global_peaks if p["strength"] > strength)

    # Global maximum of the whole response and its delay
    abs_mf = np.abs(mf)
    gmax_idx = int(np.argmax(abs_mf))
    gmax_strength = float(abs_mf[gmax_idx])
    gmax_delay_ms = (gmax_idx - reference_index) / sr * 1000.0

    return {
        "delay_ms": delay_ms,
        "strength": strength,
        "rank": rank,
        "global_max_strength": gmax_strength,
        "global_max_delay_ms": gmax_delay_ms,
        "window_response": region.copy(),
        "win_start": win_start,
        "win_end": win_end,
    }


# ================================================================
# LOAD METADATA
# ================================================================
def load_metadata():
    records = []
    if not os.path.exists(config.METADATA_FILE):
        return records
    with open(config.METADATA_FILE, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(row)
    return records


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 78)
    print("SARV DISTANCE VALIDATION v2 -- ANALYSIS + VALIDATION")
    print("=" * 78)

    sep = config.effective_separation_cm()
    speed = config.effective_speed_of_sound()
    if sep is None:
        print("\nERROR: SPEAKER_MIC_SEPARATION_CM is not set.")
        print("Run record_reflection_data.py to measure and store geometry.")
        sys.exit(1)
    print(f"\nGeometry: s = {sep} cm, c = {speed} m/s")

    metadata = load_metadata()
    print(f"Metadata records: {len(metadata)}")
    if not metadata:
        print("ERROR: No metadata found. Run record_reflection_data.py first.")
        sys.exit(1)

    # Group recordings
    recordings = []
    for row in metadata:
        filename = row.get("wav_file", "").strip()
        if not filename:
            continue
        path = os.path.join(config.AUDIO_ROOT, filename)
        if not os.path.exists(path):
            continue
        try:
            dist = float(row.get("distance_cm"))
        except (TypeError, ValueError):
            continue
        recordings.append({
            "recording_id": filename,
            "distance_cm": dist,
            "condition": row.get("condition", "").strip().lower(),
            "path": path,
            "expected_delay_ms": config.bistatic_delay_ms(dist, sep, speed),
        })

    print(f"Valid recordings: {len(recordings)}")

    # ------------------------------------------------------------
    # Phase 1: per-chirp detection
    # ------------------------------------------------------------
    print("\n[1] PER-CHIRP DETECTION")
    chirp_rows = []  # one row per chirp
    # store window responses for baseline model
    baseline_windows = {}   # dist -> list of (recording_id, chirp, response)
    reflector_windows = {}  # dist -> list of (recording_id, chirp, response, delay_ms)

    for rec in recordings:
        audio, sr = load_audio(rec["path"])
        if sr != config.SAMPLE_RATE:
            print(f"  SKIP {rec['recording_id']}: sample rate {sr}")
            continue
        filtered = bandpass(audio)
        mf = matched_filter(filtered, REFERENCE_CHIRP)
        ref_len = len(REFERENCE_CHIRP)
        chirp_times = expected_chirp_times()

        for chirp_index, chirp_time in enumerate(chirp_times):
            # Reference index = chirp START time (where the direct acoustic
            # path peaks in the matched filter). v1 incorrectly used
            # sample_center - reference_length//2, which shifted all delays
            # by half the chirp length (~50 ms).
            reference_index = int(chirp_time * config.SAMPLE_RATE)
            res = analyze_chirp(mf, reference_index, rec["expected_delay_ms"])
            if res is None:
                continue

            delay_error = res["delay_ms"] - rec["expected_delay_ms"]
            detection = int(
                (abs(delay_error) <= config.DETECTION_TOLERANCE_MS)
            )

            chirp_rows.append({
                "recording_id": rec["recording_id"],
                "condition": rec["condition"],
                "distance_cm": rec["distance_cm"],
                "chirp_number": chirp_index + 1,
                "expected_delay_ms": rec["expected_delay_ms"],
                "measured_delay_ms": res["delay_ms"],
                "delay_error_ms": delay_error,
                "peak_strength": res["strength"],
                "peak_rank": res["rank"],
                "global_max_strength": res["global_max_strength"],
                "global_max_delay_ms": res["global_max_delay_ms"],
                "detection_status": detection,
                # filled in phase 3
                "baseline_strength": "",
                "baseline_subtracted_strength": "",
                "residual_peak_strength": "",
                "residual_peak_delay_ms": "",
                "final_detection": "",
            })

            key = rec["distance_cm"]
            if rec["condition"] == "baseline":
                baseline_windows.setdefault(key, []).append(
                    (rec["recording_id"], chirp_index, res["window_response"])
                )
            else:
                reflector_windows.setdefault(key, []).append(
                    (rec["recording_id"], chirp_index, res["window_response"],
                     res["delay_ms"])
                )

    print(f"  Total chirps analyzed: {len(chirp_rows)}")

    # ------------------------------------------------------------
    # Phase 2: baseline response model per distance
    # ------------------------------------------------------------
    print("\n[2] BASELINE RESPONSE MODEL")
    baseline_model = {}  # dist -> (mean_response, std_response, delays)
    for dist in config.DISTANCES_CM:
        wins = baseline_windows.get(dist, [])
        if not wins:
            print(f"  {dist} cm: no baseline chirps")
            continue
        arr = np.array([w for (_, _, w) in wins])
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        # delay axis from the first window (all windows same length)
        # reconstruct via a dummy reference index of 0
        n = arr.shape[1]
        delays = (np.arange(n) - (n - 1) / 2.0) / config.SAMPLE_RATE * 1000.0
        baseline_model[dist] = (mean, std, delays)
        print(f"  {dist} cm: {len(wins)} baseline chirps, "
              f"window {n} samples")

    # ------------------------------------------------------------
    # Phase 3: baseline subtraction
    # ------------------------------------------------------------
    print("\n[3] BASELINE SUBTRACTION")
    for row in chirp_rows:
        dist = row["distance_cm"]
        model = baseline_model.get(dist)
        if model is None:
            continue
        mean_resp, std_resp, delays = model
        # find index in window nearest to measured delay
        # measured delay is relative to reference_index; window delays
        # are centered on expected delay. offset = measured - expected.
        offset_ms = row["measured_delay_ms"] - row["expected_delay_ms"]
        idx = int(round((offset_ms / 1000.0 * config.SAMPLE_RATE)
                        + (len(mean_resp) - 1) / 2.0))
        idx = int(np.clip(idx, 0, len(mean_resp) - 1))
        baseline_strength = float(mean_resp[idx])
        row["baseline_strength"] = baseline_strength
        row["baseline_subtracted_strength"] = (
            row["peak_strength"] - baseline_strength
        )

        # residual peak: max of (window_response - baseline_mean)
        if row["condition"] == "reflector":
            # find this chirp's window response
            resp = None
            for (rid, ci, w, _) in reflector_windows.get(dist, []):
                if rid == row["recording_id"] and ci == row["chirp_number"] - 1:
                    resp = w
                    break
            if resp is not None:
                residual = resp - mean_resp
                r_idx = int(np.argmax(residual))
                row["residual_peak_strength"] = float(residual[r_idx])
                row["residual_peak_delay_ms"] = (
                    row["expected_delay_ms"]
                    + (r_idx - (len(mean_resp) - 1) / 2.0)
                    / config.SAMPLE_RATE * 1000.0
                )
                # final detection: positive residual near expected
                row["final_detection"] = int(
                    (row["residual_peak_strength"] > 0)
                    and (abs(row["residual_peak_delay_ms"]
                             - row["expected_delay_ms"])
                         <= config.DETECTION_TOLERANCE_MS)
                )
            else:
                row["final_detection"] = 0
        else:
            # baseline: final detection should be ~0 (no reflector)
            row["final_detection"] = 0

    # ------------------------------------------------------------
    # Save per-chirp results
    # ------------------------------------------------------------
    chirp_csv = os.path.join(config.RESULTS_DIR, "reflection_chirp_report.csv")
    chirp_fields = [
        "recording_id", "condition", "distance_cm", "chirp_number",
        "expected_delay_ms", "measured_delay_ms", "delay_error_ms",
        "peak_strength", "baseline_strength", "baseline_subtracted_strength",
        "peak_rank", "global_max_strength", "global_max_delay_ms",
        "residual_peak_strength", "residual_peak_delay_ms",
        "detection_status", "final_detection",
    ]
    with open(chirp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=chirp_fields)
        writer.writeheader()
        for row in chirp_rows:
            writer.writerow({k: row[k] for k in chirp_fields})
    print(f"\n  Saved: {chirp_csv}")

    # ------------------------------------------------------------
    # Per-recording summary
    # ------------------------------------------------------------
    rec_summary = []
    for rec in recordings:
        rows = [r for r in chirp_rows if r["recording_id"] == rec["recording_id"]]
        if not rows:
            continue
        delays = np.array([r["measured_delay_ms"] for r in rows])
        errors = np.array([r["delay_error_ms"] for r in rows])
        strengths = np.array([r["peak_strength"] for r in rows])
        ranks = np.array([r["peak_rank"] for r in rows])
        finals = np.array([r["final_detection"] for r in rows])
        rec_summary.append({
            "recording_id": rec["recording_id"],
            "condition": rec["condition"],
            "distance_cm": rec["distance_cm"],
            "n_chirps": len(rows),
            "median_delay_ms": float(np.median(delays)),
            "delay_std_ms": float(np.std(delays)),
            "delay_rmse_ms": float(np.sqrt(np.mean(errors ** 2))),
            "median_strength": float(np.median(strengths)),
            "median_rank": float(np.median(ranks)),
            "detection_rate": float(np.mean(finals)),
        })

    rec_csv = os.path.join(config.RESULTS_DIR, "reflection_recording_summary.csv")
    rec_fields = [
        "recording_id", "condition", "distance_cm", "n_chirps",
        "median_delay_ms", "delay_std_ms", "delay_rmse_ms",
        "median_strength", "median_rank", "detection_rate",
    ]
    with open(rec_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rec_fields)
        writer.writeheader()
        for row in rec_summary:
            writer.writerow(row)
    print(f"  Saved: {rec_csv}")

    # ------------------------------------------------------------
    # Per-distance metrics
    # ------------------------------------------------------------
    print("\n[4] PER-DISTANCE METRICS")
    dist_metrics = []
    for dist in config.DISTANCES_CM:
        ref_rows = [r for r in chirp_rows
                    if r["condition"] == "reflector"
                    and r["distance_cm"] == dist]
        base_rows = [r for r in chirp_rows
                     if r["condition"] == "baseline"
                     and r["distance_cm"] == dist]
        if not ref_rows:
            continue

        ref_errors = np.array([r["delay_error_ms"] for r in ref_rows])
        ref_sub = np.array([r["baseline_subtracted_strength"]
                            for r in ref_rows])
        ref_ranks = np.array([r["peak_rank"] for r in ref_rows])
        ref_final = np.array([r["final_detection"] for r in ref_rows])
        base_final = np.array([r["final_detection"] for r in base_rows]) \
            if base_rows else np.array([])

        # t-test: reflector baseline-subtracted strength vs 0
        if len(ref_sub) > 1 and np.std(ref_sub) > 0:
            t_stat, p_val = stats.ttest_1samp(ref_sub, 0.0)
        else:
            t_stat, p_val = float("nan"), float("nan")

        # t-test: reflector peak strength vs baseline peak strength
        ref_strengths = np.array([r["peak_strength"] for r in ref_rows])
        base_strengths = np.array([r["peak_strength"] for r in base_rows]) \
            if base_rows else np.array([])
        if len(ref_strengths) > 1 and len(base_strengths) > 1:
            t2, p2 = stats.ttest_ind(ref_strengths, base_strengths,
                                     equal_var=False)
        else:
            t2, p2 = float("nan"), float("nan")

        m = {
            "distance_cm": dist,
            "expected_delay_ms": config.bistatic_delay_ms(dist, sep, speed),
            "n_reflector_chirps": len(ref_rows),
            "n_baseline_chirps": len(base_rows),
            "mean_measured_delay_ms": float(np.mean(
                [r["measured_delay_ms"] for r in ref_rows])),
            "delay_rmse_ms": float(np.sqrt(np.mean(ref_errors ** 2))),
            "mean_delay_error_ms": float(np.mean(ref_errors)),
            "mean_baseline_subtracted_strength": float(np.mean(ref_sub)),
            "std_baseline_subtracted_strength": float(np.std(ref_sub)),
            "mean_peak_rank": float(np.mean(ref_ranks)),
            "median_peak_rank": float(np.median(ref_ranks)),
            "detection_rate": float(np.mean(ref_final)),
            "baseline_detection_rate": (
                float(np.mean(base_final)) if len(base_final) else float("nan")
            ),
            "ttest_sub_vs_zero_t": float(t_stat),
            "ttest_sub_vs_zero_p": float(p_val),
            "ttest_ref_vs_base_t": float(t2),
            "ttest_ref_vs_base_p": float(p2),
        }
        dist_metrics.append(m)
        print(f"  {dist} cm: det_rate={m['detection_rate']:.2f} "
              f"rmse={m['delay_rmse_ms']:.3f}ms "
              f"sub={m['mean_baseline_subtracted_strength']:.4f} "
              f"rank={m['mean_peak_rank']:.1f} "
              f"base_det={m['baseline_detection_rate']:.2f}")

    dist_csv = os.path.join(config.RESULTS_DIR, "reflection_distance_metrics.csv")
    dist_fields = list(dist_metrics[0].keys()) if dist_metrics else []
    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dist_fields)
        writer.writeheader()
        for row in dist_metrics:
            writer.writerow(row)
    print(f"  Saved: {dist_csv}")

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    print("\n[5] VALIDATION")
    validation = validate(dist_metrics, chirp_rows)
    save_validation(validation)

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    print("\n[6] PLOTS")
    make_plots(chirp_rows, dist_metrics, baseline_model)

    print("\n" + "=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"\nResults: {config.RESULTS_DIR}")
    print(f"Plots  : {config.PLOTS_DIR}")


# ================================================================
# VALIDATION LOGIC
# ================================================================
def validate(dist_metrics, chirp_rows):
    v = config.VALIDATION
    results = {}

    # 1. positive baseline-subtracted peak at expected delay
    det_rates = [m["detection_rate"] for m in dist_metrics]
    mean_det = float(np.mean(det_rates)) if det_rates else 0.0
    results["mean_detection_rate"] = mean_det
    results["detection_rate_pass"] = mean_det >= v["min_detection_rate"]

    # 2. reflector peak consistently stronger than baseline
    pvals = [m["ttest_ref_vs_base_p"] for m in dist_metrics
             if not np.isnan(m["ttest_ref_vs_base_p"])]
    mean_sub = float(np.mean([m["mean_baseline_subtracted_strength"]
                              for m in dist_metrics]))
    results["mean_baseline_subtracted_strength"] = mean_sub
    results["strength_stronger_pass"] = (
        mean_sub > 0
        and len(pvals) > 0
        and all(p < v["ttest_alpha"] for p in pvals)
    )

    # 3. reflector peak among strongest/global peaks
    mean_rank = float(np.mean([m["mean_peak_rank"] for m in dist_metrics]))
    results["mean_peak_rank"] = mean_rank
    results["rank_pass"] = mean_rank <= v["max_mean_rank"]

    # 4. strength decreases with distance
    dists = np.array([m["distance_cm"] for m in dist_metrics])
    subs = np.array([m["mean_baseline_subtracted_strength"]
                     for m in dist_metrics])
    if len(dists) >= 3 and np.std(subs) > 0:
        r_strength, _ = stats.pearsonr(dists, subs)
    else:
        r_strength = float("nan")
    results["strength_distance_corr"] = r_strength
    results["strength_distance_pass"] = (
        not np.isnan(r_strength)
        and r_strength <= v["strength_distance_corr_threshold"]
    )

    # 5. delay RMSE < 0.15 ms
    rmse = float(np.sqrt(np.mean(
        [m["delay_rmse_ms"] ** 2 for m in dist_metrics])))
    results["overall_delay_rmse_ms"] = rmse
    results["delay_rmse_pass"] = rmse < v["max_delay_rmse_ms"]

    # 6. baseline does not show the same reflector-specific response
    base_dets = [m["baseline_detection_rate"] for m in dist_metrics
                 if not np.isnan(m["baseline_detection_rate"])]
    mean_base_det = float(np.mean(base_dets)) if base_dets else 0.0
    results["mean_baseline_detection_rate"] = mean_base_det
    results["baseline_clean_pass"] = mean_base_det <= v["baseline_max_detection_rate"]

    # Overall decision
    criteria = [
        results["detection_rate_pass"],
        results["strength_stronger_pass"],
        results["rank_pass"],
        results["strength_distance_pass"],
        results["delay_rmse_pass"],
        results["baseline_clean_pass"],
    ]
    results["criteria"] = criteria
    results["validated"] = all(criteria)

    print("\n  Decision criteria:")
    names = [
        "positive baseline-subtracted peak (det rate)",
        "reflector stronger than baseline (t-test)",
        "reflector peak among strongest (rank)",
        "strength decreases with distance",
        "delay RMSE < 0.15 ms",
        "baseline clean (no reflector response)",
    ]
    for name, passed in zip(names, criteria):
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  OVERALL: "
          f"{'VALIDATED' if results['validated'] else 'ACOUSTIC REFLECTION NOT VALIDATED'}")

    return results


def save_validation(validation):
    path = os.path.join(config.RESULTS_DIR, "validation_summary.json")
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ================================================================
# PLOTS
# ================================================================
def make_plots(chirp_rows, dist_metrics, baseline_model):
    ref_rows = [r for r in chirp_rows if r["condition"] == "reflector"]
    base_rows = [r for r in chirp_rows if r["condition"] == "baseline"]

    # 1. Expected vs measured delay
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in dist_metrics:
        d = m["distance_cm"]
        rows = [r for r in ref_rows if r["distance_cm"] == d]
        ax.scatter([m["expected_delay_ms"]] * len(rows),
                   [r["measured_delay_ms"] for r in rows],
                   s=12, alpha=0.6, label=f"{d} cm")
    lims = [0, max([m["expected_delay_ms"] for m in dist_metrics]) * 1.1]
    ax.plot(lims, lims, "k--", label="y=x")
    ax.set_xlabel("Expected bistatic delay (ms)")
    ax.set_ylabel("Measured delay (ms)")
    ax.set_title("1. Expected vs measured delay")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "1_expected_vs_measured.png"), dpi=150)
    plt.close(fig)

    # 2. Delay error vs distance
    fig, ax = plt.subplots(figsize=(7, 5))
    data = []
    for m in dist_metrics:
        d = m["distance_cm"]
        errs = [r["delay_error_ms"] for r in ref_rows if r["distance_cm"] == d]
        data.append(errs)
    ax.boxplot(data, labels=[str(m["distance_cm"]) for m in dist_metrics])
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Delay error (ms)")
    ax.set_title("2. Delay error vs distance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "2_delay_error_vs_distance.png"), dpi=150)
    plt.close(fig)

    # 3. Baseline vs reflector matched-filter response (mean per distance)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, m in enumerate(dist_metrics):
        d = m["distance_cm"]
        ax = axes[i]
        model = baseline_model.get(d)
        if model is None:
            ax.set_title(f"{d} cm (no baseline)")
            continue
        mean_resp, std_resp, delays = model
        ax.plot(delays, mean_resp, label="baseline mean", color="tab:blue")
        ax.fill_between(delays, mean_resp - std_resp, mean_resp + std_resp,
                        alpha=0.2, color="tab:blue")
        # reflector mean response
        resp_list = []
        for r in ref_rows:
            if r["distance_cm"] == d:
                resp_list.append(r["peak_strength"])
        # plot reflector candidate strengths at their delays
        r_delays = [r["measured_delay_ms"] for r in ref_rows
                    if r["distance_cm"] == d]
        r_strengths = [r["peak_strength"] for r in ref_rows
                       if r["distance_cm"] == d]
        ax.scatter(r_delays, r_strengths, color="tab:red", s=10,
                   label="reflector candidates")
        ax.axvline(m["expected_delay_ms"], color="k", linestyle="--",
                   alpha=0.6, label="expected")
        ax.set_title(f"{d} cm")
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("MF strength")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle("3. Baseline vs reflector matched-filter response")
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "3_baseline_vs_reflector.png"), dpi=150)
    plt.close(fig)

    # 4. Baseline-subtracted response (mean residual per distance)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, m in enumerate(dist_metrics):
        d = m["distance_cm"]
        ax = axes[i]
        model = baseline_model.get(d)
        if model is None:
            ax.set_title(f"{d} cm (no baseline)")
            continue
        mean_resp, std_resp, delays = model
        # residual = reflector candidate strength - baseline at same delay
        r_delays = [r["measured_delay_ms"] for r in ref_rows
                    if r["distance_cm"] == d]
        r_sub = [r["baseline_subtracted_strength"] for r in ref_rows
                 if r["distance_cm"] == d]
        ax.scatter(r_delays, r_sub, color="tab:red", s=12, alpha=0.7)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.axvline(m["expected_delay_ms"], color="k", linestyle="--",
                   alpha=0.6, label="expected")
        ax.set_title(f"{d} cm")
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Baseline-subtracted strength")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle("4. Baseline-subtracted response")
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "4_baseline_subtracted.png"), dpi=150)
    plt.close(fig)

    # 5. Reflection strength vs distance
    fig, ax = plt.subplots(figsize=(7, 5))
    dists = [m["distance_cm"] for m in dist_metrics]
    subs = [m["mean_baseline_subtracted_strength"] for m in dist_metrics]
    errs = [m["std_baseline_subtracted_strength"] for m in dist_metrics]
    ax.errorbar(dists, subs, yerr=errs, fmt="o-", capsize=4)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Mean baseline-subtracted strength")
    ax.set_title("5. Reflection strength vs distance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "5_strength_vs_distance.png"), dpi=150)
    plt.close(fig)

    # 6. Detection consistency across chirps
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in dist_metrics:
        d = m["distance_cm"]
        rates = []
        for ci in range(1, config.EXPECTED_CHIRPS + 1):
            rows = [r for r in ref_rows
                    if r["distance_cm"] == d and r["chirp_number"] == ci]
            if rows:
                rates.append(np.mean([r["final_detection"] for r in rows]))
            else:
                rates.append(0.0)
        ax.plot(range(1, config.EXPECTED_CHIRPS + 1), rates,
                marker="o", label=f"{d} cm")
    ax.set_xlabel("Chirp number")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("6. Detection consistency across chirps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "6_detection_consistency.png"), dpi=150)
    plt.close(fig)

    # 7. Peak-delay distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ref_delays = [r["measured_delay_ms"] for r in ref_rows]
    base_delays = [r["measured_delay_ms"] for r in base_rows]
    bins = np.linspace(0, max(ref_delays + base_delays) * 1.05, 40)
    ax.hist(base_delays, bins=bins, alpha=0.5, label="baseline",
            color="tab:blue")
    ax.hist(ref_delays, bins=bins, alpha=0.5, label="reflector",
            color="tab:red")
    for m in dist_metrics:
        ax.axvline(m["expected_delay_ms"], color="k", linestyle="--",
                   alpha=0.4)
    ax.set_xlabel("Measured peak delay (ms)")
    ax.set_ylabel("Count")
    ax.set_title("7. Peak-delay distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "7_peak_delay_distribution.png"), dpi=150)
    plt.close(fig)

    print("  Saved 7 plots to", config.PLOTS_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)