"""
===============================================================================
SARV ACOUSTIC GESTURE ANALYZER V7
RAW CORRELATION + MULTI-PEAK REFLECTION TRACKING + CONFIDENCE DIAGNOSTICS
===============================================================================

Purpose
-------
V6 established that the microphone receives a measurable response correlated
with the transmitted chirp, but the selected "echo" peak jumps between
different candidates.

V7 does NOT attempt to classify gestures.

Instead it investigates:

    1. Raw microphone signal
    2. Matched-filter / correlation response
    3. Multiple candidate reflection peaks
    4. Direct-path / leakage peak
    5. Peak-to-background ratio
    6. Peak prominence
    7. Candidate delay stability
    8. Reflection tracking across chirps
    9. Frequency centroid
   10. Spectrogram
   11. A reflection confidence score

The goal is to determine whether one acoustic reflection can be tracked
consistently across chirps.

Dataset expected:

    acoustic_test/
        acoustic_gesture_analyzer_v7.py
        gesture_dataset_v5/
            audio/
                approach/
                away/
                idle/

or:

        gesture_dataset_v5/
            audio/
                approach_001_....wav
                away_001_....wav
                idle_001_....wav

The script automatically searches recursively for WAV files and identifies
the class from the filename/path.

Dependencies:
    numpy
    scipy
    pandas
    matplotlib
    soundfile

Install if necessary:
    pip install numpy scipy pandas matplotlib soundfile

===============================================================================
"""

from __future__ import annotations

import os
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_DIR = SCRIPT_DIR / "gesture_dataset_v5"
AUDIO_DIR = DATASET_DIR / "audio"
OUTPUT_DIR = DATASET_DIR / "analysis_v7"

RECORDING_DIAGNOSTICS_CSV = OUTPUT_DIR / "v7_recording_diagnostics.csv"
CHIRP_DIAGNOSTICS_CSV = OUTPUT_DIR / "v7_chirp_diagnostics.csv"
PEAK_DIAGNOSTICS_CSV = OUTPUT_DIR / "v7_peak_diagnostics.csv"
CLASS_SUMMARY_CSV = OUTPUT_DIR / "v7_class_summary.csv"

PLOT_DIR = OUTPUT_DIR / "plots"

# -------------------------------------------------------------------------
# Chirp configuration
# -------------------------------------------------------------------------
#
# These values match the acoustic setup used in the previous SARV tests.
#
# If your original recorder used different values, change these constants.
#

CHIRP_F0 = 7500.0
CHIRP_F1 = 8500.0

# Duration of one transmitted chirp.
CHIRP_DURATION = 0.040

# Approximate interval between chirp starts.
#
# V6 plots showed chirps around:
#   0.15, 0.40, 0.65, 0.90, 1.15, 1.40, 1.65 s
#
# Therefore ~0.25 s spacing is expected.
CHIRP_INTERVAL = 0.250

# Number of chirps we try to analyze.
EXPECTED_CHIRPS = 7

# -------------------------------------------------------------------------
# Correlation configuration
# -------------------------------------------------------------------------

# Maximum physical delay we consider for a useful reflection.
#
# 20 ms corresponds to roughly:
#   0.020 * 343 / 2 = 3.43 m
#
# This is intentionally generous.
MAX_REFLECTION_DELAY_MS = 20.0

# Ignore extremely close correlation peaks because these are usually
# direct speaker -> microphone leakage / electronic-acoustic ringing.
MIN_REFLECTION_DELAY_MS = 0.50

# Search for several candidate peaks.
TOP_PEAKS = 5

# Minimum peak separation in milliseconds.
MIN_PEAK_SEPARATION_MS = 0.25

# Background region used for noise/confidence estimation.
BACKGROUND_PERCENTILE = 50

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

MAKE_REPRESENTATIVE_PLOTS = True
MAKE_ALL_RECORDING_PLOTS = False

# Set True if you want every recording plotted.
# This can generate many files.
MAKE_CORRELATION_PLOTS = True
MAKE_SPECTROGRAM_PLOTS = True
MAKE_TRACKING_PLOTS = True

# Maximum number of representative recordings per class.
REPRESENTATIVE_PER_CLASS = 1

# -------------------------------------------------------------------------
# Audio preprocessing
# -------------------------------------------------------------------------

USE_BANDPASS = True

BANDPASS_LOW = 6500.0
BANDPASS_HIGH = 9500.0

FILTER_ORDER = 4

# -------------------------------------------------------------------------
# Physical constants
# -------------------------------------------------------------------------

SPEED_OF_SOUND = 343.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def db(value: float, floor: float = 1e-12) -> float:
    value = max(abs(float(value)), floor)
    return 20.0 * math.log10(value)


def delay_to_distance_cm(delay_ms: float) -> float:
    """
    Convert round-trip acoustic delay to approximate target distance.

        distance = delay * speed_of_sound / 2
    """
    if not np.isfinite(delay_ms):
        return np.nan

    return (delay_ms / 1000.0) * SPEED_OF_SOUND / 2.0 * 100.0


# =============================================================================
# AUDIO LOADING
# =============================================================================

def load_audio(path: Path):
    """
    Load WAV using scipy.

    Returns:
        sample_rate, mono_float_signal
    """

    sr, data = wavfile.read(str(path))

    if data.ndim == 2:
        data = data.mean(axis=1)

    data = np.asarray(data)

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), info.max)
        data = data.astype(np.float64) / scale

    else:
        data = data.astype(np.float64)

    data = np.nan_to_num(data)

    # Remove DC.
    data = data - np.mean(data)

    return int(sr), data


# =============================================================================
# CHIRP GENERATION
# =============================================================================

def generate_chirp(sr: int) -> np.ndarray:
    """
    Generate the reference chirp used by the matched filter.
    """

    n = int(round(CHIRP_DURATION * sr))

    t = np.arange(n) / sr

    reference = signal.chirp(
        t,
        f0=CHIRP_F0,
        f1=CHIRP_F1,
        t1=CHIRP_DURATION,
        method="linear",
        phi=0,
    )

    # Apply a Hann window to reduce correlation sidelobes.
    reference *= signal.windows.hann(len(reference))

    # Normalize.
    norm = np.linalg.norm(reference)

    if norm > 0:
        reference /= norm

    return reference


# =============================================================================
# FILTERING
# =============================================================================

def bandpass_filter(x: np.ndarray, sr: int) -> np.ndarray:
    """
    Bandpass the microphone signal around the chirp frequency.
    """

    if not USE_BANDPASS:
        return x.copy()

    low = BANDPASS_LOW / (sr / 2.0)
    high = BANDPASS_HIGH / (sr / 2.0)

    if low <= 0 or high >= 1 or low >= high:
        return x.copy()

    sos = signal.butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass",
        output="sos",
    )

    try:
        return signal.sosfiltfilt(sos, x)
    except Exception:
        return signal.sosfilt(sos, x)


# =============================================================================
# MATCHED FILTER
# =============================================================================

def matched_filter(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Calculate normalized-ish matched-filter response.

    scipy correlation:
        correlate(received, reference)

    The resulting index corresponding to a positive lag is the location
    where the reference best matches the received signal.
    """

    corr = signal.correlate(
        x,
        reference,
        mode="full",
        method="fft",
    )

    return corr


def correlation_lags_for_received(
    received_length: int,
    reference_length: int,
) -> np.ndarray:

    return signal.correlation_lags(
        received_length,
        reference_length,
        mode="full",
    )


# =============================================================================
# CHIRP LOCATION DETECTION
# =============================================================================

def find_chirp_centers(
    x: np.ndarray,
    sr: int,
    reference: np.ndarray,
) -> list[float]:
    """
    Find likely transmitted chirp locations.

    We first correlate the complete recording with the reference chirp.
    Strong direct-path peaks should occur around each transmitted chirp.

    Returns:
        list of chirp center times in seconds.
    """

    corr = matched_filter(x, reference)

    lags = correlation_lags_for_received(
        len(x),
        len(reference),
    )

    positive = lags >= 0

    corr_pos = np.abs(corr[positive])
    lag_pos = lags[positive]

    if len(corr_pos) == 0:
        return []

    # Robust threshold.
    median = np.median(corr_pos)
    mad = np.median(np.abs(corr_pos - median)) + 1e-12

    threshold = median + 8.0 * mad

    distance_samples = max(
        1,
        int(0.12 * CHIRP_INTERVAL * sr),
    )

    peaks, properties = signal.find_peaks(
        corr_pos,
        height=threshold,
        distance=distance_samples,
        prominence=max(mad * 3.0, 1e-9),
    )

    if len(peaks) == 0:
        # Fallback: use strongest peaks.
        peaks, _ = signal.find_peaks(
            corr_pos,
            distance=distance_samples,
        )

    if len(peaks) == 0:
        return []

    peak_heights = corr_pos[peaks]

    # Keep the strongest candidates.
    max_candidates = max(EXPECTED_CHIRPS * 3, 20)

    if len(peaks) > max_candidates:
        order = np.argsort(peak_heights)[::-1][:max_candidates]
        peaks = peaks[order]

    candidate_times = lag_pos[peaks] / sr

    # Sort by time.
    candidate_times = np.sort(candidate_times)

    # Cluster nearby detections.
    clusters = []

    for t in candidate_times:
        if not clusters:
            clusters.append([t])
            continue

        if abs(t - np.mean(clusters[-1])) < 0.08:
            clusters[-1].append(t)
        else:
            clusters.append([t])

    centers = [float(np.mean(c)) for c in clusters]

    # We expect roughly regular spacing.
    if len(centers) <= EXPECTED_CHIRPS:
        return centers

    # Select a regular subset using expected interval.
    best = None
    best_score = np.inf

    for start in range(len(centers)):
        candidate = [centers[start]]

        for _ in range(EXPECTED_CHIRPS - 1):
            last = candidate[-1]

            remaining = [
                t for t in centers
                if t > last + 0.08
            ]

            if not remaining:
                break

            target = last + CHIRP_INTERVAL

            nxt = min(
                remaining,
                key=lambda z: abs(z - target),
            )

            if abs(nxt - target) > 0.12:
                break

            candidate.append(nxt)

        if len(candidate) >= 3:
            diffs = np.diff(candidate)
            score = np.mean(
                np.abs(diffs - CHIRP_INTERVAL)
            )

            if score < best_score:
                best_score = score
                best = candidate

    if best is not None:
        return best[:EXPECTED_CHIRPS]

    return centers[:EXPECTED_CHIRPS]


# =============================================================================
# LOCAL CHIRP CORRELATION
# =============================================================================

def extract_chirp_window(
    x: np.ndarray,
    sr: int,
    center_time: float,
) -> tuple[np.ndarray, int]:
    """
    Extract enough signal around a chirp to inspect reflections.

    We include:
        chirp
        + up to MAX_REFLECTION_DELAY_MS

    Returns:
        window
        starting sample
    """

    center_sample = int(round(center_time * sr))

    chirp_samples = int(round(CHIRP_DURATION * sr))
    max_delay_samples = int(
        round(MAX_REFLECTION_DELAY_MS / 1000.0 * sr)
    )

    # Start slightly before expected chirp.
    pre_samples = int(round(0.003 * sr))

    start = max(
        0,
        center_sample - pre_samples,
    )

    end = min(
        len(x),
        center_sample
        + chirp_samples
        + max_delay_samples,
    )

    return x[start:end], start


def local_correlation(
    window: np.ndarray,
    reference: np.ndarray,
    sr: int,
):
    """
    Correlate a local chirp window with the reference.

    Returns:
        correlation,
        lag_seconds
    """

    corr = signal.correlate(
        window,
        reference,
        mode="full",
        method="fft",
    )

    lags = signal.correlation_lags(
        len(window),
        len(reference),
        mode="full",
    )

    lag_seconds = lags / sr

    return corr, lag_seconds


# =============================================================================
# CANDIDATE PEAK EXTRACTION
# =============================================================================

def find_candidate_peaks(
    corr: np.ndarray,
    lag_seconds: np.ndarray,
    top_n: int = TOP_PEAKS,
):
    """
    Find several candidate positive-lag reflection peaks.

    We intentionally DO NOT return only the strongest peak.

    This is the major change from the V6 diagnostic philosophy.
    """

    abs_corr = np.abs(corr)

    min_delay = MIN_REFLECTION_DELAY_MS / 1000.0
    max_delay = MAX_REFLECTION_DELAY_MS / 1000.0

    valid = (
        (lag_seconds >= min_delay)
        & (lag_seconds <= max_delay)
    )

    if not np.any(valid):
        return []

    indices = np.where(valid)[0]

    values = abs_corr[indices]

    if len(values) < 5:
        return []

    # Robust background.
    background = np.percentile(
        values,
        BACKGROUND_PERCENTILE,
    )

    mad = np.median(
        np.abs(values - np.median(values))
    ) + 1e-12

    prominence_threshold = max(
        mad * 2.0,
        background * 0.05,
        1e-10,
    )

    distance_samples = max(
        1,
        int(
            MIN_PEAK_SEPARATION_MS
            / 1000.0
            * (
                1.0
                / np.median(
                    np.diff(lag_seconds)
                )
            )
        ),
    )

    peaks, properties = signal.find_peaks(
        values,
        distance=distance_samples,
        prominence=prominence_threshold,
    )

    if len(peaks) == 0:
        return []

    candidate_indices = indices[peaks]

    candidate_values = abs_corr[candidate_indices]

    order = np.argsort(candidate_values)[::-1]

    selected = []

    for idx in order:
        absolute_idx = candidate_indices[idx]

        delay_ms = (
            lag_seconds[absolute_idx] * 1000.0
        )

        amplitude = abs_corr[absolute_idx]

        # Estimate local prominence.
        prominence = 0.0

        if "prominences" in properties:
            # Find corresponding peak.
            original_peak_position = np.where(
                peaks == candidate_indices[idx] - indices[0]
            )[0]

            if len(original_peak_position):
                prominence = float(
                    properties["prominences"][
                        original_peak_position[0]
                    ]
                )

        selected.append(
            {
                "sample_index": int(absolute_idx),
                "delay_ms": float(delay_ms),
                "amplitude": float(amplitude),
                "prominence": float(prominence),
            }
        )

        if len(selected) >= top_n:
            break

    # Sort by delay for easier tracking.
    selected.sort(
        key=lambda p: p["delay_ms"]
    )

    return selected


# =============================================================================
# PEAK QUALITY
# =============================================================================

def calculate_peak_quality(
    corr: np.ndarray,
    lag_seconds: np.ndarray,
    peak: dict,
) -> dict:
    """
    Calculate diagnostics for one candidate peak.
    """

    abs_corr = np.abs(corr)

    delay_ms = peak["delay_ms"]

    idx = peak["sample_index"]

    # Exclude a small region around the peak when calculating background.
    exclusion_ms = 0.20

    exclusion = (
        np.abs(
            lag_seconds * 1000.0 - delay_ms
        )
        < exclusion_ms
    )

    background_values = abs_corr[
        (lag_seconds >= MIN_REFLECTION_DELAY_MS / 1000.0)
        & (lag_seconds <= MAX_REFLECTION_DELAY_MS / 1000.0)
        & (~exclusion)
    ]

    if len(background_values) == 0:
        background = 1e-12
    else:
        background = np.median(
            background_values
        ) + 1e-12

    amplitude = peak["amplitude"]

    amplitude_ratio = amplitude / background

    snr_proxy_db = db(
        amplitude_ratio
    )

    # Local peak-to-neighborhood ratio.
    neighborhood_samples = max(
        3,
        int(
            0.15
            / 1000.0
            / np.median(np.diff(lag_seconds))
        )
    )

    lo = max(
        0,
        idx - neighborhood_samples,
    )

    hi = min(
        len(abs_corr),
        idx + neighborhood_samples + 1,
    )

    neighborhood = np.concatenate(
        [
            abs_corr[lo:idx],
            abs_corr[idx + 1:hi],
        ]
    )

    if len(neighborhood):
        local_background = np.median(
            neighborhood
        ) + 1e-12
    else:
        local_background = 1e-12

    local_ratio = amplitude / local_background

    return {
        "background": float(background),
        "amplitude_ratio": float(amplitude_ratio),
        "snr_proxy_db": float(snr_proxy_db),
        "local_ratio": float(local_ratio),
    }


# =============================================================================
# DIRECT PATH DIAGNOSTIC
# =============================================================================

def estimate_direct_peak(
    corr: np.ndarray,
    lag_seconds: np.ndarray,
):
    """
    Find the strongest correlation peak close to zero delay.

    This is not assumed to be a hand reflection.
    """

    valid = (
        lag_seconds >= 0
    ) & (
        lag_seconds <= 0.50 / 1000.0
    )

    if not np.any(valid):
        return None

    idxs = np.where(valid)[0]

    local = np.abs(corr[idxs])

    if len(local) == 0:
        return None

    best_local = np.argmax(local)

    idx = idxs[best_local]

    return {
        "delay_ms": float(
            lag_seconds[idx] * 1000.0
        ),
        "amplitude": float(
            abs(corr[idx])
        ),
    }


# =============================================================================
# FREQUENCY CENTROID
# =============================================================================

def calculate_frequency_centroid(
    x: np.ndarray,
    sr: int,
):
    """
    Calculate spectral centroid inside the chirp frequency band.
    """

    if len(x) < 16:
        return np.nan

    window = signal.windows.hann(len(x))

    spectrum = np.abs(
        np.fft.rfft(
            x * window
        )
    )

    freqs = np.fft.rfftfreq(
        len(x),
        1.0 / sr,
    )

    valid = (
        (freqs >= BANDPASS_LOW)
        & (freqs <= BANDPASS_HIGH)
    )

    if not np.any(valid):
        return np.nan

    s = spectrum[valid]
    f = freqs[valid]

    denominator = np.sum(s)

    if denominator <= 0:
        return np.nan

    return float(
        np.sum(f * s) / denominator
    )


# =============================================================================
# SPECTROGRAM
# =============================================================================

def calculate_spectrogram(
    x: np.ndarray,
    sr: int,
):
    """
    Generate spectrogram data.
    """

    nperseg = min(
        1024,
        len(x),
    )

    if nperseg < 32:
        return None

    noverlap = int(
        nperseg * 0.75
    )

    f, t, Sxx = signal.spectrogram(
        x,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )

    return f, t, Sxx


# =============================================================================
# REFLECTION TRACKING
# =============================================================================

def track_reflection(
    chirp_records: list[dict],
):
    """
    Track a reflection candidate across chirps.

    The tracker tries to maintain a physically plausible delay trajectory.

    It does NOT force a track to exist.

    Candidate costs favor:
        - strong peaks
        - high SNR
        - smooth delay movement

    Returns:
        list of selected candidate indices
    """

    if not chirp_records:
        return []

    # Dynamic programming.
    states = []

    for record in chirp_records:
        peaks = record["peaks"]

        if not peaks:
            states.append([])
            continue

        state = []

        for p_idx, peak in enumerate(peaks):
            amplitude = peak["amplitude"]

            quality = peak.get(
                "snr_proxy_db",
                0.0,
            )

            # Higher is better.
            observation_score = (
                math.log1p(
                    max(amplitude, 0.0)
                    * 1e6
                )
                + 0.05 * quality
            )

            state.append(
                {
                    "peak_index": p_idx,
                    "delay_ms": peak["delay_ms"],
                    "score": observation_score,
                }
            )

        states.append(state)

    if not any(states):
        return []

    # Find first available state.
    first = next(
        i for i, s in enumerate(states)
        if s
    )

    dp = []
    back = []

    # Before first valid chirp.
    for _ in range(first):
        dp.append(None)
        back.append(None)

    initial = np.array(
        [
            item["score"]
            for item in states[first]
        ],
        dtype=float,
    )

    dp.append(initial)
    back.append(
        np.full(
            len(initial),
            -1,
            dtype=int,
        )
    )

    previous_chirp_index = first

    for chirp_index in range(first + 1, len(states)):

        current_states = states[chirp_index]

        if not current_states:
            dp.append(None)
            back.append(None)
            continue

        # Find latest previous non-empty state.
        prev_index = None

        for j in range(chirp_index - 1, -1, -1):
            if states[j]:
                prev_index = j
                break

        if prev_index is None:
            dp.append(
                np.array(
                    [
                        item["score"]
                        for item in current_states
                    ]
                )
            )

            back.append(
                np.full(
                    len(current_states),
                    -1,
                    dtype=int,
                )
            )

            continue

        # Locate corresponding DP vector.
        prev_dp = dp[prev_index]

        current_dp = np.full(
            len(current_states),
            -np.inf,
        )

        current_back = np.full(
            len(current_states),
            -1,
            dtype=int,
        )

        time_gap = max(
            1,
            chirp_index - prev_index,
        )

        for ci, current in enumerate(
            current_states
        ):

            best_value = -np.inf
            best_prev = -1

            for pi, previous in enumerate(
                states[prev_index]
            ):

                delay_difference = abs(
                    current["delay_ms"]
                    - previous["delay_ms"]
                )

                # Allow movement, but strongly penalize
                # enormous jumps.
                smooth_penalty = (
                    0.10
                    * delay_difference
                    * delay_difference
                )

                gap_penalty = (
                    0.02
                    * max(0, time_gap - 1)
                    * delay_difference
                )

                candidate_value = (
                    prev_dp[pi]
                    + current["score"]
                    - smooth_penalty
                    - gap_penalty
                )

                if candidate_value > best_value:
                    best_value = candidate_value
                    best_prev = pi

            current_dp[ci] = best_value
            current_back[ci] = best_prev

        dp.append(current_dp)
        back.append(current_back)

    # Find last valid DP state.
    last_index = None

    for i in range(len(dp) - 1, -1, -1):
        if dp[i] is not None:
            last_index = i
            break

    if last_index is None:
        return []

    current_index = int(
        np.argmax(dp[last_index])
    )

    selected = {}

    i = last_index

    while i >= first and current_index >= 0:

        selected[i] = current_index

        if i == first:
            break

        b = back[i]

        if b is None:
            break

        previous_index = int(
            b[current_index]
        )

        if previous_index < 0:
            break

        current_index = previous_index

        # Find the actual previous non-empty chirp.
        j = i - 1

        while j >= first and not states[j]:
            j -= 1

        if j < first:
            break

        i = j

    result = []

    for chirp_index in range(
        len(chirp_records)
    ):
        result.append(
            selected.get(
                chirp_index,
                None,
            )
        )

    return result


# =============================================================================
# RECORDING ANALYSIS
# =============================================================================

def analyze_recording(
    path: Path,
):
    """
    Analyze one recording.
    """

    sr, raw = load_audio(path)

    if len(raw) == 0:
        raise ValueError("Empty recording")

    filtered = bandpass_filter(
        raw,
        sr,
    )

    reference = generate_chirp(sr)

    chirp_centers = find_chirp_centers(
        filtered,
        sr,
        reference,
    )

    # If detection fails, estimate expected chirp times.
    if len(chirp_centers) == 0:

        duration = len(raw) / sr

        first = 0.15

        chirp_centers = []

        for i in range(EXPECTED_CHIRPS):

            t = first + i * CHIRP_INTERVAL

            if t < duration:
                chirp_centers.append(t)

    chirp_records = []

    for chirp_number, center_time in enumerate(
        chirp_centers,
        start=1,
    ):

        window, start_sample = extract_chirp_window(
            filtered,
            sr,
            center_time,
        )

        corr, lag_seconds = local_correlation(
            window,
            reference,
            sr,
        )

        peaks = find_candidate_peaks(
            corr,
            lag_seconds,
            TOP_PEAKS,
        )

        direct = estimate_direct_peak(
            corr,
            lag_seconds,
        )

        enriched_peaks = []

        for rank, peak in enumerate(
            peaks,
            start=1,
        ):

            quality = calculate_peak_quality(
                corr,
                lag_seconds,
                peak,
            )

            peak_record = {
                **peak,
                **quality,
                "rank": rank,
            }

            enriched_peaks.append(
                peak_record
            )

        # Frequency centroid around the chirp.
        chirp_samples = int(
            round(
                CHIRP_DURATION * sr
            )
        )

        center_sample = int(
            round(
                center_time * sr
            )
        )

        f_start = max(
            0,
            center_sample
            - int(0.002 * sr),
        )

        f_end = min(
            len(filtered),
            center_sample
            + chirp_samples
            + int(0.002 * sr),
        )

        chirp_audio = filtered[
            f_start:f_end
        ]

        frequency_centroid = (
            calculate_frequency_centroid(
                chirp_audio,
                sr,
            )
        )

        chirp_records.append(
            {
                "chirp_number": chirp_number,
                "time_s": float(center_time),
                "start_sample": int(start_sample),
                "peaks": enriched_peaks,
                "direct": direct,
                "frequency_centroid_hz": frequency_centroid,
                "corr": corr,
                "lags": lag_seconds,
            }
        )

    # Track reflection.
    tracking = track_reflection(
        chirp_records
    )

    tracked_delays = []

    for i, selected_index in enumerate(
        tracking
    ):

        if (
            selected_index is not None
            and i < len(chirp_records)
        ):

            peaks = chirp_records[i]["peaks"]

            if selected_index < len(peaks):

                peak = peaks[selected_index]

                peak["tracked"] = True

                tracked_delays.append(
                    peak["delay_ms"]
                )

            else:
                tracked_delays.append(
                    np.nan
                )

        else:
            tracked_delays.append(
                np.nan
            )

    # Delay statistics.
    valid_delays = np.asarray(
        [
            d
            for d in tracked_delays
            if np.isfinite(d)
        ],
        dtype=float,
    )

    if len(valid_delays) >= 1:

        median_delay = float(
            np.median(valid_delays)
        )

        delay_std = float(
            np.std(valid_delays)
        )

        delay_change = float(
            valid_delays[-1]
            - valid_delays[0]
        )

    else:

        median_delay = np.nan
        delay_std = np.nan
        delay_change = np.nan

    # Track coverage.
    track_fraction = (
        len(valid_delays)
        / max(
            1,
            len(chirp_records),
        )
    )

    # Frequency statistics.
    frequencies = np.asarray(
        [
            r["frequency_centroid_hz"]
            for r in chirp_records
            if np.isfinite(
                r["frequency_centroid_hz"]
            )
        ],
        dtype=float,
    )

    if len(frequencies):

        frequency_mean = float(
            np.mean(frequencies)
        )

        frequency_std = float(
            np.std(frequencies)
        )

        frequency_change = float(
            frequencies[-1]
            - frequencies[0]
        )

    else:

        frequency_mean = np.nan
        frequency_std = np.nan
        frequency_change = np.nan

    # Direct-path statistics.
    direct_values = [
        r["direct"]["amplitude"]
        for r in chirp_records
        if r["direct"] is not None
    ]

    direct_strength = (
        float(np.median(direct_values))
        if direct_values
        else np.nan
    )

    # Tracked reflection strength.
    tracked_amplitudes = []

    tracked_snr = []

    for i, selected_index in enumerate(
        tracking
    ):

        if selected_index is None:
            continue

        if i >= len(chirp_records):
            continue

        peaks = chirp_records[i]["peaks"]

        if selected_index >= len(peaks):
            continue

        peak = peaks[selected_index]

        tracked_amplitudes.append(
            peak["amplitude"]
        )

        tracked_snr.append(
            peak["snr_proxy_db"]
        )

    if tracked_amplitudes:
        reflection_strength = float(
            np.median(
                tracked_amplitudes
            )
        )
    else:
        reflection_strength = np.nan

    if tracked_snr:
        reflection_snr = float(
            np.median(
                tracked_snr
            )
        )
    else:
        reflection_snr = np.nan

    if (
        np.isfinite(reflection_strength)
        and np.isfinite(direct_strength)
        and direct_strength > 0
    ):
        reflection_direct_ratio = (
            reflection_strength
            / direct_strength
        )
    else:
        reflection_direct_ratio = np.nan

    # ---------------------------------------------------------------------
    # Reflection confidence
    # ---------------------------------------------------------------------
    #
    # This is NOT a gesture confidence.
    #
    # It answers:
    # "How believable is the existence of a trackable reflection?"
    #

    confidence = 0.0

    # Track coverage: 40 points.
    confidence += (
        40.0 * track_fraction
    )

    # Delay stability: 30 points.
    if np.isfinite(delay_std):

        # <0.5 ms = excellent
        # >5 ms = poor
        stability = max(
            0.0,
            min(
                1.0,
                1.0
                - delay_std / 5.0,
            ),
        )

        confidence += (
            30.0 * stability
        )

    # SNR: 30 points.
    if np.isfinite(reflection_snr):

        snr_quality = max(
            0.0,
            min(
                1.0,
                reflection_snr / 15.0,
            ),
        )

        confidence += (
            30.0 * snr_quality
        )

    confidence = min(
        100.0,
        confidence,
    )

    # Class from filename/path.
    lower = str(path).lower()

    if "approach" in lower:
        gesture_class = "approach"
    elif "away" in lower:
        gesture_class = "away"
    elif "idle" in lower:
        gesture_class = "idle"
    else:
        gesture_class = "unknown"

    recording = {
        "file": path.name,
        "path": str(path),
        "class": gesture_class,
        "sample_rate": sr,
        "duration_s": len(raw) / sr,
        "chirps_detected": len(chirp_records),
        "track_fraction": track_fraction,
        "tracked_delay_ms": median_delay,
        "tracked_delay_std_ms": delay_std,
        "tracked_delay_change_ms": delay_change,
        "tracked_distance_cm": (
            delay_to_distance_cm(
                median_delay
            )
        ),
        "reflection_strength": reflection_strength,
        "direct_strength": direct_strength,
        "reflection_direct_ratio": reflection_direct_ratio,
        "reflection_snr_proxy_db": reflection_snr,
        "frequency_mean_hz": frequency_mean,
        "frequency_std_hz": frequency_std,
        "frequency_change_hz": frequency_change,
        "reflection_confidence": confidence,
        "chirp_centers": chirp_centers,
        "chirps": chirp_records,
        "tracked_indices": tracking,
        "raw": raw,
        "filtered": filtered,
        "sr": sr,
    }

    return recording


# =============================================================================
# CSV ROW GENERATION
# =============================================================================

def recording_to_row(recording: dict):

    return {
        "file": recording["file"],
        "class": recording["class"],
        "sample_rate": recording["sample_rate"],
        "duration_s": recording["duration_s"],
        "chirps_detected": recording["chirps_detected"],
        "track_fraction": recording["track_fraction"],
        "tracked_delay_ms": recording["tracked_delay_ms"],
        "tracked_delay_std_ms": recording[
            "tracked_delay_std_ms"
        ],
        "tracked_delay_change_ms": recording[
            "tracked_delay_change_ms"
        ],
        "tracked_distance_cm": recording[
            "tracked_distance_cm"
        ],
        "reflection_strength": recording[
            "reflection_strength"
        ],
        "direct_strength": recording[
            "direct_strength"
        ],
        "reflection_direct_ratio": recording[
            "reflection_direct_ratio"
        ],
        "reflection_snr_proxy_db": recording[
            "reflection_snr_proxy_db"
        ],
        "frequency_mean_hz": recording[
            "frequency_mean_hz"
        ],
        "frequency_std_hz": recording[
            "frequency_std_hz"
        ],
        "frequency_change_hz": recording[
            "frequency_change_hz"
        ],
        "reflection_confidence": recording[
            "reflection_confidence"
        ],
    }


def chirps_to_rows(recording: dict):

    rows = []

    for chirp in recording["chirps"]:

        tracked_index = None

        chirp_index = (
            chirp["chirp_number"] - 1
        )

        if chirp_index < len(
            recording["tracked_indices"]
        ):
            tracked_index = recording[
                "tracked_indices"
            ][chirp_index]

        tracked_delay = np.nan
        tracked_snr = np.nan
        tracked_amplitude = np.nan

        if tracked_index is not None:

            if tracked_index < len(
                chirp["peaks"]
            ):

                tracked_peak = chirp[
                    "peaks"
                ][tracked_index]

                tracked_delay = tracked_peak[
                    "delay_ms"
                ]

                tracked_snr = tracked_peak[
                    "snr_proxy_db"
                ]

                tracked_amplitude = (
                    tracked_peak[
                        "amplitude"
                    ]
                )

        rows.append(
            {
                "file": recording["file"],
                "class": recording["class"],
                "chirp_number": chirp[
                    "chirp_number"
                ],
                "chirp_time_s": chirp[
                    "time_s"
                ],
                "frequency_centroid_hz": chirp[
                    "frequency_centroid_hz"
                ],
                "candidate_count": len(
                    chirp["peaks"]
                ),
                "tracked_peak_index": (
                    tracked_index
                ),
                "tracked_delay_ms": (
                    tracked_delay
                ),
                "tracked_amplitude": (
                    tracked_amplitude
                ),
                "tracked_snr_proxy_db": (
                    tracked_snr
                ),
                "direct_delay_ms": (
                    chirp["direct"]["delay_ms"]
                    if chirp["direct"]
                    else np.nan
                ),
                "direct_amplitude": (
                    chirp["direct"]["amplitude"]
                    if chirp["direct"]
                    else np.nan
                ),
            }
        )

    return rows


def peaks_to_rows(recording: dict):

    rows = []

    for chirp in recording["chirps"]:

        for peak in chirp["peaks"]:

            rows.append(
                {
                    "file": recording["file"],
                    "class": recording["class"],
                    "chirp_number": chirp[
                        "chirp_number"
                    ],
                    "chirp_time_s": chirp[
                        "time_s"
                    ],
                    "peak_rank_by_delay": peak[
                        "rank"
                    ],
                    "delay_ms": peak[
                        "delay_ms"
                    ],
                    "distance_cm": (
                        delay_to_distance_cm(
                            peak[
                                "delay_ms"
                            ]
                        )
                    ),
                    "amplitude": peak[
                        "amplitude"
                    ],
                    "prominence": peak[
                        "prominence"
                    ],
                    "background": peak[
                        "background"
                    ],
                    "amplitude_ratio": peak[
                        "amplitude_ratio"
                    ],
                    "snr_proxy_db": peak[
                        "snr_proxy_db"
                    ],
                    "local_ratio": peak[
                        "local_ratio"
                    ],
                }
            )

    return rows


# =============================================================================
# PLOT: CORRELATION / MULTI-PEAK
# =============================================================================

def plot_correlation(recording: dict):

    if not MAKE_CORRELATION_PLOTS:
        return

    cls = recording["class"]

    chirps = recording["chirps"]

    if not chirps:
        return

    fig, axes = plt.subplots(
        len(chirps),
        1,
        figsize=(12, max(7, 2.2 * len(chirps))),
        squeeze=False,
    )

    axes = axes.flatten()

    for i, chirp in enumerate(chirps):

        ax = axes[i]

        delay_ms = (
            chirp["lags"] * 1000.0
        )

        corr_abs = np.abs(
            chirp["corr"]
        )

        valid = (
            (delay_ms >= 0)
            & (
                delay_ms
                <= MAX_REFLECTION_DELAY_MS
            )
        )

        ax.plot(
            delay_ms[valid],
            corr_abs[valid],
        )

        for rank, peak in enumerate(
            chirp["peaks"],
            start=1,
        ):

            ax.scatter(
                peak["delay_ms"],
                peak["amplitude"],
                s=45,
            )

            ax.annotate(
                f"P{rank}\n{peak['delay_ms']:.2f} ms",
                (
                    peak["delay_ms"],
                    peak["amplitude"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # Mark tracked peak.
        tracked_index = None

        if (
            i
            < len(
                recording[
                    "tracked_indices"
                ]
            )
        ):
            tracked_index = recording[
                "tracked_indices"
            ][i]

        if tracked_index is not None:

            if tracked_index < len(
                chirp["peaks"]
            ):

                tracked = chirp[
                    "peaks"
                ][tracked_index]

                ax.axvline(
                    tracked["delay_ms"],
                    linestyle="--",
                )

        ax.set_ylabel(
            f"C{i + 1}\n|corr|"
        )

        ax.grid(
            alpha=0.25
        )

    axes[-1].set_xlabel(
        "Candidate delay (ms)"
    )

    fig.suptitle(
        f"{cls.upper()} - V7 Multi-Peak Correlation\n"
        f"{recording['file']}",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    filename = (
        Path(recording["file"]).stem
        + "_v7_correlation.png"
    )

    fig.savefig(
        PLOT_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# PLOT: DELAY TRACKING
# =============================================================================

def plot_tracking(recording: dict):

    if not MAKE_TRACKING_PLOTS:
        return

    chirps = recording["chirps"]

    if not chirps:
        return

    times = []
    tracked_delays = []

    all_peak_times = []
    all_peak_delays = []

    for i, chirp in enumerate(chirps):

        t = chirp["time_s"]

        for peak in chirp["peaks"]:

            all_peak_times.append(t)
            all_peak_delays.append(
                peak["delay_ms"]
            )

        tracked_index = None

        if (
            i
            < len(
                recording[
                    "tracked_indices"
                ]
            )
        ):
            tracked_index = recording[
                "tracked_indices"
            ][i]

        if tracked_index is not None:

            if tracked_index < len(
                chirp["peaks"]
            ):

                times.append(t)

                tracked_delays.append(
                    chirp["peaks"][
                        tracked_index
                    ]["delay_ms"]
                )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    if all_peak_times:

        ax.scatter(
            all_peak_times,
            all_peak_delays,
            s=30,
            alpha=0.35,
            label="candidate peaks",
        )

    if times:

        ax.plot(
            times,
            tracked_delays,
            marker="o",
            linewidth=2,
            label="V7 tracked reflection",
        )

    ax.set_xlabel(
        "Chirp time (seconds)"
    )

    ax.set_ylabel(
        "Candidate echo delay (ms)"
    )

    ax.set_title(
        f"{recording['class'].upper()} - V7 Reflection Tracking\n"
        f"{recording['file']}"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    filename = (
        Path(recording["file"]).stem
        + "_v7_tracking.png"
    )

    fig.savefig(
        PLOT_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# PLOT: FREQUENCY
# =============================================================================

def plot_frequency(recording: dict):

    chirps = recording["chirps"]

    if not chirps:
        return

    t = [
        c["time_s"]
        for c in chirps
    ]

    f = [
        c["frequency_centroid_hz"]
        for c in chirps
    ]

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        t,
        f,
        marker="o",
    )

    ax.set_xlabel(
        "Chirp time (seconds)"
    )

    ax.set_ylabel(
        "Frequency centroid (Hz)"
    )

    ax.set_title(
        f"{recording['class'].upper()} - V7 Frequency Centroid\n"
        f"{recording['file']}"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    filename = (
        Path(recording["file"]).stem
        + "_v7_frequency.png"
    )

    fig.savefig(
        PLOT_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# PLOT: SPECTROGRAM
# =============================================================================

def plot_spectrogram(recording: dict):

    if not MAKE_SPECTROGRAM_PLOTS:
        return

    x = recording["filtered"]
    sr = recording["sr"]

    result = calculate_spectrogram(
        x,
        sr,
    )

    if result is None:
        return

    f, t, Sxx = result

    valid = (
        (f >= 5000)
        & (f <= 11000)
    )

    if not np.any(valid):
        return

    f_plot = f[valid]
    S_plot = Sxx[valid]

    S_db = 20 * np.log10(
        S_plot + 1e-12
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    mesh = ax.pcolormesh(
        t,
        f_plot,
        S_db,
        shading="auto",
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Frequency (Hz)"
    )

    ax.set_title(
        f"{recording['class'].upper()} - V7 Spectrogram\n"
        f"{recording['file']}"
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Magnitude (dB)",
    )

    fig.tight_layout()

    filename = (
        Path(recording["file"]).stem
        + "_v7_spectrogram.png"
    )

    fig.savefig(
        PLOT_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# PLOT: RAW SIGNAL
# =============================================================================

def plot_raw_signal(recording: dict):

    x = recording["raw"]
    sr = recording["sr"]

    t = np.arange(len(x)) / sr

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        t,
        x,
        linewidth=0.7,
    )

    for chirp_time in recording[
        "chirp_centers"
    ]:

        ax.axvline(
            chirp_time,
            linestyle="--",
            alpha=0.4,
        )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_title(
        f"{recording['class'].upper()} - V7 Raw Signal\n"
        f"{recording['file']}"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    filename = (
        Path(recording["file"]).stem
        + "_v7_raw.png"
    )

    fig.savefig(
        PLOT_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# CLASS SUMMARY
# =============================================================================

def build_class_summary(
    recordings: list[dict],
):

    rows = []

    classes = [
        "approach",
        "away",
        "idle",
    ]

    for cls in classes:

        subset = [
            r
            for r in recordings
            if r["class"] == cls
        ]

        if not subset:
            continue

        def median(key):
            values = [
                r[key]
                for r in subset
                if np.isfinite(
                    r[key]
                )
            ]

            return (
                float(np.median(values))
                if values
                else np.nan
            )

        def mean(key):
            values = [
                r[key]
                for r in subset
                if np.isfinite(
                    r[key]
                )
            ]

            return (
                float(np.mean(values))
                if values
                else np.nan
            )

        rows.append(
            {
                "class": cls,
                "recordings": len(subset),
                "median_track_fraction": median(
                    "track_fraction"
                ),
                "median_delay_ms": median(
                    "tracked_delay_ms"
                ),
                "median_delay_std_ms": median(
                    "tracked_delay_std_ms"
                ),
                "median_delay_change_ms": median(
                    "tracked_delay_change_ms"
                ),
                "median_distance_cm": median(
                    "tracked_distance_cm"
                ),
                "median_reflection_direct_ratio": median(
                    "reflection_direct_ratio"
                ),
                "median_reflection_snr_db": median(
                    "reflection_snr_proxy_db"
                ),
                "median_frequency_hz": median(
                    "frequency_mean_hz"
                ),
                "median_frequency_std_hz": median(
                    "frequency_std_hz"
                ),
                "median_frequency_change_hz": median(
                    "frequency_change_hz"
                ),
                "mean_reflection_confidence": mean(
                    "reflection_confidence"
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# DIAGNOSTIC PRINTING
# =============================================================================

def print_recording_diagnostic(
    recording: dict,
):

    print()
    print(
        f"  {recording['file']}"
    )

    print(
        f"    class                 : "
        f"{recording['class']}"
    )

    print(
        f"    chirps detected      : "
        f"{recording['chirps_detected']}"
    )

    print(
        f"    track fraction       : "
        f"{recording['track_fraction']:.3f}"
    )

    print(
        f"    tracked delay        : "
        f"{recording['tracked_delay_ms']:.3f} ms"
    )

    print(
        f"    delay std            : "
        f"{recording['tracked_delay_std_ms']:.3f} ms"
    )

    print(
        f"    delay change         : "
        f"{recording['tracked_delay_change_ms']:.3f} ms"
    )

    print(
        f"    approximate distance : "
        f"{recording['tracked_distance_cm']:.2f} cm"
    )

    print(
        f"    reflection/direct    : "
        f"{recording['reflection_direct_ratio']:.4f}"
    )

    print(
        f"    reflection SNR proxy : "
        f"{recording['reflection_snr_proxy_db']:.2f} dB"
    )

    print(
        f"    frequency mean       : "
        f"{recording['frequency_mean_hz']:.2f} Hz"
    )

    print(
        f"    frequency std        : "
        f"{recording['frequency_std_hz']:.2f} Hz"
    )

    print(
        f"    frequency change     : "
        f"{recording['frequency_change_hz']:.2f} Hz"
    )

    print(
        f"    reflection confidence: "
        f"{recording['reflection_confidence']:.1f}/100"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "SARV ACOUSTIC GESTURE ANALYZER V7"
    )
    print(
        "RAW CORRELATION + MULTI-PEAK REFLECTION TRACKING"
    )
    print("=" * 78)

    print()
    print("Dataset:")
    print(DATASET_DIR)

    print()
    print("Audio:")
    print(AUDIO_DIR)

    print()
    print("V7 output:")
    print(OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # Validate paths.
    # -------------------------------------------------------------------------

    if not AUDIO_DIR.exists():

        print()
        print(
            "ERROR: Audio directory does not exist:"
        )
        print(AUDIO_DIR)

        print()
        print(
            "Expected structure:"
        )

        print(
            "gesture_dataset_v5/"
        )
        print(
            "    audio/"
        )
        print(
            "        approach_*.wav"
        )
        print(
            "        away_*.wav"
        )
        print(
            "        idle_*.wav"
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Find WAV files.
    # -------------------------------------------------------------------------

    wav_files = sorted(
        AUDIO_DIR.rglob("*.wav")
    )

    print()
    print(
        f"Audio recordings found: "
        f"{len(wav_files)}"
    )

    if not wav_files:

        print()
        print(
            "ERROR: No WAV recordings found."
        )

        return

    # -------------------------------------------------------------------------
    # Dataset distribution.
    # -------------------------------------------------------------------------

    counts = {
        "approach": 0,
        "away": 0,
        "idle": 0,
        "unknown": 0,
    }

    for path in wav_files:

        lower = str(path).lower()

        if "approach" in lower:
            counts["approach"] += 1

        elif "away" in lower:
            counts["away"] += 1

        elif "idle" in lower:
            counts["idle"] += 1

        else:
            counts["unknown"] += 1

    print()
    print("=" * 78)
    print("DATASET DISTRIBUTION")
    print("=" * 78)

    for cls, count in counts.items():

        print(
            f"{cls:<12}: {count}"
        )

    # -------------------------------------------------------------------------
    # Analyze.
    # -------------------------------------------------------------------------

    recordings = []

    failed = []

    print()
    print("=" * 78)
    print("V7 RECORDING ANALYSIS")
    print("=" * 78)

    for index, path in enumerate(
        wav_files,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(wav_files)}] "
            f"{path.name}"
        )

        try:

            recording = analyze_recording(
                path
            )

            recordings.append(
                recording
            )

            print(
                f"    chirps={recording['chirps_detected']} "
                f"track={recording['track_fraction']:.2f} "
                f"delay={recording['tracked_delay_ms']:.2f}ms "
                f"confidence={recording['reflection_confidence']:.1f}"
            )

        except Exception as exc:

            print(
                f"    FAILED: {exc}"
            )

            failed.append(
                (
                    path,
                    str(exc),
                )
            )

    # -------------------------------------------------------------------------
    # Recording CSV.
    # -------------------------------------------------------------------------

    recording_rows = [
        recording_to_row(r)
        for r in recordings
    ]

    recording_df = pd.DataFrame(
        recording_rows
    )

    recording_df.to_csv(
        RECORDING_DIAGNOSTICS_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Chirp CSV.
    # -------------------------------------------------------------------------

    chirp_rows = []

    for recording in recordings:

        chirp_rows.extend(
            chirps_to_rows(
                recording
            )
        )

    chirp_df = pd.DataFrame(
        chirp_rows
    )

    chirp_df.to_csv(
        CHIRP_DIAGNOSTICS_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Peak CSV.
    # -------------------------------------------------------------------------

    peak_rows = []

    for recording in recordings:

        peak_rows.extend(
            peaks_to_rows(
                recording
            )
        )

    peak_df = pd.DataFrame(
        peak_rows
    )

    peak_df.to_csv(
        PEAK_DIAGNOSTICS_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Class summary.
    # -------------------------------------------------------------------------

    summary_df = build_class_summary(
        recordings
    )

    summary_df.to_csv(
        CLASS_SUMMARY_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Representative plots.
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("GENERATING REPRESENTATIVE PLOTS")
    print("=" * 78)

    representatives = {}

    for recording in recordings:

        cls = recording["class"]

        if cls not in representatives:

            representatives[cls] = []

        if (
            len(
                representatives[cls]
            )
            < REPRESENTATIVE_PER_CLASS
        ):

            representatives[
                cls
            ].append(recording)

    for cls in [
        "idle",
        "approach",
        "away",
    ]:

        for recording in representatives.get(
            cls,
            [],
        ):

            print(
                f"Plotting {cls}: "
                f"{recording['file']}"
            )

            plot_raw_signal(
                recording
            )

            plot_correlation(
                recording
            )

            plot_tracking(
                recording
            )

            plot_frequency(
                recording
            )

            plot_spectrogram(
                recording
            )

    # -------------------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("V7 CLASS SUMMARY")
    print("=" * 78)

    if len(summary_df):

        print(
            summary_df.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.3f}",
            )
        )

    # -------------------------------------------------------------------------
    # Main diagnostic conclusion.
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("SARV V7 REFLECTION DIAGNOSIS")
    print("=" * 78)

    if not recordings:

        print()
        print(
            "No recordings were successfully analyzed."
        )

        return

    confidence_values = [
        r["reflection_confidence"]
        for r in recordings
        if np.isfinite(
            r["reflection_confidence"]
        )
    ]

    track_values = [
        r["track_fraction"]
        for r in recordings
        if np.isfinite(
            r["track_fraction"]
        )
    ]

    delay_std_values = [
        r["tracked_delay_std_ms"]
        for r in recordings
        if np.isfinite(
            r["tracked_delay_std_ms"]
        )
    ]

    if confidence_values:

        mean_confidence = float(
            np.mean(
                confidence_values
            )
        )

    else:

        mean_confidence = np.nan

    if track_values:

        mean_track = float(
            np.mean(
                track_values
            )
        )

    else:

        mean_track = np.nan

    if delay_std_values:

        median_delay_std = float(
            np.median(
                delay_std_values
            )
        )

    else:

        median_delay_std = np.nan

    print()
    print(
        f"Mean reflection confidence : "
        f"{mean_confidence:.1f}/100"
    )

    print(
        f"Mean track fraction         : "
        f"{mean_track:.3f}"
    )

    print(
        f"Median tracked delay std    : "
        f"{median_delay_std:.3f} ms"
    )

    print()

    # -------------------------------------------------------------------------
    # Interpretation.
    # -------------------------------------------------------------------------

    if (
        np.isfinite(mean_track)
        and np.isfinite(median_delay_std)
        and mean_track >= 0.70
        and median_delay_std <= 1.0
    ):

        print(
            "✓ A RELATIVELY STABLE REFLECTION TRACK "
            "IS PRESENT."
        )

        print()
        print(
            "This is promising."
        )

        print(
            "The next stage should investigate whether "
            "the tracked reflection moves systematically "
            "between approach, away and idle."
        )

    elif (
        np.isfinite(mean_track)
        and mean_track >= 0.50
    ):

        print(
            "⚠ PARTIAL REFLECTION TRACKING."
        )

        print()
        print(
            "Some candidate reflection structure exists, "
            "but the tracker is still losing or switching "
            "between peaks."
        )

        print(
            "Do NOT train a gesture classifier yet."
        )

    else:

        print(
            "✗ STABLE REFLECTION TRACK NOT VALIDATED."
        )

        print()
        print(
            "The microphone detects chirp-correlated "
            "energy, but V7 cannot consistently follow "
            "one candidate reflection."
        )

        print(
            "The next step should be improving the "
            "acoustic geometry / signal processing."
        )

    # -------------------------------------------------------------------------
    # Important warning.
    # -------------------------------------------------------------------------

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "V7 reflection confidence is NOT gesture confidence."
    )

    print(
        "V7 does NOT claim approach/away recognition."
    )

    print(
        "V7 is testing whether a physical reflection "
        "can be tracked reliably."
    )

    # -------------------------------------------------------------------------
    # Failed recordings.
    # -------------------------------------------------------------------------

    if failed:

        print()
        print("=" * 78)
        print(
            f"FAILED RECORDINGS: {len(failed)}"
        )
        print("=" * 78)

        for path, error in failed:

            print(
                f"{path.name}: {error}"
            )

    # -------------------------------------------------------------------------
    # Output files.
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("V7 OUTPUT FILES")
    print("=" * 78)

    print()
    print(
        "Recording diagnostics:"
    )
    print(
        RECORDING_DIAGNOSTICS_CSV
    )

    print()
    print(
        "Per-chirp diagnostics:"
    )
    print(
        CHIRP_DIAGNOSTICS_CSV
    )

    print()
    print(
        "All candidate peaks:"
    )
    print(
        PEAK_DIAGNOSTICS_CSV
    )

    print()
    print(
        "Class summary:"
    )
    print(
        CLASS_SUMMARY_CSV
    )

    print()
    print(
        "Plots:"
    )
    print(
        PLOT_DIR
    )

    print()
    print("=" * 78)
    print(
        "SARV V7 ANALYSIS COMPLETE"
    )
    print("=" * 78)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()