import os
import csv
import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import (
    butter,
    sosfiltfilt,
    correlate,
    spectrogram,
    find_peaks
)

warnings.filterwarnings("ignore")


# ======================================================================
# SARV ACOUSTIC GESTURE ANALYZER V6
#
# DIAGNOSTIC SIGNAL VALIDATION
#
# V6 DOES NOT COLLECT NEW DATA.
#
# It uses the existing V5 dataset and investigates:
#
#   1. Actual chirp timing
#   2. Raw waveform
#   3. Band-limited waveform
#   4. Matched-filter response
#   5. Direct-path response
#   6. Candidate reflection peaks
#   7. Echo-delay stability
#   8. Echo-strength stability
#   9. Frequency structure
#  10. Temporal motion signatures
#
# IMPORTANT:
#
# V6 does NOT assume that the strongest peak in the echo window
# is automatically the hand reflection.
#
# The purpose is to determine whether the current acoustic setup
# contains a measurable and repeatable reflection signal.
# ======================================================================


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "gesture_dataset_v5"
)

AUDIO_DIR = os.path.join(
    DATASET_DIR,
    "audio"
)

METADATA_FILE = os.path.join(
    DATASET_DIR,
    "metadata.csv"
)

OUTPUT_DIR = os.path.join(
    DATASET_DIR,
    "analysis_v6"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ======================================================================
# AUDIO CONFIGURATION
# ======================================================================

EXPECTED_SAMPLE_RATE = 44100

CHIRP_LOW = 7500
CHIRP_HIGH = 8500

CHIRP_DURATION = 0.100

EXPECTED_FIRST_CHIRP = 0.15
EXPECTED_CHIRP_INTERVAL = 0.250

EXPECTED_CHIRPS = 7

FILTER_LOW = 6000
FILTER_HIGH = 11000

SPEED_OF_SOUND = 343.0


# ======================================================================
# SEARCH WINDOWS
# ======================================================================

# Direct chirp localization search.
DIRECT_SEARCH_RADIUS_MS = 30.0

# Reflection search begins after direct response.
ECHO_MIN_DELAY_MS = 0.60
ECHO_MAX_DELAY_MS = 12.0

# Minimum separation between candidate peaks.
MIN_PEAK_DISTANCE_MS = 0.20

# Relative peak threshold.
#
# A candidate reflection must be at least this fraction
# of the strongest candidate in the search region.
MIN_RELATIVE_PEAK = 0.15


# ======================================================================
# OUTPUT
# ======================================================================

RECORDING_CSV = os.path.join(
    OUTPUT_DIR,
    "v6_recording_diagnostics.csv"
)

CHIRP_CSV = os.path.join(
    OUTPUT_DIR,
    "v6_chirp_diagnostics.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "v6_class_summary.csv"
)

GLOBAL_PLOT = os.path.join(
    OUTPUT_DIR,
    "v6_class_comparison.png"
)


# ======================================================================
# HEADER
# ======================================================================

print("=" * 78)
print("SARV ACOUSTIC GESTURE ANALYZER V6")
print("RAW SIGNAL + CHIRP VALIDATION + REFLECTION DIAGNOSTICS")
print("=" * 78)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Audio:")
print(AUDIO_DIR)

print()
print("V6 output:")
print(OUTPUT_DIR)


# ======================================================================
# CHECK DATASET
# ======================================================================

if not os.path.exists(
    METADATA_FILE
):
    raise FileNotFoundError(
        f"\nMetadata file not found:\n{METADATA_FILE}"
    )

if not os.path.exists(
    AUDIO_DIR
):
    raise FileNotFoundError(
        f"\nAudio directory not found:\n{AUDIO_DIR}"
    )


# ======================================================================
# LOAD METADATA
# ======================================================================

records = []

with open(
    METADATA_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        records.append(row)


print()
print(
    f"Metadata records: {len(records)}"
)


# ======================================================================
# FIND AUDIO
# ======================================================================

def find_audio(row):

    filename = row.get(
        "wav_file",
        ""
    ).strip()

    if not filename:
        return None

    path = os.path.join(
        AUDIO_DIR,
        filename
    )

    if os.path.exists(path):
        return path

    return None


# ======================================================================
# GENERATE REFERENCE CHIRP
# ======================================================================

def generate_reference_chirp():

    n = int(
        CHIRP_DURATION *
        EXPECTED_SAMPLE_RATE
    )

    t = (
        np.arange(n)
        /
        EXPECTED_SAMPLE_RATE
    )

    k = (
        CHIRP_HIGH -
        CHIRP_LOW
    ) / CHIRP_DURATION

    phase = (
        2 *
        np.pi *
        (
            CHIRP_LOW * t
            +
            0.5 * k * t * t
        )
    )

    chirp = np.sin(
        phase
    )

    chirp *= np.hanning(
        len(chirp)
    )

    chirp = chirp.astype(
        np.float64
    )

    chirp -= np.mean(
        chirp
    )

    norm = np.linalg.norm(
        chirp
    )

    if norm > 0:
        chirp /= norm

    return chirp


REFERENCE_CHIRP = (
    generate_reference_chirp()
)


# ======================================================================
# BANDPASS
# ======================================================================

def bandpass(
    signal,
    low=FILTER_LOW,
    high=FILTER_HIGH
):

    nyquist = EXPECTED_SAMPLE_RATE / 2.0

    sos = butter(
        6,
        [
            low / nyquist,
            high / nyquist
        ],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )


# ======================================================================
# LOAD AUDIO
# ======================================================================

def load_audio(path):

    sr, data = wavfile.read(
        path
    )

    original_dtype = data.dtype

    data = data.astype(
        np.float64
    )

    if data.ndim > 1:

        data = np.mean(
            data,
            axis=1
        )

    if np.issubdtype(
        original_dtype,
        np.integer
    ):

        info = np.iinfo(
            original_dtype
        )

        scale = max(
            abs(info.min),
            info.max
        )

        if scale > 0:
            data /= scale

    else:

        max_abs = np.max(
            np.abs(data)
        )

        if max_abs > 1.5:

            data /= max_abs

    return data, sr


# ======================================================================
# NORMALIZED MATCHED FILTER
# ======================================================================

def matched_filter(
    signal,
    reference
):

    signal = (
        signal -
        np.mean(signal)
    )

    reference = (
        reference -
        np.mean(reference)
    )

    correlation = correlate(
        signal,
        reference,
        mode="valid",
        method="fft"
    )

    ref_norm = np.linalg.norm(
        reference
    )

    ref_len = len(
        reference
    )

    energy = (
        signal ** 2
    )

    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(energy)
        ]
    )

    window_energy = (
        cumulative[
            ref_len:
        ]
        -
        cumulative[
            :-ref_len
        ]
    )

    denominator = (
        np.sqrt(
            np.maximum(
                window_energy,
                1e-20
            )
        )
        *
        ref_norm
        +
        1e-12
    )

    normalized = (
        correlation /
        denominator
    )

    return normalized


# ======================================================================
# EXPECTED CHIRP TIMES
# ======================================================================

def expected_chirp_times():

    return np.array(
        [
            EXPECTED_FIRST_CHIRP
            +
            i *
            EXPECTED_CHIRP_INTERVAL
            for i in range(
                EXPECTED_CHIRPS
            )
        ]
    )


# ======================================================================
# FIND ACTUAL DIRECT CHIRP
# ======================================================================

def find_direct_peak(
    mf,
    expected_time
):

    expected_index = int(
        expected_time *
        EXPECTED_SAMPLE_RATE
    )

    radius = int(
        DIRECT_SEARCH_RADIUS_MS
        /
        1000.0
        *
        EXPECTED_SAMPLE_RATE
    )

    start = max(
        0,
        expected_index - radius
    )

    end = min(
        len(mf),
        expected_index + radius
    )

    if end <= start:

        return {
            "index": -1,
            "time": np.nan,
            "strength": 0.0,
            "offset_ms": np.nan
        }

    region = np.abs(
        mf[start:end]
    )

    if len(region) == 0:

        return {
            "index": -1,
            "time": np.nan,
            "strength": 0.0,
            "offset_ms": np.nan
        }

    local = int(
        np.argmax(region)
    )

    index = (
        start +
        local
    )

    strength = float(
        region[local]
    )

    time = (
        index /
        EXPECTED_SAMPLE_RATE
    )

    offset_ms = (
        time -
        expected_time
    ) * 1000.0

    return {
        "index": index,
        "time": time,
        "strength": strength,
        "offset_ms": offset_ms
    }


# ======================================================================
# FIND CANDIDATE REFLECTIONS
# ======================================================================

def find_reflection_peaks(
    mf,
    direct_index
):

    if direct_index < 0:

        return []

    min_offset = int(
        ECHO_MIN_DELAY_MS
        /
        1000.0
        *
        EXPECTED_SAMPLE_RATE
    )

    max_offset = int(
        ECHO_MAX_DELAY_MS
        /
        1000.0
        *
        EXPECTED_SAMPLE_RATE
    )

    start = (
        direct_index +
        min_offset
    )

    end = (
        direct_index +
        max_offset
    )

    start = max(
        0,
        start
    )

    end = min(
        len(mf),
        end
    )

    if end <= start:
        return []

    region = np.abs(
        mf[start:end]
    )

    if len(region) < 10:
        return []

    max_value = np.max(
        region
    )

    if max_value <= 0:
        return []

    threshold = (
        max_value *
        MIN_RELATIVE_PEAK
    )

    distance = int(
        MIN_PEAK_DISTANCE_MS
        /
        1000.0
        *
        EXPECTED_SAMPLE_RATE
    )

    distance = max(
        distance,
        1
    )

    peaks, properties = find_peaks(
        region,
        height=threshold,
        distance=distance
    )

    candidates = []

    for peak_number, local_index in enumerate(
        peaks
    ):

        absolute_index = (
            start +
            int(local_index)
        )

        strength = float(
            region[
                local_index
            ]
        )

        delay_ms = (
            (
                absolute_index -
                direct_index
            )
            /
            EXPECTED_SAMPLE_RATE
            *
            1000.0
        )

        prominence = 0.0

        if (
            "prominences"
            in properties
        ):

            prominence = float(
                properties[
                    "prominences"
                ][
                    peak_number
                ]
            )

        candidates.append(
            {
                "index":
                    absolute_index,

                "delay_ms":
                    delay_ms,

                "strength":
                    strength,

                "relative_strength":
                    (
                        strength /
                        (
                            max_value +
                            1e-12
                        )
                    ),

                "prominence":
                    prominence
            }
        )

    candidates.sort(
        key=lambda x:
        x["strength"],
        reverse=True
    )

    return candidates


# ======================================================================
# ESTIMATE FREQUENCY CONTENT
# ======================================================================

def frequency_analysis(
    signal,
    center_time
):

    center = int(
        center_time *
        EXPECTED_SAMPLE_RATE
    )

    half = int(
        0.070 *
        EXPECTED_SAMPLE_RATE
    )

    start = max(
        0,
        center - half
    )

    end = min(
        len(signal),
        center + half
    )

    segment = signal[
        start:end
    ]

    if len(segment) < 1024:

        return {
            "frequency_centroid": np.nan,
            "frequency_peak": np.nan,
            "frequency_spread": np.nan,
            "band_energy": 0.0
        }

    frequencies, times, Sxx = spectrogram(
        segment,
        fs=EXPECTED_SAMPLE_RATE,
        nperseg=1024,
        noverlap=768,
        scaling="spectrum",
        mode="magnitude"
    )

    mask = (
        (frequencies >= 6500)
        &
        (frequencies <= 10000)
    )

    if not np.any(mask):

        return {
            "frequency_centroid": np.nan,
            "frequency_peak": np.nan,
            "frequency_spread": np.nan,
            "band_energy": 0.0
        }

    f = frequencies[
        mask
    ]

    spectrum = Sxx[
        mask
    ]

    mean_spectrum = np.mean(
        spectrum,
        axis=1
    )

    total = np.sum(
        mean_spectrum
    )

    if total <= 1e-12:

        return {
            "frequency_centroid": np.nan,
            "frequency_peak": np.nan,
            "frequency_spread": np.nan,
            "band_energy": 0.0
        }

    centroid = (
        np.sum(
            f *
            mean_spectrum
        )
        /
        total
    )

    peak_frequency = float(
        f[
            np.argmax(
                mean_spectrum
            )
        ]
    )

    variance = (
        np.sum(
            (
                f -
                centroid
            ) ** 2
            *
            mean_spectrum
        )
        /
        total
    )

    spread = np.sqrt(
        max(
            variance,
            0.0
        )
    )

    band_energy = float(
        np.sum(
            mean_spectrum ** 2
        )
    )

    return {
        "frequency_centroid":
            float(centroid),

        "frequency_peak":
            peak_frequency,

        "frequency_spread":
            float(spread),

        "band_energy":
            band_energy
    }


# ======================================================================
# ANALYZE ONE CHIRP
# ======================================================================

def analyze_chirp(
    filtered,
    expected_time,
    mf
):

    direct = find_direct_peak(
        mf,
        expected_time
    )

    if direct["index"] < 0:

        return {
            "expected_time":
                expected_time,

            "direct_time":
                np.nan,

            "direct_offset_ms":
                np.nan,

            "direct_strength":
                0.0,

            "candidate_count":
                0,

            "best_echo_delay_ms":
                np.nan,

            "best_echo_strength":
                0.0,

            "best_echo_relative":
                0.0,

            "echo_to_direct_ratio":
                0.0,

            "noise_median":
                0.0,

            "echo_snr_proxy":
                0.0,

            "frequency_centroid":
                np.nan,

            "frequency_peak":
                np.nan,

            "frequency_spread":
                np.nan,

            "band_energy":
                0.0,

            "candidates":
                []
        }

    candidates = find_reflection_peaks(
        mf,
        direct["index"]
    )

    if candidates:

        best = candidates[0]

        echo_delay = (
            best["delay_ms"]
        )

        echo_strength = (
            best["strength"]
        )

        relative = (
            best["relative_strength"]
        )

    else:

        echo_delay = np.nan
        echo_strength = 0.0
        relative = 0.0

    # --------------------------------------------------------------
    # Noise region BEFORE direct chirp
    # --------------------------------------------------------------

    noise_radius = int(
        25.0 /
        1000.0 *
        EXPECTED_SAMPLE_RATE
    )

    noise_gap = int(
        8.0 /
        1000.0 *
        EXPECTED_SAMPLE_RATE
    )

    noise_end = max(
        0,
        direct["index"] -
        noise_gap
    )

    noise_start = max(
        0,
        noise_end -
        noise_radius
    )

    if noise_end > noise_start:

        noise_values = np.abs(
            mf[
                noise_start:
                noise_end
            ]
        )

        noise_median = float(
            np.median(
                noise_values
            )
        )

        noise_std = float(
            np.std(
                noise_values
            )
        )

    else:

        noise_median = 0.0
        noise_std = 0.0

    echo_snr = (
        echo_strength /
        (
            noise_median +
            1e-12
        )
    )

    if direct["strength"] > 0:

        echo_direct_ratio = (
            echo_strength /
            direct["strength"]
        )

    else:

        echo_direct_ratio = 0.0

    frequency = frequency_analysis(
        filtered,
        direct["time"]
    )

    return {
        "expected_time":
            expected_time,

        "direct_time":
            direct["time"],

        "direct_offset_ms":
            direct["offset_ms"],

        "direct_strength":
            direct["strength"],

        "candidate_count":
            len(candidates),

        "best_echo_delay_ms":
            echo_delay,

        "best_echo_strength":
            echo_strength,

        "best_echo_relative":
            relative,

        "echo_to_direct_ratio":
            echo_direct_ratio,

        "noise_median":
            noise_median,

        "echo_snr_proxy":
            echo_snr,

        "frequency_centroid":
            frequency[
                "frequency_centroid"
            ],

        "frequency_peak":
            frequency[
                "frequency_peak"
            ],

        "frequency_spread":
            frequency[
                "frequency_spread"
            ],

        "band_energy":
            frequency[
                "band_energy"
            ],

        "candidates":
            candidates
    }


# ======================================================================
# ANALYZE RECORDING
# ======================================================================

def analyze_recording(
    audio_path
):

    audio, sr = load_audio(
        audio_path
    )

    if sr != EXPECTED_SAMPLE_RATE:

        raise ValueError(
            f"Expected "
            f"{EXPECTED_SAMPLE_RATE} Hz "
            f"but audio is "
            f"{sr} Hz"
        )

    if len(audio) < 0.5 * sr:

        raise ValueError(
            "Recording is too short."
        )

    filtered = bandpass(
        audio
    )

    mf = matched_filter(
        filtered,
        REFERENCE_CHIRP
    )

    times = (
        expected_chirp_times()
    )

    responses = []

    for t in times:

        response = analyze_chirp(
            filtered,
            t,
            mf
        )

        responses.append(
            response
        )

    return (
        audio,
        filtered,
        mf,
        responses
    )


# ======================================================================
# SAFE STATISTICS
# ======================================================================

def finite_values(values):

    values = np.asarray(
        values,
        dtype=float
    )

    return values[
        np.isfinite(values)
    ]


def safe_mean(values):

    values = finite_values(
        values
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def safe_std(values):

    values = finite_values(
        values
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.std(values)
    )


def safe_median(values):

    values = finite_values(
        values
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


def safe_slope(
    x,
    y
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if np.sum(mask) < 2:
        return np.nan

    try:

        return float(
            np.polyfit(
                x[mask],
                y[mask],
                1
            )[0]
        )

    except Exception:

        return np.nan


# ======================================================================
# RECORDING-LEVEL FEATURES
# ======================================================================

def recording_features(
    responses
):

    times = np.array(
        [
            r["expected_time"]
            for r in responses
        ]
    )

    delays = np.array(
        [
            r["best_echo_delay_ms"]
            for r in responses
        ]
    )

    strengths = np.array(
        [
            r["best_echo_strength"]
            for r in responses
        ]
    )

    ratios = np.array(
        [
            r["echo_to_direct_ratio"]
            for r in responses
        ]
    )

    snr = np.array(
        [
            r["echo_snr_proxy"]
            for r in responses
        ]
    )

    direct = np.array(
        [
            r["direct_strength"]
            for r in responses
        ]
    )

    direct_offsets = np.array(
        [
            r["direct_offset_ms"]
            for r in responses
        ]
    )

    frequencies = np.array(
        [
            r["frequency_centroid"]
            for r in responses
        ]
    )

    candidates = np.array(
        [
            r["candidate_count"]
            for r in responses
        ]
    )

    valid_delay_count = np.sum(
        np.isfinite(delays)
    )

    valid_delay_fraction = (
        valid_delay_count /
        len(delays)
    )

    delay_slope = safe_slope(
        times,
        delays
    )

    strength_slope = safe_slope(
        times,
        strengths
    )

    ratio_slope = safe_slope(
        times,
        ratios
    )

    frequency_slope = safe_slope(
        times,
        frequencies
    )

    midpoint = len(
        times
    ) // 2

    delay_first = safe_mean(
        delays[:midpoint]
    )

    delay_second = safe_mean(
        delays[midpoint:]
    )

    strength_first = safe_mean(
        strengths[:midpoint]
    )

    strength_second = safe_mean(
        strengths[midpoint:]
    )

    frequency_first = safe_mean(
        frequencies[:midpoint]
    )

    frequency_second = safe_mean(
        frequencies[midpoint:]
    )

    return {

        "valid_delay_fraction":
            valid_delay_fraction,

        "direct_strength_mean":
            safe_mean(direct),

        "direct_strength_std":
            safe_std(direct),

        "direct_offset_mean_ms":
            safe_mean(
                direct_offsets
            ),

        "direct_offset_std_ms":
            safe_std(
                direct_offsets
            ),

        "echo_delay_mean_ms":
            safe_mean(delays),

        "echo_delay_std_ms":
            safe_std(delays),

        "echo_delay_slope_ms_s":
            delay_slope,

        "echo_delay_first_ms":
            delay_first,

        "echo_delay_second_ms":
            delay_second,

        "echo_delay_change_ms":
            delay_second -
            delay_first
            if (
                np.isfinite(delay_first)
                and
                np.isfinite(delay_second)
            )
            else np.nan,

        "echo_strength_mean":
            safe_mean(strengths),

        "echo_strength_std":
            safe_std(strengths),

        "echo_strength_slope":
            strength_slope,

        "echo_direct_ratio_mean":
            safe_mean(ratios),

        "echo_direct_ratio_std":
            safe_std(ratios),

        "echo_direct_ratio_slope":
            ratio_slope,

        "echo_snr_mean":
            safe_mean(snr),

        "echo_snr_std":
            safe_std(snr),

        "frequency_mean":
            safe_mean(frequencies),

        "frequency_std":
            safe_std(frequencies),

        "frequency_slope":
            frequency_slope,

        "frequency_change":
            (
                frequency_second -
                frequency_first
            )
            if (
                np.isfinite(
                    frequency_first
                )
                and
                np.isfinite(
                    frequency_second
                )
            )
            else np.nan,

        "candidate_count_mean":
            safe_mean(candidates),

        "candidate_count_std":
            safe_std(candidates)
    }


# ======================================================================
# QUALITY SCORE
# ======================================================================

def calculate_quality(
    features
):

    score = 0.0

    # --------------------------------------------------------------
    # Valid echo measurements
    # --------------------------------------------------------------

    fraction = features[
        "valid_delay_fraction"
    ]

    if np.isfinite(fraction):

        if fraction >= 0.85:
            score += 30

        elif fraction >= 0.60:
            score += 20

        elif fraction >= 0.40:
            score += 10

    # --------------------------------------------------------------
    # Echo/direct ratio
    # --------------------------------------------------------------

    ratio = features[
        "echo_direct_ratio_mean"
    ]

    if np.isfinite(ratio):

        if ratio >= 0.10:
            score += 20

        elif ratio >= 0.05:
            score += 12

        elif ratio >= 0.02:
            score += 6

    # --------------------------------------------------------------
    # Echo SNR
    # --------------------------------------------------------------

    snr = features[
        "echo_snr_mean"
    ]

    if np.isfinite(snr):

        if snr >= 5:
            score += 25

        elif snr >= 3:
            score += 15

        elif snr >= 2:
            score += 8

    # --------------------------------------------------------------
    # Delay stability
    # --------------------------------------------------------------

    delay_std = features[
        "echo_delay_std_ms"
    ]

    if np.isfinite(delay_std):

        if delay_std <= 1.0:
            score += 15

        elif delay_std <= 2.0:
            score += 8

    # --------------------------------------------------------------
    # Candidate stability
    # --------------------------------------------------------------

    candidate_std = features[
        "candidate_count_std"
    ]

    if np.isfinite(candidate_std):

        if candidate_std <= 1.5:
            score += 10

        elif candidate_std <= 3:
            score += 5

    return min(
        score,
        100.0
    )


# ======================================================================
# SAVE DIAGNOSTIC PLOT
# ======================================================================

def save_recording_plot(
    label,
    filename,
    audio,
    filtered,
    mf,
    responses
):

    safe_name = (
        os.path.splitext(
            filename
        )[0]
        .replace(
            " ",
            "_"
        )
    )

    # --------------------------------------------------------------
    # FULL RAW WAVEFORM
    # --------------------------------------------------------------

    duration = (
        len(audio) /
        EXPECTED_SAMPLE_RATE
    )

    time_axis = (
        np.arange(
            len(audio)
        )
        /
        EXPECTED_SAMPLE_RATE
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time_axis,
        audio,
        linewidth=0.7
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        f"{label.upper()} - Raw waveform\n{filename}"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            safe_name +
            "_raw.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------------
    # MATCHED FILTER
    # --------------------------------------------------------------

    mf_time = (
        np.arange(
            len(mf)
        )
        /
        EXPECTED_SAMPLE_RATE
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        mf_time,
        np.abs(mf),
        linewidth=0.8
    )

    for r in responses:

        if np.isfinite(
            r["direct_time"]
        ):

            plt.axvline(
                r["direct_time"],
                linestyle="--",
                alpha=0.35
            )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized correlation"
    )

    plt.title(
        f"{label.upper()} - Matched filter\n{filename}"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            safe_name +
            "_matched_filter.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------------
    # ECHO DELAY
    # --------------------------------------------------------------

    times = np.array(
        [
            r["expected_time"]
            for r in responses
        ]
    )

    delays = np.array(
        [
            r["best_echo_delay_ms"]
            for r in responses
        ]
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        delays,
        marker="o"
    )

    plt.xlabel(
        "Expected chirp time (seconds)"
    )

    plt.ylabel(
        "Candidate echo delay (ms)"
    )

    plt.title(
        f"{label.upper()} - Candidate echo delay\n{filename}"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            safe_name +
            "_delay.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------------
    # ECHO/DIRECT RATIO
    # --------------------------------------------------------------

    ratios = np.array(
        [
            r["echo_to_direct_ratio"]
            for r in responses
        ]
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        ratios,
        marker="o"
    )

    plt.xlabel(
        "Expected chirp time (seconds)"
    )

    plt.ylabel(
        "Echo / direct ratio"
    )

    plt.title(
        f"{label.upper()} - Echo/direct ratio\n{filename}"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            safe_name +
            "_echo_ratio.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------------

    frequencies = np.array(
        [
            r["frequency_centroid"]
            for r in responses
        ]
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        frequencies,
        marker="o"
    )

    plt.xlabel(
        "Expected chirp time (seconds)"
    )

    plt.ylabel(
        "Frequency centroid (Hz)"
    )

    plt.title(
        f"{label.upper()} - Frequency centroid\n{filename}"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            safe_name +
            "_frequency.png"
        ),
        dpi=150
    )

    plt.close()


# ======================================================================
# MAIN ANALYSIS
# ======================================================================

print()
print("=" * 78)
print("V6 DATASET DIAGNOSTIC")
print("=" * 78)

usable = 0
missing = 0
failed = 0

recording_results = []
chirp_results = []

representatives = {}


for index, row in enumerate(
    records
):

    label = row.get(
        "label",
        ""
    ).strip()

    audio_path = find_audio(
        row
    )

    if audio_path is None:

        missing += 1

        print(
            f"WARNING: missing audio "
            f"for record {index + 1}"
        )

        continue

    filename = os.path.basename(
        audio_path
    )

    try:

        audio, filtered, mf, responses = (
            analyze_recording(
                audio_path
            )
        )

        features = recording_features(
            responses
        )

        quality = calculate_quality(
            features
        )

        recording_row = {
            "label": label,
            "wav_file": filename,
            **features,
            "quality_score":
                quality
        }

        recording_results.append(
            recording_row
        )

        for chirp_index, response in enumerate(
            responses
        ):

            chirp_results.append(
                {
                    "label": label,
                    "wav_file": filename,
                    "chirp_number":
                        chirp_index + 1,
                    "expected_time":
                        response[
                            "expected_time"
                        ],
                    "direct_time":
                        response[
                            "direct_time"
                        ],
                    "direct_offset_ms":
                        response[
                            "direct_offset_ms"
                        ],
                    "direct_strength":
                        response[
                            "direct_strength"
                        ],
                    "candidate_count":
                        response[
                            "candidate_count"
                        ],
                    "best_echo_delay_ms":
                        response[
                            "best_echo_delay_ms"
                        ],
                    "best_echo_strength":
                        response[
                            "best_echo_strength"
                        ],
                    "best_echo_relative":
                        response[
                            "best_echo_relative"
                        ],
                    "echo_to_direct_ratio":
                        response[
                            "echo_to_direct_ratio"
                        ],
                    "noise_median":
                        response[
                            "noise_median"
                        ],
                    "echo_snr_proxy":
                        response[
                            "echo_snr_proxy"
                        ],
                    "frequency_centroid":
                        response[
                            "frequency_centroid"
                        ],
                    "frequency_peak":
                        response[
                            "frequency_peak"
                        ],
                    "frequency_spread":
                        response[
                            "frequency_spread"
                        ],
                    "band_energy":
                        response[
                            "band_energy"
                        ]
                }
            )

        usable += 1

        # ----------------------------------------------------------
        # Save one representative recording per class.
        # ----------------------------------------------------------

        if label not in representatives:

            representatives[label] = (
                filename,
                audio,
                filtered,
                mf,
                responses
            )

    except Exception as e:

        failed += 1

        print()
        print(
            "ERROR PROCESSING:"
        )

        print(
            filename
        )

        print(
            str(e)
        )


# ======================================================================
# BASIC DATASET REPORT
# ======================================================================

print()
print("=" * 78)
print("DATASET STATUS")
print("=" * 78)

print()
print(
    f"Usable recordings : {usable}"
)

print(
    f"Missing recordings: {missing}"
)

print(
    f"Failed recordings : {failed}"
)


# ======================================================================
# CLASS DISTRIBUTION
# ======================================================================

labels = [
    row["label"]
    for row in recording_results
]

unique_labels, counts = np.unique(
    labels,
    return_counts=True
)

print()
print("=" * 78)
print("CLASS DISTRIBUTION")
print("=" * 78)

for label, count in zip(
    unique_labels,
    counts
):

    print(
        f"{label:<15}: {count}"
    )


# ======================================================================
# SAVE RECORDING CSV
# ======================================================================

recording_fields = [

    "label",
    "wav_file",

    "valid_delay_fraction",

    "direct_strength_mean",
    "direct_strength_std",

    "direct_offset_mean_ms",
    "direct_offset_std_ms",

    "echo_delay_mean_ms",
    "echo_delay_std_ms",
    "echo_delay_slope_ms_s",

    "echo_delay_first_ms",
    "echo_delay_second_ms",
    "echo_delay_change_ms",

    "echo_strength_mean",
    "echo_strength_std",
    "echo_strength_slope",

    "echo_direct_ratio_mean",
    "echo_direct_ratio_std",
    "echo_direct_ratio_slope",

    "echo_snr_mean",
    "echo_snr_std",

    "frequency_mean",
    "frequency_std",
    "frequency_slope",
    "frequency_change",

    "candidate_count_mean",
    "candidate_count_std",

    "quality_score"
]


with open(
    RECORDING_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=recording_fields
    )

    writer.writeheader()

    for row in recording_results:

        writer.writerow(
            row
        )


# ======================================================================
# SAVE CHIRP CSV
# ======================================================================

chirp_fields = [

    "label",
    "wav_file",
    "chirp_number",

    "expected_time",

    "direct_time",
    "direct_offset_ms",
    "direct_strength",

    "candidate_count",

    "best_echo_delay_ms",
    "best_echo_strength",
    "best_echo_relative",

    "echo_to_direct_ratio",

    "noise_median",
    "echo_snr_proxy",

    "frequency_centroid",
    "frequency_peak",
    "frequency_spread",
    "band_energy"
]


with open(
    CHIRP_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=chirp_fields
    )

    writer.writeheader()

    for row in chirp_results:

        writer.writerow(
            row
        )


# ======================================================================
# CLASS-LEVEL DIAGNOSTICS
# ======================================================================

print()
print("=" * 78)
print("CLASS-LEVEL ACOUSTIC DIAGNOSTICS")
print("=" * 78)


class_summaries = []


for label in unique_labels:

    class_rows = [
        row
        for row in recording_results
        if row["label"] == label
    ]

    def class_values(
        key
    ):

        return [
            row[key]
            for row in class_rows
        ]

    summary = {

        "label":
            label,

        "recordings":
            len(class_rows),

        "valid_delay_fraction":
            safe_mean(
                class_values(
                    "valid_delay_fraction"
                )
            ),

        "direct_strength_mean":
            safe_mean(
                class_values(
                    "direct_strength_mean"
                )
            ),

        "echo_delay_mean_ms":
            safe_mean(
                class_values(
                    "echo_delay_mean_ms"
                )
            ),

        "echo_delay_std_ms":
            safe_mean(
                class_values(
                    "echo_delay_std_ms"
                )
            ),

        "echo_delay_slope_ms_s":
            safe_mean(
                class_values(
                    "echo_delay_slope_ms_s"
                )
            ),

        "echo_delay_change_ms":
            safe_mean(
                class_values(
                    "echo_delay_change_ms"
                )
            ),

        "echo_strength_mean":
            safe_mean(
                class_values(
                    "echo_strength_mean"
                )
            ),

        "echo_direct_ratio_mean":
            safe_mean(
                class_values(
                    "echo_direct_ratio_mean"
                )
            ),

        "echo_snr_mean":
            safe_mean(
                class_values(
                    "echo_snr_mean"
                )
            ),

        "frequency_mean":
            safe_mean(
                class_values(
                    "frequency_mean"
                )
            ),

        "frequency_std":
            safe_mean(
                class_values(
                    "frequency_std"
                )
            ),

        "frequency_slope":
            safe_mean(
                class_values(
                    "frequency_slope"
                )
            ),

        "quality_score":
            safe_mean(
                class_values(
                    "quality_score"
                )
            )
    }

    class_summaries.append(
        summary
    )

    print()
    print(
        f"[{label.upper()}]"
    )

    print(
        f"Recordings              : "
        f"{summary['recordings']}"
    )

    print(
        f"Valid echo fraction     : "
        f"{summary['valid_delay_fraction']:.3f}"
    )

    print(
        f"Direct strength         : "
        f"{summary['direct_strength_mean']:.6f}"
    )

    print(
        f"Echo delay              : "
        f"{summary['echo_delay_mean_ms']:.4f} ms"
    )

    print(
        f"Echo delay std          : "
        f"{summary['echo_delay_std_ms']:.4f} ms"
    )

    print(
        f"Echo delay slope        : "
        f"{summary['echo_delay_slope_ms_s']:.4f} ms/s"
    )

    print(
        f"Echo delay change       : "
        f"{summary['echo_delay_change_ms']:.4f} ms"
    )

    print(
        f"Echo/direct ratio       : "
        f"{summary['echo_direct_ratio_mean']:.6f}"
    )

    print(
        f"Echo SNR proxy          : "
        f"{summary['echo_snr_mean']:.3f}"
    )

    print(
        f"Frequency mean          : "
        f"{summary['frequency_mean']:.2f} Hz"
    )

    print(
        f"Frequency slope         : "
        f"{summary['frequency_slope']:.3f} Hz/s"
    )

    print(
        f"Quality score           : "
        f"{summary['quality_score']:.1f}/100"
    )


# ======================================================================
# SAVE SUMMARY CSV
# ======================================================================

summary_fields = list(
    class_summaries[0].keys()
) if class_summaries else []

with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    if summary_fields:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields
        )

        writer.writeheader()

        for row in class_summaries:

            writer.writerow(
                row
            )


# ======================================================================
# PHYSICAL DISTANCE CONVERSION
# ======================================================================

print()
print("=" * 78)
print("ECHO DELAY → APPROXIMATE DISTANCE")
print("=" * 78)

print()
print(
    "For reference:"
)

print(
    "distance ≈ delay × speed_of_sound / 2"
)

print()

for summary in class_summaries:

    delay = summary[
        "echo_delay_mean_ms"
    ]

    if np.isfinite(delay):

        distance_cm = (
            delay /
            1000.0
            *
            SPEED_OF_SOUND
            /
            2.0
            *
            100.0
        )

        print(
            f"{summary['label']:<15}"
            f": {distance_cm:.2f} cm "
            f"from measured delay"
        )


# ======================================================================
# DIRECTION TEST
# ======================================================================

print()
print("=" * 78)
print("DIRECTIONAL SIGNAL TEST")
print("=" * 78)

summary_map = {
    row["label"]: row
    for row in class_summaries
}


if (
    "approach" in summary_map
    and
    "away" in summary_map
):

    approach = summary_map[
        "approach"
    ]

    away = summary_map[
        "away"
    ]

    approach_slope = approach[
        "echo_delay_slope_ms_s"
    ]

    away_slope = away[
        "echo_delay_slope_ms_s"
    ]

    approach_change = approach[
        "echo_delay_change_ms"
    ]

    away_change = away[
        "echo_delay_change_ms"
    ]

    print()

    print(
        f"Approach delay slope : "
        f"{approach_slope:.6f} ms/s"
    )

    print(
        f"Away delay slope     : "
        f"{away_slope:.6f} ms/s"
    )

    print()

    print(
        f"Approach delay change: "
        f"{approach_change:.6f} ms"
    )

    print(
        f"Away delay change    : "
        f"{away_change:.6f} ms"
    )

    print()

    if (
        np.isfinite(
            approach_slope
        )
        and
        np.isfinite(
            away_slope
        )
        and
        approach_slope < 0
        and
        away_slope > 0
    ):

        print(
            "✓ EXPECTED DIRECTIONAL "
            "DELAY PATTERN FOUND."
        )

        print()
        print(
            "This is the physical signature "
            "we were looking for."
        )

    elif (
        np.isfinite(
            approach_change
        )
        and
        np.isfinite(
            away_change
        )
        and
        approach_change < 0
        and
        away_change > 0
    ):

        print(
            "✓ DIRECTIONAL CHANGE PATTERN FOUND."
        )

        print()
        print(
            "The first-half/second-half "
            "measurements show the expected "
            "direction."
        )

    else:

        print(
            "✗ NO EXPECTED DIRECTIONAL "
            "DELAY SIGNATURE."
        )

        print()
        print(
            "The existing recordings do not "
            "currently demonstrate the expected "
            "approach/away echo-delay behavior."
        )


# ======================================================================
# ECHO QUALITY TEST
# ======================================================================

print()
print("=" * 78)
print("REFLECTION QUALITY TEST")
print("=" * 78)

all_valid_fraction = [
    row["valid_delay_fraction"]
    for row in recording_results
]

all_ratio = [
    row["echo_direct_ratio_mean"]
    for row in recording_results
]

all_snr = [
    row["echo_snr_mean"]
    for row in recording_results
]

overall_valid = safe_mean(
    all_valid_fraction
)

overall_ratio = safe_mean(
    all_ratio
)

overall_snr = safe_mean(
    all_snr
)

print()
print(
    f"Average valid echo fraction : "
    f"{overall_valid:.3f}"
)

print(
    f"Average echo/direct ratio   : "
    f"{overall_ratio:.6f}"
)

print(
    f"Average echo SNR proxy      : "
    f"{overall_snr:.3f}"
)

print()

if (
    np.isfinite(overall_valid)
    and
    overall_valid >= 0.80
    and
    np.isfinite(overall_snr)
    and
    overall_snr >= 3
):

    print(
        "✓ REFLECTION SIGNAL APPEARS "
        "MEASURABLE."
    )

    print()
    print(
        "The acoustic setup is capturing "
        "a reasonably repeatable candidate "
        "reflection."
    )

elif (
    np.isfinite(overall_valid)
    and
    overall_valid >= 0.50
):

    print(
        "⚠ WEAK / INCONSISTENT "
        "REFLECTION SIGNAL."
    )

    print()
    print(
        "There may be usable acoustic "
        "information, but the reflection "
        "measurement is unstable."
    )

else:

    print(
        "✗ REFLECTION SIGNAL IS NOT "
        "RELIABLY DETECTED."
    )

    print()
    print(
        "The current recordings do not "
        "provide enough evidence of a "
        "repeatable hand reflection."
    )


# ======================================================================
# GENERATE REPRESENTATIVE PLOTS
# ======================================================================

print()
print("=" * 78)
print("GENERATING REPRESENTATIVE PLOTS")
print("=" * 78)

for label in [
    "idle",
    "approach",
    "away"
]:

    if label not in representatives:
        continue

    (
        filename,
        audio,
        filtered,
        mf,
        responses
    ) = representatives[label]

    print(
        f"Plotting {label}: {filename}"
    )

    save_recording_plot(
        label,
        filename,
        audio,
        filtered,
        mf,
        responses
    )


# ======================================================================
# GLOBAL CLASS COMPARISON PLOT
# ======================================================================

plt.figure(
    figsize=(10, 6)
)

for summary in class_summaries:

    label = summary[
        "label"
    ]

    delay = summary[
        "echo_delay_mean_ms"
    ]

    if np.isfinite(delay):

        plt.scatter(
            label,
            delay,
            s=100
        )

plt.xlabel(
    "Gesture class"
)

plt.ylabel(
    "Mean candidate echo delay (ms)"
)

plt.title(
    "V6 Mean Candidate Echo Delay by Class"
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    GLOBAL_PLOT,
    dpi=150
)

plt.close()


# ======================================================================
# FINAL V6 DIAGNOSIS
# ======================================================================

print()
print("=" * 78)
print("SARV V6 FINAL DIAGNOSIS")
print("=" * 78)

print()

if (
    np.isfinite(overall_valid)
    and
    overall_valid >= 0.80
    and
    np.isfinite(overall_snr)
    and
    overall_snr >= 3
):

    reflection_status = (
        "MEASURABLE"
    )

elif (
    np.isfinite(overall_valid)
    and
    overall_valid >= 0.50
):

    reflection_status = (
        "WEAK / INCONSISTENT"
    )

else:

    reflection_status = (
        "NOT RELIABLY DETECTED"
    )


direction_status = (
    "NOT VALIDATED"
)


if (
    "approach" in summary_map
    and
    "away" in summary_map
):

    a = summary_map[
        "approach"
    ]

    w = summary_map[
        "away"
    ]

    a_slope = a[
        "echo_delay_slope_ms_s"
    ]

    w_slope = w[
        "echo_delay_slope_ms_s"
    ]

    a_change = a[
        "echo_delay_change_ms"
    ]

    w_change = w[
        "echo_delay_change_ms"
    ]

    if (
        np.isfinite(a_slope)
        and
        np.isfinite(w_slope)
        and
        a_slope < 0
        and
        w_slope > 0
    ):

        direction_status = (
            "VALIDATED BY DELAY SLOPE"
        )

    elif (
        np.isfinite(a_change)
        and
        np.isfinite(w_change)
        and
        a_change < 0
        and
        w_change > 0
    ):

        direction_status = (
            "VALIDATED BY DELAY CHANGE"
        )


print(
    f"Reflection signal : "
    f"{reflection_status}"
)

print(
    f"Direction signal  : "
    f"{direction_status}"
)

print()

# ----------------------------------------------------------------------
# Decision tree
# ----------------------------------------------------------------------

if (
    reflection_status == "MEASURABLE"
    and
    direction_status != "NOT VALIDATED"
):

    print(
        "✓ V6 RESULT: ACOUSTIC DIRECTION "
        "SIGNAL IS PROMISING."
    )

    print()
    print(
        "The existing recordings contain:"
    )

    print(
        "1. A measurable candidate reflection."
    )

    print(
        "2. A directional temporal signature."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Build V7 using temporal tracking "
        "and conservative thresholds."
    )

elif (
    reflection_status == "MEASURABLE"
    and
    direction_status == "NOT VALIDATED"
):

    print(
        "⚠ V6 RESULT: REFLECTION EXISTS, "
        "BUT DIRECTION IS NOT VALIDATED."
    )

    print()
    print(
        "This is actually useful."
    )

    print(
        "The acoustic hardware/setup may be "
        "capturing reflections, but approach "
        "and away cannot yet be separated "
        "physically."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Inspect the V6 delay and frequency "
        "plots before collecting more data."
    )

elif (
    reflection_status == "WEAK / INCONSISTENT"
):

    print(
        "⚠ V6 RESULT: SIGNAL IS TOO WEAK "
        "OR UNSTABLE."
    )

    print()
    print(
        "Do NOT train another classifier."
    )

    print(
        "Do NOT collect another large dataset yet."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Inspect the representative raw "
        "waveform and matched-filter plots."
    )

else:

    print(
        "✗ V6 RESULT: NO RELIABLE "
        "REFLECTION SIGNAL."
    )

    print()
    print(
        "The current acoustic setup does not "
        "provide sufficient evidence that the "
        "hand reflection is being measured."
    )

    print()
    print(
        "Do NOT collect another large hand dataset."
    )

    print(
        "The acoustic hardware/configuration "
        "should be reconsidered first."
    )


# ======================================================================
# OUTPUT FILES
# ======================================================================

print()
print("=" * 78)
print("V6 OUTPUT FILES")
print("=" * 78)

print()
print(
    "Recording diagnostics:"
)

print(
    RECORDING_CSV
)

print()
print(
    "Per-chirp diagnostics:"
)

print(
    CHIRP_CSV
)

print()
print(
    "Class summary:"
)

print(
    SUMMARY_CSV
)

print()
print(
    "Analysis plots:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 78)
print("SARV V6 ANALYSIS COMPLETE")
print("=" * 78)