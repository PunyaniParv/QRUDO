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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

# The diagnostics below use Unicode symbols (✓ ✗ ⚠) that the
# default Windows console encoding (cp1252) cannot print.
# Reconfigure stdout so the script runs completely under the
# installed environment without requiring a console change.
try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )
except Exception:
    pass

warnings.filterwarnings("ignore")


# ======================================================================
# SARV ACOUSTIC GESTURE ANALYZER V8
#
# EXISTING DATASET MOTION ANALYSIS
#
# V8 objective:
#
#   V5 -> basic echo timing
#   V6 -> reflection validation
#   V7 -> multi-peak reflection tracking
#   V8 -> MOTION EXTRACTION
#
# V8 does NOT collect new recordings.
#
# It re-analyzes the existing V5 dataset and asks:
#
#   "Does the tracked acoustic reflection contain a systematic
#    temporal motion signature for APPROACH, AWAY and IDLE?"
#
# Main measurements:
#
#   1. Sub-sample echo delay
#   2. Delay velocity
#   3. Delay acceleration
#   4. Reflection strength velocity
#   5. Reflection strength change
#   6. Peak trajectory stability
#   7. Peak-to-peak movement
#   8. Frequency centroid movement
#   9. Frequency velocity
#  10. Candidate peak consistency
#
# IMPORTANT:
#
# This is an experimental signal-analysis system.
# Classification accuracy alone is NOT considered proof of physical
# direction detection.
#
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
    "analysis_v8"
)

PLOTS_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ======================================================================
# AUDIO CONFIGURATION
# ======================================================================

SAMPLE_RATE = 44100

CHIRP_LOW = 7500
CHIRP_HIGH = 8500

CHIRP_DURATION = 0.100

FIRST_CHIRP_TIME = 0.15
CHIRP_INTERVAL = 0.250

EXPECTED_CHIRPS = 7

SPEED_OF_SOUND = 343.0

FILTER_LOW = 6000
FILTER_HIGH = 11000

# Reflection search range.
#
# We deliberately search farther than V5.
#
# 1 ms ≈ 17.15 cm round-trip distance
#
# 15 ms ≈ 2.57 m round-trip distance
#
# This gives the system room to observe anomalous peaks while
# still allowing the tracker to reject unstable candidates.
ECHO_MIN_DELAY_MS = 0.60
ECHO_MAX_DELAY_MS = 18.0

# Candidate peak detection.
PEAK_DISTANCE_MS = 0.25

# Relative peak threshold.
#
# Candidate peaks below this fraction of the strongest candidate
# are ignored.
PEAK_RELATIVE_THRESHOLD = 0.18

# Tracker maximum movement between consecutive chirps.
MAX_TRACK_JUMP_MS = 5.0

# Maximum number of candidate peaks kept per chirp.
MAX_CANDIDATE_PEAKS = 12

# Sub-sample interpolation is enabled.
SUBSAMPLE_INTERPOLATION = True

# Frequency analysis.
FREQ_LOW = 6500
FREQ_HIGH = 10000


# ======================================================================
# HEADER
# ======================================================================

print("=" * 78)
print("SARV ACOUSTIC GESTURE ANALYZER V8")
print("SUB-SAMPLE REFLECTION MOTION + TRAJECTORY ANALYSIS")
print("=" * 78)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Audio:")
print(AUDIO_DIR)

print()
print("V8 output:")
print(OUTPUT_DIR)


# ======================================================================
# VALIDATE DATASET
# ======================================================================

if not os.path.exists(METADATA_FILE):

    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE}"
    )

if not os.path.exists(AUDIO_DIR):

    raise FileNotFoundError(
        f"Audio directory not found:\n{AUDIO_DIR}"
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
# FIND AUDIO FILE
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
# LABEL FALLBACK
# ======================================================================

def infer_label_from_filename(filename):

    lower = filename.lower()

    if lower.startswith("approach"):
        return "approach"

    if lower.startswith("away"):
        return "away"

    if lower.startswith("idle"):
        return "idle"

    return "unknown"


# ======================================================================
# GENERATE REFERENCE CHIRP
# ======================================================================

def generate_reference_chirp():

    n = int(
        CHIRP_DURATION *
        SAMPLE_RATE
    )

    t = np.arange(n) / SAMPLE_RATE

    k = (
        CHIRP_HIGH -
        CHIRP_LOW
    ) / CHIRP_DURATION

    phase = 2.0 * np.pi * (
        CHIRP_LOW * t +
        0.5 * k * t * t
    )

    signal = np.sin(phase)

    signal *= np.hanning(n)

    signal = signal.astype(
        np.float64
    )

    norm = np.linalg.norm(
        signal
    )

    if norm > 0:
        signal /= norm

    return signal


REFERENCE_CHIRP = (
    generate_reference_chirp()
)


# ======================================================================
# BANDPASS
# ======================================================================

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

    return data, sr


# ======================================================================
# NORMALIZED MATCHED FILTER
# ======================================================================

def matched_filter(
    signal,
    reference
):

    reference = (
        reference -
        np.mean(reference)
    )

    signal = (
        signal -
        np.mean(signal)
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

    ref_len = len(reference)

    squared = signal ** 2

    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(squared)
        ]
    )

    window_energy = (
        cumulative[ref_len:]
        -
        cumulative[:-ref_len]
    )

    denominator = (
        np.sqrt(
            np.maximum(
                window_energy,
                1e-18
            )
        )
        *
        ref_norm
        +
        1e-12
    )

    return (
        correlation /
        denominator
    )


# ======================================================================
# EXPECTED CHIRP TIMES
# ======================================================================

def expected_chirp_times():

    return np.array(
        [
            FIRST_CHIRP_TIME +
            i * CHIRP_INTERVAL
            for i in range(
                EXPECTED_CHIRPS
            )
        ],
        dtype=np.float64
    )


# ======================================================================
# SUB-SAMPLE PEAK INTERPOLATION
# ======================================================================

def refine_peak_subsample(
    values,
    index
):

    index = int(index)

    if not SUBSAMPLE_INTERPOLATION:
        return float(index)

    if index <= 0:
        return float(index)

    if index >= len(values) - 1:
        return float(index)

    y1 = float(
        values[index - 1]
    )

    y2 = float(
        values[index]
    )

    y3 = float(
        values[index + 1]
    )

    denominator = (
        y1 -
        2.0 * y2 +
        y3
    )

    if abs(denominator) < 1e-12:
        return float(index)

    offset = (
        0.5 *
        (
            y1 - y3
        )
        /
        denominator
    )

    offset = np.clip(
        offset,
        -0.5,
        0.5
    )

    return float(
        index + offset
    )


# ======================================================================
# CANDIDATE PEAK EXTRACTION
# ======================================================================

def extract_candidate_peaks(
    mf,
    expected_index
):

    min_offset = int(
        ECHO_MIN_DELAY_MS /
        1000.0 *
        SAMPLE_RATE
    )

    max_offset = int(
        ECHO_MAX_DELAY_MS /
        1000.0 *
        SAMPLE_RATE
    )

    start = max(
        0,
        int(
            expected_index +
            min_offset
        )
    )

    end = min(
        len(mf),
        int(
            expected_index +
            max_offset
        )
    )

    if end <= start + 3:

        return []

    region = np.abs(
        mf[start:end]
    )

    if len(region) < 5:
        return []

    maximum = float(
        np.max(region)
    )

    if maximum <= 0:
        return []

    min_distance = max(
        1,
        int(
            PEAK_DISTANCE_MS /
            1000.0 *
            SAMPLE_RATE
        )
    )

    threshold = (
        maximum *
        PEAK_RELATIVE_THRESHOLD
    )

    peaks, properties = find_peaks(
        region,
        height=threshold,
        distance=min_distance
    )

    if len(peaks) == 0:

        # Fallback to strongest point.
        strongest = int(
            np.argmax(region)
        )

        peaks = np.array(
            [strongest]
        )

    candidates = []

    for peak in peaks:

        absolute_index = (
            start +
            int(peak)
        )

        refined = (
            refine_peak_subsample(
                np.abs(mf),
                absolute_index
            )
        )

        delay_samples = (
            refined -
            expected_index
        )

        delay_ms = (
            delay_samples /
            SAMPLE_RATE *
            1000.0
        )

        strength = float(
            region[peak]
        )

        candidates.append(
            {
                "index":
                    refined,

                "delay_ms":
                    float(delay_ms),

                "strength":
                    strength
            }
        )

    candidates.sort(
        key=lambda x: x["strength"],
        reverse=True
    )

    return candidates[
        :MAX_CANDIDATE_PEAKS
    ]


# ======================================================================
# TRACK REFLECTION ACROSS CHIRPS
# ======================================================================

def track_reflection(
    candidates_per_chirp
):

    if len(
        candidates_per_chirp
    ) == 0:

        return []

    track = []

    previous_delay = None

    for chirp_index, candidates in enumerate(
        candidates_per_chirp
    ):

        if not candidates:

            track.append(
                None
            )

            continue

        # --------------------------------------------------------------
        # First chirp:
        #
        # Choose strongest candidate.
        # --------------------------------------------------------------

        if previous_delay is None:

            chosen = max(
                candidates,
                key=lambda x:
                x["strength"]
            )

        else:

            # ----------------------------------------------------------
            # Prefer a candidate close to previous delay.
            # Penalize large jumps.
            # ----------------------------------------------------------

            scored = []

            for candidate in candidates:

                jump = abs(
                    candidate["delay_ms"]
                    -
                    previous_delay
                )

                if jump <= MAX_TRACK_JUMP_MS:

                    score = (
                        candidate["strength"]
                        /
                        (
                            1.0 +
                            jump
                        )
                    )

                    scored.append(
                        (
                            score,
                            candidate
                        )
                    )

            if scored:

                scored.sort(
                    key=lambda x:
                    x[0],
                    reverse=True
                )

                chosen = scored[0][1]

            else:

                # No close candidate.
                # Use strongest candidate, but this is a
                # tracker break.
                chosen = max(
                    candidates,
                    key=lambda x:
                    x["strength"]
                )

        track.append(
            chosen
        )

        previous_delay = (
            chosen["delay_ms"]
        )

    return track


# ======================================================================
# ROBUST SLOPE
# ======================================================================

def safe_slope(
    x,
    y
):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    valid = np.isfinite(y)

    if np.sum(valid) < 2:
        return 0.0

    x = x[valid]
    y = y[valid]

    try:

        return float(
            np.polyfit(
                x,
                y,
                1
            )[0]
        )

    except Exception:

        return 0.0


# ======================================================================
# SAFE MEAN
# ======================================================================

def safe_mean(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 0.0

    return float(
        np.mean(values)
    )


# ======================================================================
# SAFE STD
# ======================================================================

def safe_std(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 0.0

    return float(
        np.std(values)
    )


# ======================================================================
# SAFE MEDIAN
# ======================================================================

def safe_median(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 0.0

    return float(
        np.median(values)
    )


# ======================================================================
# DIFFERENCE FEATURES
# ======================================================================

def calculate_difference_features(
    times,
    values
):

    times = np.asarray(
        times,
        dtype=np.float64
    )

    values = np.asarray(
        values,
        dtype=np.float64
    )

    valid = np.isfinite(values)

    times = times[valid]
    values = values[valid]

    if len(values) < 2:

        return {
            "velocity_mean": 0.0,
            "velocity_std": 0.0,
            "velocity_slope": 0.0,
            "acceleration_mean": 0.0,
            "acceleration_std": 0.0,
            "net_change": 0.0,
            "range": 0.0
        }

    velocity = np.diff(values) / (
        np.diff(times) +
        1e-12
    )

    if len(velocity) >= 2:

        acceleration = np.diff(
            velocity
        ) / (
            np.diff(times[:-1]) +
            1e-12
        )

    else:

        acceleration = np.array(
            []
        )

    return {
        "velocity_mean":
            safe_mean(velocity),

        "velocity_std":
            safe_std(velocity),

        "velocity_slope":
            safe_slope(
                times[1:],
                velocity
            ),

        "acceleration_mean":
            safe_mean(acceleration),

        "acceleration_std":
            safe_std(acceleration),

        "net_change":
            float(
                values[-1] -
                values[0]
            ),

        "range":
            float(
                np.max(values) -
                np.min(values)
            )
    }


# ======================================================================
# FREQUENCY FEATURES
# ======================================================================

def frequency_features(
    signal,
    chirp_time
):

    center = int(
        chirp_time *
        SAMPLE_RATE
    )

    half_window = int(
        0.060 *
        SAMPLE_RATE
    )

    start = max(
        0,
        center -
        half_window
    )

    end = min(
        len(signal),
        center +
        half_window
    )

    segment = signal[
        start:end
    ]

    if len(segment) < 1024:

        return {
            "frequency_centroid": 0.0,
            "frequency_peak": 0.0,
            "frequency_std": 0.0
        }

    frequencies, times, Sxx = spectrogram(
        segment,
        fs=SAMPLE_RATE,
        nperseg=1024,
        noverlap=768,
        scaling="spectrum",
        mode="magnitude"
    )

    mask = (
        (frequencies >= FREQ_LOW)
        &
        (frequencies <= FREQ_HIGH)
    )

    if not np.any(mask):

        return {
            "frequency_centroid": 0.0,
            "frequency_peak": 0.0,
            "frequency_std": 0.0
        }

    freqs = frequencies[
        mask
    ]

    power = Sxx[
        mask
    ]

    mean_power = np.mean(
        power,
        axis=1
    )

    total = (
        np.sum(mean_power)
        +
        1e-12
    )

    centroid = (
        np.sum(
            freqs *
            mean_power
        )
        /
        total
    )

    peak = float(
        freqs[
            np.argmax(
                mean_power
            )
        ]
    )

    variance = (
        np.sum(
            (
                freqs -
                centroid
            ) ** 2
            *
            mean_power
        )
        /
        total
    )

    return {
        "frequency_centroid":
            float(centroid),

        "frequency_peak":
            peak,

        "frequency_std":
            float(
                np.sqrt(
                    max(
                        variance,
                        0.0
                    )
                )
            )
    }


# ======================================================================
# ANALYZE ONE RECORDING
# ======================================================================

def analyze_recording(
    audio_path
):

    audio, sr = load_audio(
        audio_path
    )

    if sr != SAMPLE_RATE:

        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, "
            f"got {sr} Hz"
        )

    if len(audio) < SAMPLE_RATE:

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

    chirp_times = (
        expected_chirp_times()
    )

    candidates_per_chirp = []

    chirp_results = []

    reference_length = len(
        REFERENCE_CHIRP
    )

    for chirp_time in chirp_times:

        sample_center = int(
            chirp_time *
            SAMPLE_RATE
        )

        expected_index = (
            sample_center
            -
            reference_length // 2
        )

        candidates = (
            extract_candidate_peaks(
                mf,
                expected_index
            )
        )

        candidates_per_chirp.append(
            candidates
        )

        freq = frequency_features(
            filtered,
            chirp_time
        )

        chirp_results.append(
            {
                "time":
                    float(chirp_time),

                "expected_index":
                    expected_index,

                "candidates":
                    candidates,

                "frequency":
                    freq
            }
        )

    track = track_reflection(
        candidates_per_chirp
    )

    # Attach tracked peak.
    for i, result in enumerate(
        chirp_results
    ):

        if i < len(track):

            result[
                "tracked"
            ] = track[i]

        else:

            result[
                "tracked"
            ] = None

    return chirp_results


# ======================================================================
# EXTRACT RECORDING FEATURES
# ======================================================================

def calculate_recording_features(
    chirps
):

    times = np.array(
        [
            c["time"]
            for c in chirps
        ],
        dtype=np.float64
    )

    delays = np.array(
        [
            (
                c["tracked"]["delay_ms"]
                if c["tracked"] is not None
                else np.nan
            )
            for c in chirps
        ],
        dtype=np.float64
    )

    strengths = np.array(
        [
            (
                c["tracked"]["strength"]
                if c["tracked"] is not None
                else np.nan
            )
            for c in chirps
        ],
        dtype=np.float64
    )

    frequencies = np.array(
        [
            c[
                "frequency"
            ][
                "frequency_centroid"
            ]
            for c in chirps
        ],
        dtype=np.float64
    )

    frequency_std = np.array(
        [
            c[
                "frequency"
            ][
                "frequency_std"
            ]
            for c in chirps
        ],
        dtype=np.float64
    )

    valid_delay = np.isfinite(
        delays
    )

    track_fraction = (
        np.sum(valid_delay)
        /
        max(
            len(delays),
            1
        )
    )

    delay_features = (
        calculate_difference_features(
            times,
            delays
        )
    )

    strength_features = (
        calculate_difference_features(
            times,
            strengths
        )
    )

    frequency_features_temporal = (
        calculate_difference_features(
            times,
            frequencies
        )
    )

    # --------------------------------------------------------------
    # First half / second half
    # --------------------------------------------------------------

    midpoint = len(times) // 2

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

    # --------------------------------------------------------------
    # Delay -> approximate distance
    #
    # distance = delay * c / 2
    # --------------------------------------------------------------

    median_delay = safe_median(
        delays
    )

    distance_cm = (
        median_delay /
        1000.0 *
        SPEED_OF_SOUND /
        2.0 *
        100.0
    )

    # --------------------------------------------------------------
    # Candidate count
    # --------------------------------------------------------------

    candidate_counts = np.array(
        [
            len(
                c["candidates"]
            )
            for c in chirps
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------------
    # Strongest-vs-tracked stability
    # --------------------------------------------------------------

    strongest_matches = 0

    for c in chirps:

        candidates = c[
            "candidates"
        ]

        tracked = c[
            "tracked"
        ]

        if (
            candidates
            and
            tracked is not None
        ):

            strongest = max(
                candidates,
                key=lambda x:
                x["strength"]
            )

            if abs(
                strongest["delay_ms"]
                -
                tracked["delay_ms"]
            ) < 0.25:

                strongest_matches += 1

    strongest_match_fraction = (
        strongest_matches /
        max(
            len(chirps),
            1
        )
    )

    # --------------------------------------------------------------
    # Sign consistency
    #
    # Negative delay velocity:
    # reflection delay decreasing.
    #
    # Positive delay velocity:
    # reflection delay increasing.
    # --------------------------------------------------------------

    valid_velocities = []

    valid_delay_values = delays[
        np.isfinite(delays)
    ]

    if len(valid_delay_values) >= 2:

        dt = np.diff(
            times
        )

        dd = np.diff(
            valid_delay_values
        )

        valid_dt = dt[
            :len(dd)
        ]

        valid_velocities = (
            dd /
            (
                valid_dt +
                1e-12
            )
        )

    valid_velocities = np.asarray(
        valid_velocities,
        dtype=np.float64
    )

    positive_fraction = 0.0
    negative_fraction = 0.0

    if len(valid_velocities):

        positive_fraction = (
            np.sum(
                valid_velocities > 0
            )
            /
            len(valid_velocities)
        )

        negative_fraction = (
            np.sum(
                valid_velocities < 0
            )
            /
            len(valid_velocities)
        )

    # --------------------------------------------------------------
    # Motion energy
    # --------------------------------------------------------------

    delay_velocity_energy = (
        safe_mean(
            valid_velocities ** 2
        )
        if len(valid_velocities)
        else 0.0
    )

    # --------------------------------------------------------------
    # Reflection strength normalized by itself
    # --------------------------------------------------------------

    strength_mean = safe_mean(
        strengths
    )

    strength_std = safe_std(
        strengths
    )

    strength_cv = (
        strength_std /
        (
            abs(strength_mean)
            +
            1e-12
        )
    )

    # --------------------------------------------------------------
    # Frequency change
    # --------------------------------------------------------------

    frequency_change = (
        frequency_second -
        frequency_first
    )

    # --------------------------------------------------------------
    # Delay change
    # --------------------------------------------------------------

    delay_change = (
        delay_second -
        delay_first
    )

    # --------------------------------------------------------------
    # A simple physical-motion score.
    #
    # This is NOT a gesture confidence score.
    # It simply quantifies how much the tracked reflection moves.
    # --------------------------------------------------------------

    motion_score = (
        min(
            100.0,
            abs(
                delay_features[
                    "net_change"
                ]
            ) * 20.0
            +
            delay_features[
                "range"
            ] * 10.0
            +
            delay_features[
                "velocity_std"
            ] * 2.0
        )
    )

    return {
        "track_fraction":
            float(track_fraction),

        "median_delay_ms":
            median_delay,

        "delay_mean_ms":
            safe_mean(delays),

        "delay_std_ms":
            safe_std(delays),

        "delay_velocity_mean_ms_s":
            delay_features[
                "velocity_mean"
            ],

        "delay_velocity_std_ms_s":
            delay_features[
                "velocity_std"
            ],

        "delay_velocity_slope":
            delay_features[
                "velocity_slope"
            ],

        "delay_acceleration_mean":
            delay_features[
                "acceleration_mean"
            ],

        "delay_acceleration_std":
            delay_features[
                "acceleration_std"
            ],

        "delay_net_change_ms":
            delay_features[
                "net_change"
            ],

        "delay_range_ms":
            delay_features[
                "range"
            ],

        "delay_first_ms":
            delay_first,

        "delay_second_ms":
            delay_second,

        "delay_change_ms":
            delay_change,

        "positive_delay_velocity_fraction":
            positive_fraction,

        "negative_delay_velocity_fraction":
            negative_fraction,

        "delay_motion_energy":
            delay_velocity_energy,

        "distance_cm":
            distance_cm,

        "strength_mean":
            strength_mean,

        "strength_std":
            strength_std,

        "strength_cv":
            strength_cv,

        "strength_velocity_mean":
            strength_features[
                "velocity_mean"
            ],

        "strength_velocity_std":
            strength_features[
                "velocity_std"
            ],

        "strength_net_change":
            strength_features[
                "net_change"
            ],

        "strength_range":
            strength_features[
                "range"
            ],

        "strength_first":
            strength_first,

        "strength_second":
            strength_second,

        "strength_change":
            strength_second -
            strength_first,

        "frequency_mean":
            safe_mean(frequencies),

        "frequency_std":
            safe_mean(
                frequency_std
            ),

        "frequency_velocity_mean":
            frequency_features_temporal[
                "velocity_mean"
            ],

        "frequency_velocity_std":
            frequency_features_temporal[
                "velocity_std"
            ],

        "frequency_net_change":
            frequency_features_temporal[
                "net_change"
            ],

        "frequency_range":
            frequency_features_temporal[
                "range"
            ],

        "frequency_first":
            frequency_first,

        "frequency_second":
            frequency_second,

        "frequency_change":
            frequency_change,

        "candidate_count_mean":
            safe_mean(
                candidate_counts
            ),

        "candidate_count_std":
            safe_std(
                candidate_counts
            ),

        "strongest_match_fraction":
            strongest_match_fraction,

        "motion_score":
            motion_score
    }


# ======================================================================
# DATASET ANALYSIS
# ======================================================================

print()
print("=" * 78)
print("V8 DATASET ANALYSIS")
print("=" * 78)

all_results = []

missing = 0
failed = 0

for index, row in enumerate(
    records
):

    audio_path = find_audio(
        row
    )

    if audio_path is None:

        missing += 1

        continue

    filename = os.path.basename(
        audio_path
    )

    label = row.get(
        "label",
        ""
    ).strip().lower()

    if not label:

        label = infer_label_from_filename(
            filename
        )

    try:

        print(
            f"[{index + 1}/{len(records)}] "
            f"{filename}"
        )

        chirps = analyze_recording(
            audio_path
        )

        features = (
            calculate_recording_features(
                chirps
            )
        )

        all_results.append(
            {
                "label": label,
                "filename": filename,
                "chirps": chirps,
                "features": features
            }
        )

        print(
            f"    track="
            f"{features['track_fraction']:.2f} "
            f"delay="
            f"{features['median_delay_ms']:.3f}ms "
            f"net="
            f"{features['delay_net_change_ms']:+.3f}ms "
            f"velocity="
            f"{features['delay_velocity_mean_ms_s']:+.3f}ms/s "
            f"motion="
            f"{features['motion_score']:.1f}"
        )

    except Exception as e:

        failed += 1

        print(
            "    ERROR:",
            e
        )


print()
print(
    f"Usable recordings : {len(all_results)}"
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

print()
print("=" * 78)
print("V8 CLASS DISTRIBUTION")
print("=" * 78)

labels = [
    r["label"]
    for r in all_results
]

unique_labels = sorted(
    set(labels)
)

for label in [
    "approach",
    "away",
    "idle",
    "unknown"
]:

    count = labels.count(
        label
    )

    print(
        f"{label:<12}: {count}"
    )


# ======================================================================
# FEATURE LIST
# ======================================================================

feature_names = [

    "track_fraction",

    "median_delay_ms",
    "delay_mean_ms",
    "delay_std_ms",

    "delay_velocity_mean_ms_s",
    "delay_velocity_std_ms_s",
    "delay_velocity_slope",

    "delay_acceleration_mean",
    "delay_acceleration_std",

    "delay_net_change_ms",
    "delay_range_ms",

    "delay_first_ms",
    "delay_second_ms",
    "delay_change_ms",

    "positive_delay_velocity_fraction",
    "negative_delay_velocity_fraction",

    "delay_motion_energy",

    "distance_cm",

    "strength_mean",
    "strength_std",
    "strength_cv",

    "strength_velocity_mean",
    "strength_velocity_std",

    "strength_net_change",
    "strength_range",

    "strength_first",
    "strength_second",
    "strength_change",

    "frequency_mean",
    "frequency_std",

    "frequency_velocity_mean",
    "frequency_velocity_std",

    "frequency_net_change",
    "frequency_range",

    "frequency_first",
    "frequency_second",
    "frequency_change",

    "candidate_count_mean",
    "candidate_count_std",

    "strongest_match_fraction",

    "motion_score"
]


# ======================================================================
# CLASS STATISTICS
# ======================================================================

print()
print("=" * 78)
print("V8 CLASS-LEVEL MOTION STATISTICS")
print("=" * 78)

class_summaries = []

for label in [
    "approach",
    "away",
    "idle"
]:

    items = [
        r
        for r in all_results
        if r["label"] == label
    ]

    if not items:
        continue

    print()
    print(
        f"[{label.upper()}]"
    )

    summary = {
        "class": label,
        "recordings": len(items)
    }

    for feature in feature_names:

        values = np.array(
            [
                r[
                    "features"
                ][
                    feature
                ]
                for r in items
            ],
            dtype=np.float64
        )

        mean_value = safe_mean(
            values
        )

        median_value = safe_median(
            values
        )

        std_value = safe_std(
            values
        )

        summary[
            feature + "_mean"
        ] = mean_value

        summary[
            feature + "_median"
        ] = median_value

        summary[
            feature + "_std"
        ] = std_value

    print(
        f"Recordings                    : "
        f"{len(items)}"
    )

    print(
        f"Track fraction                : "
        f"{summary['track_fraction_mean']:.3f}"
    )

    print(
        f"Median delay                  : "
        f"{summary['median_delay_ms_median']:.3f} ms"
    )

    print(
        f"Delay std                     : "
        f"{summary['delay_std_ms_median']:.3f} ms"
    )

    print(
        f"Delay velocity                : "
        f"{summary['delay_velocity_mean_ms_s_mean']:+.4f} ms/s"
    )

    print(
        f"Delay velocity std            : "
        f"{summary['delay_velocity_std_ms_s_mean']:.4f} ms/s"
    )

    print(
        f"Delay acceleration            : "
        f"{summary['delay_acceleration_mean_mean']:+.4f}"
    )

    print(
        f"Delay net change              : "
        f"{summary['delay_net_change_ms_mean']:+.4f} ms"
    )

    print(
        f"Delay range                   : "
        f"{summary['delay_range_ms_mean']:.4f} ms"
    )

    print(
        f"Positive velocity fraction    : "
        f"{summary['positive_delay_velocity_fraction_mean']:.3f}"
    )

    print(
        f"Negative velocity fraction    : "
        f"{summary['negative_delay_velocity_fraction_mean']:.3f}"
    )

    print(
        f"Frequency net change          : "
        f"{summary['frequency_net_change_mean']:+.2f} Hz"
    )

    print(
        f"Strength net change           : "
        f"{summary['strength_net_change_mean']:+.6f}"
    )

    print(
        f"Motion score                  : "
        f"{summary['motion_score_mean']:.2f}"
    )

    class_summaries.append(
        summary
    )


# ======================================================================
# PHYSICAL DIRECTION TEST
# ======================================================================

print()
print("=" * 78)
print("V8 PHYSICAL DIRECTION TEST")
print("=" * 78)

approach_items = [
    r
    for r in all_results
    if r["label"] == "approach"
]

away_items = [
    r
    for r in all_results
    if r["label"] == "away"
]

idle_items = [
    r
    for r in all_results
    if r["label"] == "idle"
]


def class_feature_mean(
    items,
    feature
):

    if not items:
        return np.nan

    return safe_mean(
        [
            r[
                "features"
            ][
                feature
            ]
            for r in items
        ]
    )


def class_feature_median(
    items,
    feature
):

    if not items:
        return np.nan

    return safe_median(
        [
            r[
                "features"
            ][
                feature
            ]
            for r in items
        ]
    )


approach_velocity = class_feature_mean(
    approach_items,
    "delay_velocity_mean_ms_s"
)

away_velocity = class_feature_mean(
    away_items,
    "delay_velocity_mean_ms_s"
)

idle_velocity = class_feature_mean(
    idle_items,
    "delay_velocity_mean_ms_s"
)

approach_net = class_feature_mean(
    approach_items,
    "delay_net_change_ms"
)

away_net = class_feature_mean(
    away_items,
    "delay_net_change_ms"
)

idle_net = class_feature_mean(
    idle_items,
    "delay_net_change_ms"
)

approach_positive = class_feature_mean(
    approach_items,
    "positive_delay_velocity_fraction"
)

away_positive = class_feature_mean(
    away_items,
    "positive_delay_velocity_fraction"
)

idle_positive = class_feature_mean(
    idle_items,
    "positive_delay_velocity_fraction"
)

approach_negative = class_feature_mean(
    approach_items,
    "negative_delay_velocity_fraction"
)

away_negative = class_feature_mean(
    away_items,
    "negative_delay_velocity_fraction"
)

idle_negative = class_feature_mean(
    idle_items,
    "negative_delay_velocity_fraction"
)


print()

print(
    f"Approach delay velocity : "
    f"{approach_velocity:+.5f} ms/s"
)

print(
    f"Away delay velocity     : "
    f"{away_velocity:+.5f} ms/s"
)

print(
    f"Idle delay velocity     : "
    f"{idle_velocity:+.5f} ms/s"
)

print()

print(
    f"Approach net delay      : "
    f"{approach_net:+.5f} ms"
)

print(
    f"Away net delay          : "
    f"{away_net:+.5f} ms"
)

print(
    f"Idle net delay          : "
    f"{idle_net:+.5f} ms"
)

print()

print(
    f"Approach positive frac  : "
    f"{approach_positive:.3f}"
)

print(
    f"Away positive frac      : "
    f"{away_positive:.3f}"
)

print(
    f"Idle positive frac      : "
    f"{idle_positive:.3f}"
)

print()

print(
    f"Approach negative frac  : "
    f"{approach_negative:.3f}"
)

print(
    f"Away negative frac      : "
    f"{away_negative:.3f}"
)

print(
    f"Idle negative frac      : "
    f"{idle_negative:.3f}"
)


# ======================================================================
# DIRECTIONAL SEPARATION SCORE
# ======================================================================

print()
print("=" * 78)
print("DIRECTIONAL SEPARATION ANALYSIS")
print("=" * 78)

velocity_separation = abs(
    approach_velocity -
    away_velocity
)

net_separation = abs(
    approach_net -
    away_net
)

positive_separation = abs(
    approach_positive -
    away_positive
)

negative_separation = abs(
    approach_negative -
    away_negative
)

print()
print(
    f"Velocity separation : "
    f"{velocity_separation:.5f} ms/s"
)

print(
    f"Net delay separation: "
    f"{net_separation:.5f} ms"
)

print(
    f"Positive fraction separation: "
    f"{positive_separation:.3f}"
)

print(
    f"Negative fraction separation: "
    f"{negative_separation:.3f}"
)


# ======================================================================
# SIGN TEST
# ======================================================================

directional_sign_pattern = (
    approach_velocity < 0
    and
    away_velocity > 0
)

opposite_sign_pattern = (
    approach_velocity > 0
    and
    away_velocity < 0
)

print()

if directional_sign_pattern:

    print(
        "✓ EXPECTED APPROACH/AWAY DELAY-VELOCITY "
        "SIGN PATTERN DETECTED."
    )

    print()
    print(
        "Approach tends toward decreasing delay."
    )

    print(
        "Away tends toward increasing delay."
    )

elif opposite_sign_pattern:

    print(
        "⚠ OPPOSITE SIGN PATTERN DETECTED."
    )

    print()
    print(
        "The measured direction is opposite to "
        "the expected physical interpretation."
    )

else:

    print(
        "✗ NO CLEAN APPROACH/AWAY SIGN PATTERN."
    )

    print()
    print(
        "Approach and away do not currently show "
        "opposite mean delay velocity signs."
    )


# ======================================================================
# MOTION VS IDLE TEST
# ======================================================================

print()
print("=" * 78)
print("MOTION VS IDLE TEST")
print("=" * 78)

approach_motion = class_feature_mean(
    approach_items,
    "motion_score"
)

away_motion = class_feature_mean(
    away_items,
    "motion_score"
)

idle_motion = class_feature_mean(
    idle_items,
    "motion_score"
)

print()
print(
    f"Approach motion score : "
    f"{approach_motion:.3f}"
)

print(
    f"Away motion score     : "
    f"{away_motion:.3f}"
)

print(
    f"Idle motion score     : "
    f"{idle_motion:.3f}"
)

if (
    approach_motion > idle_motion
    and
    away_motion > idle_motion
):

    print()
    print(
        "✓ Motion appears stronger than idle."
    )

else:

    print()
    print(
        "⚠ Motion score does not cleanly separate "
        "movement from idle."
    )


# ======================================================================
# RANDOM FOREST CLASSIFICATION
# ======================================================================

print()
print("=" * 78)
print("3-CLASS MOTION FEATURE CLASSIFICATION")
print("=" * 78)

usable_results = [
    r
    for r in all_results
    if r["label"] in [
        "approach",
        "away",
        "idle"
    ]
]

if len(usable_results) >= 6:

    X = np.array(
        [
            [
                r[
                    "features"
                ][
                    feature
                ]
                for feature in feature_names
            ]
            for r in usable_results
        ],
        dtype=np.float64
    )

    y = np.array(
        [
            r["label"]
            for r in usable_results
        ]
    )

    classifier = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    counts = [
        np.sum(
            y == label
        )
        for label in np.unique(y)
    ]

    minimum_count = min(
        counts
    )

    folds = min(
        5,
        minimum_count
    )

    if folds >= 2:

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

        scores = cross_val_score(
            classifier,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        print()
        print(
            "Fold accuracies:",
            np.round(
                scores,
                3
            )
        )

        print(
            f"Mean accuracy: "
            f"{np.mean(scores) * 100:.2f}%"
        )

        print(
            f"Std deviation: "
            f"{np.std(scores) * 100:.2f}%"
        )

    else:

        scores = np.array([])

        print(
            "Insufficient samples for CV."
        )

else:

    X = np.array([])
    y = np.array([])
    scores = np.array([])

    print(
        "Not enough usable recordings."
    )


# ======================================================================
# APPROACH VS AWAY CLASSIFICATION
# ======================================================================

print()
print("=" * 78)
print("APPROACH vs AWAY MOTION CLASSIFICATION")
print("=" * 78)

pair_results = [
    r
    for r in all_results
    if r["label"] in [
        "approach",
        "away"
    ]
]

pair_scores = np.array([])

if len(pair_results) >= 4:

    X_pair = np.array(
        [
            [
                r[
                    "features"
                ][
                    feature
                ]
                for feature in feature_names
            ]
            for r in pair_results
        ],
        dtype=np.float64
    )

    y_pair = np.array(
        [
            r["label"]
            for r in pair_results
        ]
    )

    approach_count = np.sum(
        y_pair == "approach"
    )

    away_count = np.sum(
        y_pair == "away"
    )

    pair_folds = min(
        5,
        approach_count,
        away_count
    )

    if pair_folds >= 2:

        pair_classifier = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=5,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]
        )

        pair_cv = StratifiedKFold(
            n_splits=pair_folds,
            shuffle=True,
            random_state=42
        )

        pair_scores = cross_val_score(
            pair_classifier,
            X_pair,
            y_pair,
            cv=pair_cv,
            scoring="accuracy"
        )

        print()
        print(
            "Fold accuracies:",
            np.round(
                pair_scores,
                3
            )
        )

        print(
            f"Mean accuracy: "
            f"{np.mean(pair_scores) * 100:.2f}%"
        )

        print(
            f"Std deviation: "
            f"{np.std(pair_scores) * 100:.2f}%"
        )

    else:

        print(
            "Insufficient approach/away data."
        )

else:

    print(
        "Insufficient approach/away recordings."
    )


# ======================================================================
# FEATURE IMPORTANCE
# ======================================================================

print()
print("=" * 78)
print("V8 FEATURE IMPORTANCE")
print("=" * 78)

if (
    len(X) > 0
    and
    len(np.unique(y)) >= 2
):

    classifier.fit(
        X,
        y
    )

    forest = (
        classifier
        .named_steps[
            "model"
        ]
    )

    importance = (
        forest.feature_importances_
    )

    ranking = sorted(
        zip(
            feature_names,
            importance
        ),
        key=lambda x:
        x[1],
        reverse=True
    )

    for name, value in ranking:

        print(
            f"{name:<42} "
            f"{value:.5f}"
        )

else:

    ranking = []


# ======================================================================
# SAVE RECORDING FEATURES
# ======================================================================

recording_file = os.path.join(
    OUTPUT_DIR,
    "v8_recording_motion_features.csv"
)

recording_fields = [
    "label",
    "wav_file"
] + feature_names


with open(
    recording_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=recording_fields
    )

    writer.writeheader()

    for result in all_results:

        row = {
            "label":
                result["label"],

            "wav_file":
                result["filename"]
        }

        for feature in feature_names:

            row[
                feature
            ] = result[
                "features"
            ][
                feature
            ]

        writer.writerow(
            row
        )


# ======================================================================
# SAVE PER-CHIRP TRACKING
# ======================================================================

chirp_file = os.path.join(
    OUTPUT_DIR,
    "v8_chirp_motion_tracking.csv"
)

chirp_fields = [
    "label",
    "wav_file",
    "chirp_index",
    "chirp_time",
    "tracked",
    "delay_ms",
    "strength",
    "frequency_hz",
    "frequency_std_hz",
    "candidate_count"
]


with open(
    chirp_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=chirp_fields
    )

    writer.writeheader()

    for result in all_results:

        for index, chirp in enumerate(
            result["chirps"]
        ):

            tracked = chirp[
                "tracked"
            ]

            writer.writerow(
                {
                    "label":
                        result["label"],

                    "wav_file":
                        result["filename"],

                    "chirp_index":
                        index,

                    "chirp_time":
                        chirp["time"],

                    "tracked":
                        1
                        if tracked is not None
                        else 0,

                    "delay_ms":
                        (
                            tracked[
                                "delay_ms"
                            ]
                            if tracked is not None
                            else ""
                        ),

                    "strength":
                        (
                            tracked[
                                "strength"
                            ]
                            if tracked is not None
                            else ""
                        ),

                    "frequency_hz":
                        chirp[
                            "frequency"
                        ][
                            "frequency_centroid"
                        ],

                    "frequency_std_hz":
                        chirp[
                            "frequency"
                        ][
                            "frequency_std"
                        ],

                    "candidate_count":
                        len(
                            chirp[
                                "candidates"
                            ]
                        )
                }
            )


# ======================================================================
# SAVE ALL CANDIDATE PEAKS
# ======================================================================

peaks_file = os.path.join(
    OUTPUT_DIR,
    "v8_all_candidate_peaks.csv"
)

peak_fields = [
    "label",
    "wav_file",
    "chirp_index",
    "chirp_time",
    "candidate_rank",
    "delay_ms",
    "strength",
    "tracked"
]


with open(
    peaks_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=peak_fields
    )

    writer.writeheader()

    for result in all_results:

        for chirp_index, chirp in enumerate(
            result["chirps"]
        ):

            tracked = chirp[
                "tracked"
            ]

            for rank, candidate in enumerate(
                chirp["candidates"]
            ):

                is_tracked = (
                    tracked is not None
                    and
                    abs(
                        candidate[
                            "delay_ms"
                        ]
                        -
                        tracked[
                            "delay_ms"
                        ]
                    ) < 0.001
                )

                writer.writerow(
                    {
                        "label":
                            result["label"],

                        "wav_file":
                            result["filename"],

                        "chirp_index":
                            chirp_index,

                        "chirp_time":
                            chirp["time"],

                        "candidate_rank":
                            rank + 1,

                        "delay_ms":
                            candidate[
                                "delay_ms"
                            ],

                        "strength":
                            candidate[
                                "strength"
                            ],

                        "tracked":
                            int(
                                is_tracked
                            )
                    }
                )


# ======================================================================
# SAVE CLASS SUMMARY
# ======================================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "v8_class_summary.csv"
)

if class_summaries:

    summary_fields = list(
        class_summaries[0].keys()
    )

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields
        )

        writer.writeheader()

        for summary in class_summaries:

            writer.writerow(
                summary
            )


# ======================================================================
# PLOT ONE RECORDING
# ======================================================================

def plot_recording(
    result
):

    label = result[
        "label"
    ]

    filename = result[
        "filename"
    ]

    chirps = result[
        "chirps"
    ]

    times = np.array(
        [
            c["time"]
            for c in chirps
        ]
    )

    delays = np.array(
        [
            (
                c["tracked"]["delay_ms"]
                if c["tracked"] is not None
                else np.nan
            )
            for c in chirps
        ]
    )

    strengths = np.array(
        [
            (
                c["tracked"]["strength"]
                if c["tracked"] is not None
                else np.nan
            )
            for c in chirps
        ]
    )

    frequencies = np.array(
        [
            c[
                "frequency"
            ][
                "frequency_centroid"
            ]
            for c in chirps
        ]
    )

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
    # Delay trajectory
    # --------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        times,
        delays,
        marker="o"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Tracked reflection delay (ms)"
    )

    plt.title(
        f"V8 Reflection Delay Trajectory\n"
        f"{label.upper()} - {filename}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            safe_name +
            "_delay_trajectory.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------------
    # Delay velocity
    # --------------------------------------------------------------

    velocity = np.diff(
        delays
    ) / (
        np.diff(times) +
        1e-12
    )

    velocity[
        ~np.isfinite(velocity)
    ] = np.nan

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        times[1:],
        velocity,
        marker="o"
    )

    plt.axhline(
        0.0,
        linestyle="--"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Delay velocity (ms/s)"
    )

    plt.title(
        f"V8 Reflection Delay Velocity\n"
        f"{label.upper()} - {filename}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            safe_name +
            "_delay_velocity.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------------
    # Reflection strength
    # --------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        times,
        strengths,
        marker="o"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Tracked reflection strength"
    )

    plt.title(
        f"V8 Reflection Strength\n"
        f"{label.upper()} - {filename}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            safe_name +
            "_strength.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------------
    # Frequency
    # --------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        times,
        frequencies,
        marker="o"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Frequency centroid (Hz)"
    )

    plt.title(
        f"V8 Frequency Trajectory\n"
        f"{label.upper()} - {filename}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            safe_name +
            "_frequency.png"
        ),
        dpi=150
    )

    plt.close()


# ======================================================================
# REPRESENTATIVE PLOTS
# ======================================================================

print()
print("=" * 78)
print("GENERATING V8 REPRESENTATIVE PLOTS")
print("=" * 78)

for label in [
    "idle",
    "approach",
    "away"
]:

    candidates = [
        r
        for r in all_results
        if r["label"] == label
    ]

    if candidates:

        # Prefer a recording with good tracking and
        # non-trivial motion, rather than always selecting #1.
        candidates.sort(
            key=lambda r:
            (
                r[
                    "features"
                ][
                    "track_fraction"
                ],
                r[
                    "features"
                ][
                    "motion_score"
                ]
            ),
            reverse=True
        )

        selected = candidates[0]

        print(
            f"Plotting {label}: "
            f"{selected['filename']}"
        )

        plot_recording(
            selected
        )


# ======================================================================
# CLASS-LEVEL PLOTS
# ======================================================================

print()
print(
    "Generating class-level motion plots..."
)


def class_values(
    items,
    feature
):

    return np.array(
        [
            r[
                "features"
            ][
                feature
            ]
            for r in items
        ],
        dtype=np.float64
    )


# --------------------------------------------------------------
# Delay velocity distribution
# --------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)

plot_labels = []
plot_values = []

for label, items in [
    ("approach", approach_items),
    ("away", away_items),
    ("idle", idle_items)
]:

    if items:

        plot_labels.append(
            label
        )

        plot_values.append(
            class_values(
                items,
                "delay_velocity_mean_ms_s"
            )
        )

plt.boxplot(
    plot_values,
    tick_labels=plot_labels
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.ylabel(
    "Mean delay velocity (ms/s)"
)

plt.title(
    "V8 Class-Level Reflection Motion"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "class_delay_velocity.png"
    ),
    dpi=150
)

plt.close()


# --------------------------------------------------------------
# Net delay change
# --------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)

plot_labels = []
plot_values = []

for label, items in [
    ("approach", approach_items),
    ("away", away_items),
    ("idle", idle_items)
]:

    if items:

        plot_labels.append(
            label
        )

        plot_values.append(
            class_values(
                items,
                "delay_net_change_ms"
            )
        )

plt.boxplot(
    plot_values,
    tick_labels=plot_labels
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.ylabel(
    "Net tracked delay change (ms)"
)

plt.title(
    "V8 Net Reflection Delay Movement"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "class_delay_net_change.png"
    ),
    dpi=150
)

plt.close()


# ======================================================================
# FINAL DIAGNOSIS
# ======================================================================

print()
print("=" * 78)
print("SARV V8 FINAL DIAGNOSIS")
print("=" * 78)

pair_accuracy = (
    np.mean(pair_scores)
    if len(pair_scores)
    else 0.0
)

three_class_accuracy = (
    np.mean(scores)
    if len(scores)
    else 0.0
)

print()

print(
    f"Reflection tracking available : "
    f"{len(all_results)} recordings"
)

print(
    f"Mean approach delay velocity   : "
    f"{approach_velocity:+.5f} ms/s"
)

print(
    f"Mean away delay velocity       : "
    f"{away_velocity:+.5f} ms/s"
)

print(
    f"Mean idle delay velocity       : "
    f"{idle_velocity:+.5f} ms/s"
)

print()

print(
    f"Approach/Away ML accuracy      : "
    f"{pair_accuracy * 100:.2f}%"
)

print(
    f"3-class ML accuracy            : "
    f"{three_class_accuracy * 100:.2f}%"
)


# ======================================================================
# PHYSICAL INTERPRETATION
# ======================================================================

print()
print("-" * 78)
print("PHYSICAL INTERPRETATION")
print("-" * 78)

if directional_sign_pattern:

    print()
    print(
        "✓ The tracked reflection shows an "
        "expected directional sign pattern."
    )

    print(
        "Approach tends toward decreasing delay."
    )

    print(
        "Away tends toward increasing delay."
    )

    if pair_accuracy >= 0.75:

        print()
        print(
            "✓ The physical signal and classification "
            "are both promising."
        )

    else:

        print()
        print(
            "⚠ Physical evidence is promising, but "
            "classification is not yet reliable."
        )

elif opposite_sign_pattern:

    print()
    print(
        "⚠ A direction-like pattern exists, but "
        "its sign is opposite to the expected interpretation."
    )

    print()
    print(
        "This should be investigated before building "
        "a gesture engine."
    )

else:

    print()
    print(
        "✗ V8 does not find the expected physical "
        "approach/away delay-velocity signature."
    )

    print()
    print(
        "The reflection is trackable, but trackability "
        "is not equivalent to motion-direction information."
    )


# ======================================================================
# IMPORTANT CLASSIFICATION WARNING
# ======================================================================

print()
print("-" * 78)
print("CLASSIFICATION WARNING")
print("-" * 78)

if pair_accuracy >= 0.80:

    print()
    print(
        "⚠ High ML accuracy was observed."
    )

    print()
    print(
        "Do NOT automatically interpret this as proof "
        "of acoustic direction detection."
    )

    print(
        "The dataset may contain recording-order, "
        "amplitude, timing, or environmental artifacts."
    )

else:

    print()
    print(
        "No strong approach/away classifier result."
    )


# ======================================================================
# V8 DECISION
# ======================================================================

print()
print("=" * 78)
print("V8 DECISION")
print("=" * 78)

if (
    directional_sign_pattern
    and
    pair_accuracy >= 0.75
):

    print()
    print(
        "PROMISING."
    )

    print()
    print(
        "Existing recordings show both a physically "
        "consistent delay-motion pattern and useful "
        "approach/away separability."
    )

    print()
    print(
        "Recommended next step:"
    )

    print(
        "V9 controlled validation + real-time prototype."
    )

elif (
    pair_accuracy >= 0.75
):

    print()
    print(
        "INTERESTING BUT NOT PHYSICALLY VALIDATED."
    )

    print()
    print(
        "The machine-learning features contain "
        "class information, but the expected physical "
        "direction signature is not established."
    )

    print()
    print(
        "Do NOT build the final gesture engine yet."
    )

elif (
    approach_motion > idle_motion
    and
    away_motion > idle_motion
):

    print()
    print(
        "MOTION IS DETECTABLE, BUT DIRECTION IS UNCLEAR."
    )

    print()
    print(
        "The acoustic reflection appears to respond "
        "to movement, but approach and away remain "
        "physically ambiguous."
    )

    print()
    print(
        "This is still useful for SARV."
    )

else:

    print()
    print(
        "NO RELIABLE MOTION-DIRECTION RESULT."
    )

    print()
    print(
        "The current recordings do not establish "
        "a usable acoustic approach/away signature."
    )


# ======================================================================
# OUTPUT FILES
# ======================================================================

print()
print("=" * 78)
print("V8 OUTPUT FILES")
print("=" * 78)

print()
print(
    "Recording motion features:"
)

print(
    recording_file
)

print()
print(
    "Per-chirp motion tracking:"
)

print(
    chirp_file
)

print()
print(
    "All candidate reflection peaks:"
)

print(
    peaks_file
)

print()
print(
    "Class summary:"
)

print(
    summary_file
)

print()
print(
    "Plots:"
)

print(
    PLOTS_DIR
)

print()
print("=" * 78)
print("SARV V8 ANALYSIS COMPLETE")
print("=" * 78)