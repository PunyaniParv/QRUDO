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
    find_peaks,
    spectrogram
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ================================================================
# SARV ACOUSTIC GESTURE ANALYZER V5
#
# CONTROLLED APPROACH / AWAY / IDLE ANALYSIS
#
# Main objectives:
#
# 1. Detect transmitted chirps
# 2. Detect reflected chirp energy
# 3. Estimate echo delay
# 4. Estimate echo strength
# 5. Track delay over time
# 6. Track echo strength over time
# 7. Estimate frequency/Doppler-related movement
# 8. Determine whether APPROACH and AWAY have
#    physically separable temporal signatures
#
# IMPORTANT:
# This is an experimental acoustic analysis.
# It does NOT assume that the signal is already usable.
# ================================================================


# ================================================================
# CONFIGURATION
# ================================================================

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
    "analysis"

)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# AUDIO CONFIGURATION
# ================================================================

SAMPLE_RATE = 44100

CHIRP_LOW = 7500
CHIRP_HIGH = 8500

CHIRP_DURATION = 0.100

# V5 recorder starts chirps approximately here
FIRST_CHIRP_TIME = 0.15

# Chirps were scheduled every 250 ms
CHIRP_INTERVAL = 0.250

EXPECTED_CHIRPS = 7

# Search for reflected chirp after the direct signal.
#
# 20 cm round trip:
#
# 0.20 * 2 / 343 = ~1.17 ms
#
# 60 cm round trip:
#
# 0.60 * 2 / 343 = ~3.50 ms
#
# We therefore search a wider window.
ECHO_MIN_DELAY_MS = 0.60
ECHO_MAX_DELAY_MS = 12.0

# Speed of sound approximation
SPEED_OF_SOUND = 343.0

# Bandpass used for acoustic analysis
FILTER_LOW = 6000
FILTER_HIGH = 11000


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("SARV ACOUSTIC GESTURE ANALYZER V5")
print("ECHO DELAY + REFLECTION + TEMPORAL MOTION")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Audio:")
print(AUDIO_DIR)

print()
print("Analysis output:")
print(OUTPUT_DIR)


# ================================================================
# CHECK DATASET
# ================================================================

if not os.path.exists(
    METADATA_FILE
):
    raise FileNotFoundError(
        f"Metadata not found:\n{METADATA_FILE}"
    )

if not os.path.exists(
    AUDIO_DIR
):
    raise FileNotFoundError(
        f"Audio directory not found:\n{AUDIO_DIR}"
    )


# ================================================================
# LOAD METADATA
# ================================================================

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


# ================================================================
# FIND AUDIO FILE
# ================================================================

def find_audio(row):

    filename = row.get(
        "wav_file",
        ""
    )

    if not filename:
        return None

    candidate = os.path.join(
        AUDIO_DIR,
        filename
    )

    if os.path.exists(candidate):
        return candidate

    return None


# ================================================================
# GENERATE REFERENCE CHIRP
# ================================================================

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

    phase = 2 * np.pi * (
        CHIRP_LOW * t +
        0.5 * k * t * t
    )

    signal = np.sin(phase)

    window = np.hanning(n)

    signal *= window

    signal = signal.astype(
        np.float64
    )

    # Normalize reference
    norm = np.linalg.norm(signal)

    if norm > 0:
        signal /= norm

    return signal


REFERENCE_CHIRP = (
    generate_reference_chirp()
)


# ================================================================
# BANDPASS FILTER
# ================================================================

def bandpass(
    signal,
    low=FILTER_LOW,
    high=FILTER_HIGH
):

    nyquist = SAMPLE_RATE / 2

    low_norm = low / nyquist
    high_norm = high / nyquist

    sos = butter(
        6,
        [
            low_norm,
            high_norm
        ],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )


# ================================================================
# LOAD AUDIO
# ================================================================

def load_audio(path):

    sr, data = wavfile.read(
        path
    )

    data = data.astype(
        np.float64
    )

    # Handle stereo just in case
    if data.ndim > 1:
        data = np.mean(
            data,
            axis=1
        )

    # Convert integer WAV to approximately [-1, 1]
    if np.issubdtype(
        data.dtype,
        np.integer
    ):
        info = np.iinfo(
            data.dtype
        )

        data /= max(
            abs(info.min),
            info.max
        )

    return data, sr


# ================================================================
# NORMALIZED MATCHED FILTER
# ================================================================

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

    # Sliding normalization
    ref_len = len(reference)

    signal_squared = (
        signal ** 2
    )

    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(
                signal_squared
            )
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
            window_energy
        ) *
        ref_norm
        +
        1e-12
    )

    normalized = (
        correlation /
        denominator
    )

    return normalized


# ================================================================
# EXPECTED CHIRP TIMES
# ================================================================

def expected_chirp_times():

    return np.array(
        [
            FIRST_CHIRP_TIME +
            i * CHIRP_INTERVAL
            for i in range(
                EXPECTED_CHIRPS
            )
        ]
    )


# ================================================================
# EXTRACT CHIRP RESPONSE
# ================================================================

def analyze_chirp_response(
    filtered,
    chirp_time
):

    sample_center = int(
        chirp_time *
        SAMPLE_RATE
    )

    # ------------------------------------------------------------
    # The matched-filter output is shorter than the original
    # signal by len(reference)-1 samples.
    # ------------------------------------------------------------

    mf = matched_filter(
        filtered,
        REFERENCE_CHIRP
    )

    # ------------------------------------------------------------
    # Expected direct chirp location
    # ------------------------------------------------------------

    reference_length = len(
        REFERENCE_CHIRP
    )

    expected_index = (
        sample_center
        -
        reference_length
        // 2
    )

    # ------------------------------------------------------------
    # Search region corresponding to possible hand reflections.
    # ------------------------------------------------------------

    min_offset = int(
        ECHO_MIN_DELAY_MS
        / 1000
        *
        SAMPLE_RATE
    )

    max_offset = int(
        ECHO_MAX_DELAY_MS
        / 1000
        *
        SAMPLE_RATE
    )

    start = (
        expected_index +
        min_offset
    )

    end = (
        expected_index +
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

        return {
            "echo_delay_ms": np.nan,
            "echo_strength": 0.0,
            "direct_strength": 0.0,
            "snr_proxy": 0.0,
            "peak_index": -1
        }

    region = mf[
        start:end
    ]

    abs_region = np.abs(
        region
    )

    if len(abs_region) == 0:

        return {
            "echo_delay_ms": np.nan,
            "echo_strength": 0.0,
            "direct_strength": 0.0,
            "snr_proxy": 0.0,
            "peak_index": -1
        }

    # ------------------------------------------------------------
    # Echo peak
    # ------------------------------------------------------------

    local_peak = int(
        np.argmax(
            abs_region
        )
    )

    peak_index = (
        start +
        local_peak
    )

    echo_strength = float(
        abs_region[
            local_peak
        ]
    )

    delay_samples = (
        peak_index -
        expected_index
    )

    echo_delay_ms = (
        delay_samples /
        SAMPLE_RATE *
        1000
    )

    # ------------------------------------------------------------
    # Direct signal strength
    # ------------------------------------------------------------

    direct_start = max(
        0,
        expected_index
    )

    direct_end = min(
        len(mf),
        expected_index +
        int(
            0.6 /
            1000 *
            SAMPLE_RATE
        )
    )

    if direct_end > direct_start:

        direct_strength = float(
            np.max(
                np.abs(
                    mf[
                        direct_start:
                        direct_end
                    ]
                )
            )
        )

    else:

        direct_strength = 0.0

    # ------------------------------------------------------------
    # Noise estimate
    #
    # Use region around the expected chirp but outside the
    # reflection window.
    # ------------------------------------------------------------

    noise_start = max(
        0,
        expected_index -
        int(
            20 /
            1000 *
            SAMPLE_RATE
        )
    )

    noise_end = max(
        0,
        expected_index -
        int(
            5 /
            1000 *
            SAMPLE_RATE
        )
    )

    if noise_end > noise_start:

        noise = np.abs(
            mf[
                noise_start:
                noise_end
            ]
        )

        noise_level = float(
            np.median(noise)
        )

    else:

        noise_level = 1e-6

    snr_proxy = (
        echo_strength /
        (
            noise_level +
            1e-9
        )
    )

    return {
        "echo_delay_ms":
            float(echo_delay_ms),

        "echo_strength":
            echo_strength,

        "direct_strength":
            direct_strength,

        "snr_proxy":
            float(snr_proxy),

        "peak_index":
            peak_index
    }


# ================================================================
# FREQUENCY / DOPPLER PROXY
# ================================================================

def estimate_frequency_features(
    signal,
    chirp_time
):

    # ------------------------------------------------------------
    # Analyze a small window around the chirp.
    #
    # This is a Doppler-related proxy, not a laboratory-grade
    # Doppler estimator.
    # ------------------------------------------------------------

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

    frequencies, times, Sxx = (
        spectrogram(
            segment,
            fs=SAMPLE_RATE,
            nperseg=1024,
            noverlap=768,
            scaling="spectrum",
            mode="magnitude"
        )
    )

    mask = (
        (frequencies >= 6500)
        &
        (frequencies <= 10000)
    )

    if not np.any(mask):

        return {
            "frequency_centroid": 0.0,
            "frequency_peak": 0.0,
            "frequency_std": 0.0
        }

    band_freqs = frequencies[
        mask
    ]

    band_power = Sxx[
        mask
    ]

    power_mean = np.mean(
        band_power,
        axis=1
    )

    total = (
        np.sum(power_mean)
        +
        1e-12
    )

    centroid = (
        np.sum(
            band_freqs *
            power_mean
        )
        /
        total
    )

    peak_frequency = float(
        band_freqs[
            np.argmax(
                power_mean
            )
        ]
    )

    # Frequency distribution spread
    variance = (
        np.sum(
            (
                band_freqs -
                centroid
            ) ** 2
            *
            power_mean
        )
        /
        total
    )

    frequency_std = np.sqrt(
        max(
            variance,
            0
        )
    )

    return {
        "frequency_centroid":
            float(centroid),

        "frequency_peak":
            peak_frequency,

        "frequency_std":
            float(frequency_std)
    }


# ================================================================
# ANALYZE ONE RECORDING
# ================================================================

def analyze_recording(
    audio_path
):

    audio, sr = load_audio(
        audio_path
    )

    if sr != SAMPLE_RATE:

        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz "
            f"but got {sr} Hz"
        )

    if len(audio) < SAMPLE_RATE:

        raise ValueError(
            "Recording is too short."
        )

    filtered = bandpass(
        audio
    )

    chirp_times = (
        expected_chirp_times()
    )

    results = []

    for chirp_time in chirp_times:

        response = (
            analyze_chirp_response(
                filtered,
                chirp_time
            )
        )

        freq = (
            estimate_frequency_features(
                filtered,
                chirp_time
            )
        )

        result = {
            "time": chirp_time,
            **response,
            **freq
        }

        results.append(
            result
        )

    return results


# ================================================================
# TEMPORAL FEATURES
# ================================================================

def safe_slope(
    x,
    y
):

    x = np.asarray(x)
    y = np.asarray(y)

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


def safe_mean(x):

    x = np.asarray(x)

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return 0.0

    return float(
        np.mean(x)
    )


def safe_std(x):

    x = np.asarray(x)

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return 0.0

    return float(
        np.std(x)
    )


def calculate_temporal_features(
    response
):

    times = np.array(
        [
            r["time"]
            for r in response
        ]
    )

    delays = np.array(
        [
            r["echo_delay_ms"]
            for r in response
        ]
    )

    strengths = np.array(
        [
            r["echo_strength"]
            for r in response
        ]
    )

    frequencies = np.array(
        [
            r["frequency_centroid"]
            for r in response
        ]
    )

    snr = np.array(
        [
            r["snr_proxy"]
            for r in response
        ]
    )

    # ------------------------------------------------------------
    # Temporal slopes
    # ------------------------------------------------------------

    delay_slope = safe_slope(
        times,
        delays
    )

    strength_slope = safe_slope(
        times,
        strengths
    )

    frequency_slope = safe_slope(
        times,
        frequencies
    )

    # ------------------------------------------------------------
    # First-half / second-half changes
    # ------------------------------------------------------------

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

    return {
        "delay_mean":
            safe_mean(delays),

        "delay_std":
            safe_std(delays),

        "delay_slope":
            delay_slope,

        "delay_first":
            delay_first,

        "delay_second":
            delay_second,

        "delay_change":
            delay_second -
            delay_first,

        "strength_mean":
            safe_mean(strengths),

        "strength_std":
            safe_std(strengths),

        "strength_slope":
            strength_slope,

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
            safe_std(frequencies),

        "frequency_slope":
            frequency_slope,

        "frequency_first":
            frequency_first,

        "frequency_second":
            frequency_second,

        "frequency_change":
            frequency_second -
            frequency_first,

        "snr_mean":
            safe_mean(snr),

        "snr_std":
            safe_std(snr)
    }


# ================================================================
# EXTRACT DATASET
# ================================================================

print()
print("=" * 70)
print("ANALYZING RECORDINGS")
print("=" * 70)

X = []
labels = []
names = []

all_temporal = []

missing = 0
failed = 0

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

        print(
            f"WARNING: audio not found "
            f"for record {index + 1}"
        )

        missing += 1
        continue

    try:

        response = analyze_recording(
            audio_path
        )

        features = (
            calculate_temporal_features(
                response
            )
        )

        X.append(
            [
                features[
                    "delay_mean"
                ],

                features[
                    "delay_std"
                ],

                features[
                    "delay_slope"
                ],

                features[
                    "delay_change"
                ],

                features[
                    "strength_mean"
                ],

                features[
                    "strength_std"
                ],

                features[
                    "strength_slope"
                ],

                features[
                    "strength_change"
                ],

                features[
                    "frequency_mean"
                ],

                features[
                    "frequency_std"
                ],

                features[
                    "frequency_slope"
                ],

                features[
                    "frequency_change"
                ],

                features[
                    "snr_mean"
                ],

                features[
                    "snr_std"
                ]
            ]
        )

        labels.append(
            label
        )

        names.append(
            os.path.basename(
                audio_path
            )
        )

        all_temporal.append(
            (
                label,
                os.path.basename(
                    audio_path
                ),
                response,
                features
            )
        )

    except Exception as e:

        print()
        print(
            f"ERROR processing:"
        )

        print(
            audio_path
        )

        print(
            e
        )

        failed += 1


X = np.asarray(
    X,
    dtype=np.float64
)

labels = np.asarray(
    labels
)

print()
print(
    f"Usable recordings: {len(X)}"
)

print(
    f"Missing recordings: {missing}"
)

print(
    f"Failed recordings: {failed}"
)

if len(X) == 0:

    raise RuntimeError(
        "No usable recordings."
    )


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

unique_labels, counts = np.unique(
    labels,
    return_counts=True
)

for label, count in zip(
    unique_labels,
    counts
):

    print(
        f"{label:<15}: {count}"
    )


# ================================================================
# FEATURE NAMES
# ================================================================

feature_names = [

    "delay_mean",
    "delay_std",
    "delay_slope",
    "delay_change",

    "strength_mean",
    "strength_std",
    "strength_slope",
    "strength_change",

    "frequency_mean",
    "frequency_std",
    "frequency_slope",
    "frequency_change",

    "snr_mean",
    "snr_std"
]


# ================================================================
# PHYSICAL SIGNAL ANALYSIS
# ================================================================

print()
print("=" * 70)
print("PHYSICAL SIGNAL ANALYSIS")
print("=" * 70)

print()
print(
    "Echo delay is the most important measurement."
)

print(
    "For a hand moving TOWARD the laptop,"
)

print(
    "the expected reflection delay should generally DECREASE."
)

print()
print(
    "For a hand moving AWAY,"
)

print(
    "the expected reflection delay should generally INCREASE."
)


# ================================================================
# CLASS STATISTICS
# ================================================================

print()
print("=" * 70)
print("TEMPORAL FEATURE STATISTICS")
print("=" * 70)

for label in unique_labels:

    mask = (
        labels == label
    )

    class_X = X[
        mask
    ]

    print()
    print(
        f"[{label.upper()}]"
    )

    for feature_index, name in enumerate(
        feature_names
    ):

        values = class_X[
            :,
            feature_index
        ]

        print(
            f"{name:<22} "
            f"mean={np.mean(values): .6f} "
            f"std={np.std(values): .6f}"
        )


# ================================================================
# EXPECTED DIRECTION TEST
# ================================================================

print()
print("=" * 70)
print("APPROACH / AWAY PHYSICAL DIRECTION TEST")
print("=" * 70)

for label in [
    "idle",
    "approach",
    "away"
]:

    mask = (
        labels == label
    )

    if not np.any(mask):
        continue

    delay_slopes = X[
        mask,
        feature_names.index(
            "delay_slope"
        )
    ]

    delay_changes = X[
        mask,
        feature_names.index(
            "delay_change"
        )
    ]

    print()
    print(
        f"{label.upper():<10}"
    )

    print(
        f"Delay slope  : "
        f"{np.mean(delay_slopes): .6f} "
        f"+/- "
        f"{np.std(delay_slopes): .6f}"
    )

    print(
        f"Delay change : "
        f"{np.mean(delay_changes): .6f} "
        f"+/- "
        f"{np.std(delay_changes): .6f}"
    )


# ================================================================
# SIMPLE PHYSICAL INTERPRETATION
# ================================================================

approach_mask = (
    labels == "approach"
)

away_mask = (
    labels == "away"
)

if (
    np.any(approach_mask)
    and
    np.any(away_mask)
):

    approach_delay_slope = np.mean(
        X[
            approach_mask,
            feature_names.index(
                "delay_slope"
            )
        ]
    )

    away_delay_slope = np.mean(
        X[
            away_mask,
            feature_names.index(
                "delay_slope"
            )
        ]
    )

    print()
    print("=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)

    print()

    if (
        approach_delay_slope < 0
        and
        away_delay_slope > 0
    ):

        print(
            "✓ EXPECTED DIRECTIONAL PATTERN DETECTED"
        )

        print()
        print(
            "Approach tends to produce decreasing "
            "echo delay."
        )

        print(
            "Away tends to produce increasing "
            "echo delay."
        )

        print()
        print(
            "This is an important result."
        )

    elif (
        approach_delay_slope > 0
        and
        away_delay_slope < 0
    ):

        print(
            "⚠ OPPOSITE DIRECTIONAL PATTERN"
        )

        print()
        print(
            "The measured trend is opposite to "
            "the expected physical direction."
        )

        print(
            "This requires inspection of the raw recordings."
        )

    else:

        print(
            "✗ NO CLEAR DIRECTIONAL DELAY PATTERN"
        )

        print()
        print(
            "Approach and away do not currently show "
            "the expected opposite delay trends."
        )

        print(
            "Do NOT build the final classifier yet."
        )


# ================================================================
# CLASSIFICATION
# ================================================================

print()
print("=" * 70)
print("3-CLASS TEMPORAL CLASSIFICATION")
print("=" * 70)

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
                max_depth=6,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


class_counts = {
    label: np.sum(
        labels == label
    )
    for label in unique_labels
}


minimum_class_count = min(
    class_counts.values()
)

folds = min(
    5,
    minimum_class_count
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
        labels,
        cv=cv,
        scoring="accuracy"
    )

    print()

    print(
        f"Fold accuracies: "
        f"{np.round(scores, 3)}"
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

    print(
        "Not enough recordings for cross-validation."
    )

    scores = np.array([])


# ================================================================
# PAIRWISE APPROACH / AWAY TEST
# ================================================================

print()
print("=" * 70)
print("APPROACH vs AWAY")
print("=" * 70)

pair_mask = (
    (labels == "approach")
    |
    (labels == "away")
)

X_pair = X[
    pair_mask
]

y_pair = labels[
    pair_mask
]

if (
    len(np.unique(y_pair)) == 2
    and
    min(
        np.sum(
            y_pair == "approach"
        ),
        np.sum(
            y_pair == "away"
        )
    ) >= 2
):

    pair_folds = min(
        5,
        np.min(
            [
                np.sum(
                    y_pair == "approach"
                ),
                np.sum(
                    y_pair == "away"
                )
            ]
        )
    )

    pair_cv = StratifiedKFold(
        n_splits=pair_folds,
        shuffle=True,
        random_state=42
    )

    pair_scores = cross_val_score(
        classifier,
        X_pair,
        y_pair,
        cv=pair_cv,
        scoring="accuracy"
    )

    print()

    print(
        f"Fold accuracies: "
        f"{np.round(pair_scores, 3)}"
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

    pair_scores = np.array([])

    print(
        "Insufficient data for pairwise test."
    )


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

classifier.fit(
    X,
    labels
)

forest = (
    classifier
    .named_steps["model"]
)

importance = (
    forest.feature_importances_
)

ranking = sorted(
    zip(
        feature_names,
        importance
    ),
    key=lambda x: x[1],
    reverse=True
)

for name, value in ranking:

    print(
        f"{name:<25} "
        f"{value:.5f}"
    )


# ================================================================
# SAVE PER-RECORDING RESULTS
# ================================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "v5_recording_features.csv"
)

result_fields = [
    "label",
    "wav_file"
] + feature_names


with open(
    results_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=result_fields
    )

    writer.writeheader()

    for label, name, response, features in (
        all_temporal
    ):

        row = {
            "label": label,
            "wav_file": name
        }

        for feature in feature_names:

            row[
                feature
            ] = features[
                feature
            ]

        writer.writerow(
            row
        )


# ================================================================
# SAVE CHIRP RESPONSE DATA
# ================================================================

response_file = os.path.join(
    OUTPUT_DIR,
    "v5_chirp_responses.csv"
)

response_fields = [
    "label",
    "wav_file",
    "chirp_time",
    "echo_delay_ms",
    "echo_strength",
    "direct_strength",
    "snr_proxy",
    "frequency_centroid",
    "frequency_peak",
    "frequency_std"
]


with open(
    response_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=response_fields
    )

    writer.writeheader()

    for label, name, response, features in (
        all_temporal
    ):

        for r in response:

            writer.writerow(
                {
                    "label": label,
                    "wav_file": name,
                    "chirp_time":
                        r["time"],
                    "echo_delay_ms":
                        r["echo_delay_ms"],
                    "echo_strength":
                        r["echo_strength"],
                    "direct_strength":
                        r["direct_strength"],
                    "snr_proxy":
                        r["snr_proxy"],
                    "frequency_centroid":
                        r[
                            "frequency_centroid"
                        ],
                    "frequency_peak":
                        r[
                            "frequency_peak"
                        ],
                    "frequency_std":
                        r[
                            "frequency_std"
                        ]
                }
            )


# ================================================================
# PLOT REPRESENTATIVE RECORDINGS
# ================================================================

print()
print("=" * 70)
print("GENERATING REPRESENTATIVE PLOTS")
print("=" * 70)


def plot_recording(
    label,
    filename,
    response
):

    times = np.array(
        [
            r["time"]
            for r in response
        ]
    )

    delays = np.array(
        [
            r["echo_delay_ms"]
            for r in response
        ]
    )

    strengths = np.array(
        [
            r["echo_strength"]
            for r in response
        ]
    )

    frequencies = np.array(
        [
            r["frequency_centroid"]
            for r in response
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

    # ------------------------------------------------------------
    # Delay plot
    # ------------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
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
        "Estimated echo delay (ms)"
    )

    plt.title(
        f"{label.upper()} - Echo Delay\n"
        f"{filename}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    delay_path = os.path.join(
        OUTPUT_DIR,
        safe_name +
        "_delay.png"
    )

    plt.savefig(
        delay_path,
        dpi=150
    )

    plt.close()

    # ------------------------------------------------------------
    # Strength plot
    # ------------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
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
        "Matched-filter echo strength"
    )

    plt.title(
        f"{label.upper()} - Echo Strength\n"
        f"{filename}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    strength_path = os.path.join(
        OUTPUT_DIR,
        safe_name +
        "_strength.png"
    )

    plt.savefig(
        strength_path,
        dpi=150
    )

    plt.close()

    # ------------------------------------------------------------
    # Frequency plot
    # ------------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
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
        f"{label.upper()} - Frequency\n"
        f"{filename}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    frequency_path = os.path.join(
        OUTPUT_DIR,
        safe_name +
        "_frequency.png"
    )

    plt.savefig(
        frequency_path,
        dpi=150
    )

    plt.close()


# Select one representative recording per class

for label in [
    "idle",
    "approach",
    "away"
]:

    candidates = [
        item
        for item in all_temporal
        if item[0] == label
    ]

    if candidates:

        label_value, filename, response, _ = (
            candidates[0]
        )

        plot_recording(
            label_value,
            filename,
            response
        )


# ================================================================
# FINAL INTERPRETATION
# ================================================================

print()
print("=" * 70)
print("SARV V5 INTERPRETATION")
print("=" * 70)

print()

if (
    np.any(approach_mask)
    and
    np.any(away_mask)
):

    approach_slope = np.mean(
        X[
            approach_mask,
            feature_names.index(
                "delay_slope"
            )
        ]
    )

    away_slope = np.mean(
        X[
            away_mask,
            feature_names.index(
                "delay_slope"
            )
        ]
    )

    slope_difference = (
        abs(
            approach_slope -
            away_slope
        )
    )

    print(
        f"Approach delay slope : "
        f"{approach_slope:.6f} ms/s"
    )

    print(
        f"Away delay slope     : "
        f"{away_slope:.6f} ms/s"
    )

    print(
        f"Absolute difference  : "
        f"{slope_difference:.6f} ms/s"
    )

    print()

    # ------------------------------------------------------------
    # Strong physical result
    # ------------------------------------------------------------

    if (
        approach_slope < 0
        and
        away_slope > 0
    ):

        print(
            "✓ DIRECTIONAL ECHO TREND EXISTS."
        )

        print()
        print(
            "The acoustic measurements show the "
            "expected opposite delay trends."
        )

        if (
            len(pair_scores)
            and
            np.mean(pair_scores) >= 0.80
        ):

            print()
            print(
                "✓ APPROACH/AWAY CLASSIFICATION "
                "IS ALSO PROMISING."
            )

            print(
                "The next step should be real-time "
                "temporal tracking."
            )

        else:

            print()
            print(
                "However, classification is not "
                "yet reliable enough."
            )

    else:

        print(
            "⚠ NO CLEAN APPROACH/AWAY "
            "DELAY SIGNATURE DETECTED."
        )

        print()
        print(
            "The current acoustic setup does not "
            "yet demonstrate the expected physical "
            "direction signal."
        )


# ================================================================
# FINAL DECISION
# ================================================================

print()
print("=" * 70)
print("V5 DECISION")
print("=" * 70)

if len(pair_scores):

    pair_accuracy = (
        np.mean(pair_scores)
    )

    print()
    print(
        f"Approach/Away accuracy: "
        f"{pair_accuracy * 100:.2f}%"
    )

else:

    pair_accuracy = 0.0


if (
    np.any(approach_mask)
    and
    np.any(away_mask)
):

    approach_slope = np.mean(
        X[
            approach_mask,
            feature_names.index(
                "delay_slope"
            )
        ]
    )

    away_slope = np.mean(
        X[
            away_mask,
            feature_names.index(
                "delay_slope"
            )
        ]
    )

else:

    approach_slope = 0.0
    away_slope = 0.0


if (
    approach_slope < 0
    and
    away_slope > 0
    and
    pair_accuracy >= 0.80
):

    print()
    print(
        "PROMISING ACOUSTIC RESULT."
    )

    print()
    print(
        "The experiment shows both:"
    )

    print(
        "1. A physically consistent echo-delay trend."
    )

    print(
        "2. Useful APPROACH/AWAY classification."
    )

    print()
    print(
        "Proceed to V6 real-time approach/away tracking."
    )

elif (
    pair_accuracy >= 0.70
):

    print()
    print(
        "PROMISING BUT NOT READY."
    )

    print()
    print(
        "The classifier sees information, "
        "but the physical signal needs further validation."
    )

else:

    print()
    print(
        "NO RELIABLE APPROACH/AWAY RESULT YET."
    )

    print()
    print(
        "Do not build the final gesture engine."
    )

    print(
        "Inspect the raw recordings and acoustic setup."
    )


# ================================================================
# OUTPUT FILES
# ================================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print()
print(
    "Per-recording features:"
)

print(
    results_file
)

print()
print(
    "Per-chirp measurements:"
)

print(
    response_file
)

print()
print(
    "Representative plots:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 70)
print("V5 ANALYSIS COMPLETE")
print("=" * 70)