import os
import csv
import warnings
import numpy as np
import librosa

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.signal import correlate, butter, sosfilt

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "gesture_dataset_v3"
)

AUDIO_DIR = os.path.join(
    DATASET_DIR,
    "audio"
)

METADATA_FILE = os.path.join(
    DATASET_DIR,
    "metadata.csv"
)

SAMPLE_RATE = 44100

CHIRP_LOW = 7500
CHIRP_HIGH = 8500

# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("SARV ACOUSTIC GESTURE ANALYZER V4")
print("RAW SIGNAL + TIME/FREQUENCY + CHIRP ANALYSIS")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Audio:")
print(AUDIO_DIR)

# ================================================================
# DATASET VALIDATION
# ================================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset directory not found:\n{DATASET_DIR}"
    )

if not os.path.exists(AUDIO_DIR):
    raise FileNotFoundError(
        f"Audio directory not found:\n{AUDIO_DIR}"
    )

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(
        f"Metadata not found:\n{METADATA_FILE}"
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

    print()
    print("Metadata columns:")
    print(reader.fieldnames)

    for row in reader:
        records.append(row)

print()
print(f"Metadata records: {len(records)}")

# ================================================================
# FIND AUDIO FILE
# ================================================================

def find_audio(row):

    # ------------------------------------------------------------
    # PRIMARY METHOD
    # ------------------------------------------------------------
    # Your metadata contains:
    #
    # wav_file
    #
    # Example:
    # approach_001_20260814_134912_982182.wav
    #
    # This is the correct source of truth.
    # ------------------------------------------------------------

    wav_filename = row.get("wav_file")

    if wav_filename:

        wav_filename = str(wav_filename).strip()

        # If metadata contains an absolute path
        if os.path.isabs(wav_filename):

            candidate = wav_filename

        else:

            candidate = os.path.join(
                AUDIO_DIR,
                wav_filename
            )

        if os.path.exists(candidate):
            return candidate

    # ------------------------------------------------------------
    # SECONDARY FALLBACK
    # ------------------------------------------------------------

    possible_keys = [
        "filename",
        "file",
        "audio",
        "audio_file",
        "path"
    ]

    for key in possible_keys:

        value = row.get(key)

        if not value:
            continue

        candidate = str(value).strip()

        if not os.path.isabs(candidate):

            candidate = os.path.join(
                AUDIO_DIR,
                candidate
            )

        if os.path.exists(candidate):

            return candidate

    # ------------------------------------------------------------
    # LAST RESORT:
    # SEARCH USING TIMESTAMP / IDENTIFIERS
    # ------------------------------------------------------------

    identifiers = []

    for key in [
        "timestamp",
        "id",
        "recording_id",
        "sample",
        "name"
    ]:

        value = row.get(key)

        if value:
            identifiers.append(
                str(value).lower()
            )

    # Only search if necessary
    if identifiers:

        for root, _, files in os.walk(AUDIO_DIR):

            for filename in files:

                lower = filename.lower()

                if not lower.endswith(
                    (".wav", ".flac", ".mp3", ".ogg")
                ):
                    continue

                for identifier in identifiers:

                    if identifier in lower:

                        return os.path.join(
                            root,
                            filename
                        )

    return None


# ================================================================
# BANDPASS FILTER
# ================================================================

def bandpass_signal(
    y,
    sr,
    low=7000,
    high=10000
):

    nyquist = sr / 2

    low_norm = low / nyquist

    high_norm = min(
        high / nyquist,
        0.999
    )

    sos = butter(
        6,
        [low_norm, high_norm],
        btype="bandpass",
        output="sos"
    )

    return sosfilt(
        sos,
        y
    )


# ================================================================
# CHIRP GENERATOR
# ================================================================

def generate_chirp(
    duration,
    sr
):

    t = np.linspace(
        0,
        duration,
        int(sr * duration),
        endpoint=False
    )

    k = (
        CHIRP_HIGH -
        CHIRP_LOW
    ) / duration

    phase = 2 * np.pi * (
        CHIRP_LOW * t +
        0.5 * k * t * t
    )

    chirp = np.sin(phase)

    window = np.hanning(
        len(chirp)
    )

    return chirp * window


REFERENCE_CHIRP = generate_chirp(
    0.100,
    SAMPLE_RATE
)

# ================================================================
# SIGNAL FEATURES
# ================================================================

def extract_features(
    audio_path
):

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) < 1000:
        return None

    y = y.astype(
        np.float32
    )

    # ------------------------------------------------------------
    # BASIC TIME DOMAIN
    # ------------------------------------------------------------

    rms = float(
        np.sqrt(
            np.mean(
                y ** 2
            )
        )
    )

    peak = float(
        np.max(
            np.abs(y)
        )
    )

    crest = (
        peak /
        (rms + 1e-9)
    )

    zero_crossings = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                y
            )[0]
        )
    )

    # ------------------------------------------------------------
    # ENERGY BY TIME
    # ------------------------------------------------------------

    n = len(y)

    q1 = y[
        : n // 4
    ]

    q2 = y[
        n // 4 : n // 2
    ]

    q3 = y[
        n // 2 : 3 * n // 4
    ]

    q4 = y[
        3 * n // 4 :
    ]

    energies = [

        np.sqrt(
            np.mean(
                x ** 2
            ) + 1e-12
        )

        for x in [
            q1,
            q2,
            q3,
            q4
        ]
    ]

    energy_mean = float(
        np.mean(
            energies
        )
    )

    energy_std = float(
        np.std(
            energies
        )
    )

    energy_slope = float(
        np.polyfit(
            np.arange(4),
            energies,
            1
        )[0]
    )

    # ------------------------------------------------------------
    # SPECTRAL FEATURES
    # ------------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        roll_percent=0.85
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y
    )[0]

    centroid_mean = float(
        np.mean(centroid)
    )

    centroid_std = float(
        np.std(centroid)
    )

    bandwidth_mean = float(
        np.mean(bandwidth)
    )

    bandwidth_std = float(
        np.std(bandwidth)
    )

    rolloff_mean = float(
        np.mean(rolloff)
    )

    flatness_mean = float(
        np.mean(flatness)
    )

    # ------------------------------------------------------------
    # STFT
    # ------------------------------------------------------------

    stft = np.abs(
        librosa.stft(
            y,
            n_fft=2048,
            hop_length=256
        )
    )

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=2048
    )

    # ------------------------------------------------------------
    # BAND ENERGY
    # ------------------------------------------------------------

    def band_energy(
        low,
        high
    ):

        mask = (
            (freqs >= low) &
            (freqs <= high)
        )

        if not np.any(mask):

            return 0.0

        return float(
            np.mean(
                stft[mask]
            )
        )

    e_5_7 = band_energy(
        5000,
        7000
    )

    e_7_8 = band_energy(
        7000,
        8000
    )

    e_8_9 = band_energy(
        8000,
        9000
    )

    e_9_11 = band_energy(
        9000,
        11000
    )

    e_11_14 = band_energy(
        11000,
        14000
    )

    # ------------------------------------------------------------
    # TARGET BAND
    # ------------------------------------------------------------

    target_mask = (
        (freqs >= CHIRP_LOW) &
        (freqs <= CHIRP_HIGH)
    )

    if np.any(
        target_mask
    ):

        target_energy = float(
            np.mean(
                stft[target_mask]
            )
        )

        target_std = float(
            np.std(
                stft[target_mask]
            )
        )

        target_max = float(
            np.max(
                stft[target_mask]
            )
        )

    else:

        target_energy = 0.0
        target_std = 0.0
        target_max = 0.0

    # ------------------------------------------------------------
    # CHIRP CORRELATION
    # ------------------------------------------------------------

    filtered = bandpass_signal(
        y,
        sr,
        6500,
        10000
    )

    corr = correlate(
        filtered,
        REFERENCE_CHIRP,
        mode="valid",
        method="fft"
    )

    if len(corr) > 0:

        corr_abs = np.abs(
            corr
        )

        corr_peak = float(
            np.max(
                corr_abs
            )
        )

        corr_mean = float(
            np.mean(
                corr_abs
            )
        )

        corr_std = float(
            np.std(
                corr_abs
            )
        )

    else:

        corr_peak = 0.0
        corr_mean = 0.0
        corr_std = 0.0

    # ------------------------------------------------------------
    # NORMALIZED CHIRP CORRELATION
    # ------------------------------------------------------------

    chirp_norm = np.linalg.norm(
        REFERENCE_CHIRP
    )

    signal_norm = np.linalg.norm(
        filtered
    )

    normalized_corr = (
        corr_peak /
        (
            chirp_norm *
            signal_norm +
            1e-9
        )
    )

    # ------------------------------------------------------------
    # TEMPORAL CHIRP VARIATION
    # ------------------------------------------------------------

    windows = 8

    segment_length = (
        len(filtered) //
        windows
    )

    local_corrs = []

    if (
        segment_length >
        len(REFERENCE_CHIRP)
    ):

        for i in range(
            windows
        ):

            start = (
                i *
                segment_length
            )

            end = (
                start +
                segment_length
            )

            segment = filtered[
                start:end
            ]

            local = correlate(
                segment,
                REFERENCE_CHIRP,
                mode="valid",
                method="fft"
            )

            if len(local):

                local_corrs.append(
                    float(
                        np.max(
                            np.abs(
                                local
                            )
                        )
                    )
                )

    if local_corrs:

        local_corr_mean = float(
            np.mean(
                local_corrs
            )
        )

        local_corr_std = float(
            np.std(
                local_corrs
            )
        )

        local_corr_slope = float(
            np.polyfit(
                np.arange(
                    len(local_corrs)
                ),
                local_corrs,
                1
            )[0]
        )

    else:

        local_corr_mean = 0.0
        local_corr_std = 0.0
        local_corr_slope = 0.0

    # ------------------------------------------------------------
    # SPECTRAL PEAK NEAR CHIRP
    # ------------------------------------------------------------

    spectrum = np.mean(
        stft,
        axis=1
    )

    target_indices = np.where(
        target_mask
    )[0]

    if len(
        target_indices
    ):

        target_spectrum = spectrum[
            target_indices
        ]

        strongest_index = np.argmax(
            target_spectrum
        )

        strongest_frequency = float(
            freqs[
                target_indices[
                    strongest_index
                ]
            ]
        )

    else:

        strongest_frequency = 0.0

    # ------------------------------------------------------------
    # HIGH / LOW ENERGY RATIO
    # ------------------------------------------------------------

    low_energy = float(
        np.mean(
            np.abs(y)
        )
    )

    high_band = band_energy(
        7000,
        10000
    )

    high_low_ratio = (
        high_band /
        (
            low_energy +
            1e-9
        )
    )

    # ------------------------------------------------------------
    # FEATURE VECTOR
    # ------------------------------------------------------------

    features = [

        rms,
        peak,
        crest,
        zero_crossings,

        energy_mean,
        energy_std,
        energy_slope,

        centroid_mean,
        centroid_std,

        bandwidth_mean,
        bandwidth_std,

        rolloff_mean,
        flatness_mean,

        e_5_7,
        e_7_8,
        e_8_9,
        e_9_11,
        e_11_14,

        target_energy,
        target_std,
        target_max,

        corr_peak,
        corr_mean,
        corr_std,

        normalized_corr,

        local_corr_mean,
        local_corr_std,
        local_corr_slope,

        strongest_frequency,

        high_low_ratio
    ]

    return np.array(
        features,
        dtype=np.float64
    )


# ================================================================
# EXTRACT DATASET
# ================================================================

print()
print("=" * 70)
print("EXTRACTING RAW SIGNAL FEATURES")
print("=" * 70)

X = []
y = []
paths = []

class_counts = {}

missing_files = 0
failed_files = 0

for index, row in enumerate(
    records
):

    label = (
        row.get("gesture")
        or row.get("label")
        or row.get("class")
    )

    if not label:
        print(
            f"WARNING: no label for record "
            f"{index + 1}"
        )
        continue

    label = str(
        label
    ).strip()

    audio_path = find_audio(
        row
    )

    if audio_path is None:

        missing_files += 1

        print(
            f"WARNING: audio not found "
            f"for record {index + 1}: "
            f"{row.get('wav_file', 'unknown')}"
        )

        continue

    try:

        features = extract_features(
            audio_path
        )

        if features is None:

            failed_files += 1

            continue

        X.append(
            features
        )

        y.append(
            label
        )

        paths.append(
            audio_path
        )

        class_counts[label] = (
            class_counts.get(
                label,
                0
            ) + 1
        )

    except Exception as e:

        failed_files += 1

        print(
            f"ERROR processing "
            f"{audio_path}: {e}"
        )


X = np.array(
    X,
    dtype=np.float64
)

y = np.array(
    y
)

print()
print(
    f"Usable recordings: "
    f"{len(X)}"
)

print(
    f"Missing audio files: "
    f"{missing_files}"
)

print(
    f"Failed recordings: "
    f"{failed_files}"
)

# ================================================================
# SAFETY CHECK
# ================================================================

if len(X) == 0:

    print()
    print("=" * 70)
    print("FATAL: NO USABLE AUDIO")
    print("=" * 70)

    print()
    print(
        "Check that metadata.csv contains "
        "a wav_file column and that the WAV files "
        "exist inside:"
    )

    print(
        AUDIO_DIR
    )

    raise SystemExit(
        1
    )

if X.ndim != 2:

    raise RuntimeError(
        f"Unexpected feature matrix shape: "
        f"{X.shape}"
    )

print(
    f"Feature count: "
    f"{X.shape[1]}"
)

# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for label in sorted(
    class_counts
):

    print(
        f"{label:<15}: "
        f"{class_counts[label]}"
    )

# ================================================================
# FEATURE STATISTICS
# ================================================================

print()
print("=" * 70)
print("FEATURE VARIATION")
print("=" * 70)

feature_std = np.std(
    X,
    axis=0
)

print(
    f"Average feature std: "
    f"{np.mean(feature_std):.6f}"
)

print(
    f"Maximum feature std: "
    f"{np.max(feature_std):.6f}"
)

# ================================================================
# FULL CLASSIFICATION
# ================================================================

print()
print("=" * 70)
print("FULL 7-CLASS CLASSIFICATION")
print("=" * 70)

classifier = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced"
        )
    )
])

cv = StratifiedKFold(
    n_splits=5,
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

# ================================================================
# PAIRWISE TEST
# ================================================================

def pairwise_test(
    label_a,
    label_b
):

    mask = (
        (y == label_a) |
        (y == label_b)
    )

    X_pair = X[
        mask
    ]

    y_pair = y[
        mask
    ]

    if len(
        np.unique(
            y_pair
        )
    ) != 2:

        return None

    clf = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    pair_scores = cross_val_score(
        clf,
        X_pair,
        y_pair,
        cv=5,
        scoring="accuracy"
    )

    return pair_scores


print()
print("=" * 70)
print("PAIRWISE GESTURE SEPARATION")
print("=" * 70)

pairs = [

    ("idle", "approach"),
    ("idle", "away"),

    ("idle", "swipe_left"),
    ("idle", "swipe_right"),

    ("idle", "hand_up"),
    ("idle", "hand_down"),

    ("swipe_left", "swipe_right"),

    ("hand_up", "hand_down"),

    ("approach", "away"),

    ("swipe_left", "hand_up"),
    ("swipe_left", "hand_down"),

    ("swipe_right", "hand_up"),
    ("swipe_right", "hand_down"),

    ("approach", "swipe_left"),
    ("approach", "swipe_right"),

    ("away", "swipe_left"),
    ("away", "swipe_right")
]

pair_results = []

for a, b in pairs:

    result = pairwise_test(
        a,
        b
    )

    if result is None:
        continue

    mean_score = np.mean(
        result
    )

    std_score = np.std(
        result
    )

    pair_results.append(
        (
            a,
            b,
            mean_score
        )
    )

    print(
        f"{a:<15} vs "
        f"{b:<15} : "
        f"{mean_score * 100:6.2f}% "
        f"+/- "
        f"{std_score * 100:5.2f}%"
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
    y
)

forest = classifier.named_steps[
    "model"
]

feature_names = [

    "rms",
    "peak",
    "crest",
    "zero_crossings",

    "energy_mean",
    "energy_std",
    "energy_slope",

    "centroid_mean",
    "centroid_std",

    "bandwidth_mean",
    "bandwidth_std",

    "rolloff_mean",
    "flatness_mean",

    "energy_5_7k",
    "energy_7_8k",
    "energy_8_9k",
    "energy_9_11k",
    "energy_11_14k",

    "target_energy",
    "target_std",
    "target_max",

    "corr_peak",
    "corr_mean",
    "corr_std",

    "normalized_corr",

    "local_corr_mean",
    "local_corr_std",
    "local_corr_slope",

    "strongest_frequency",

    "high_low_ratio"
]

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
# FINAL INTERPRETATION
# ================================================================

print()
print("=" * 70)
print("SARV V4 INTERPRETATION")
print("=" * 70)

best_pair = None

if pair_results:

    best_pair = max(
        pair_results,
        key=lambda x: x[2]
    )

    print()

    print(
        f"Best pair: "
        f"{best_pair[0]} vs "
        f"{best_pair[1]}"
    )

    print(
        f"Pairwise accuracy: "
        f"{best_pair[2] * 100:.2f}%"
    )

full_accuracy = np.mean(
    scores
)

print()

if full_accuracy >= 0.80:

    print(
        "STRONG RESULT."
    )

    print(
        "The current acoustic signal contains "
        "substantial information for gesture recognition."
    )

elif full_accuracy >= 0.60:

    print(
        "PROMISING BUT NOT READY."
    )

    print(
        "Acoustic information is present, "
        "but the feature/model pipeline needs improvement."
    )

elif full_accuracy >= 0.45:

    print(
        "WEAK SIGNAL."
    )

    print(
        "Some acoustic differences may exist, "
        "but the current setup is not reliable."
    )

else:

    print(
        "POOR SEPARATION."
    )

    print(
        "The current feature representation "
        "does not reliably distinguish the gestures."
    )

print()
print(
    "IMPORTANT:"
)

print(
    "Pairwise results are more informative than "
    "the raw 7-class accuracy."
)

print(
    "If specific gesture pairs show high separation, "
    "we can build SARV around those gestures."
)

print()
print("=" * 70)
print("V4 ANALYSIS COMPLETE")
print("=" * 70)