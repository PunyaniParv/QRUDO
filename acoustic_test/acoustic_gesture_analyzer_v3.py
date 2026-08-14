import os
import csv
import wave

import numpy as np
from scipy import signal
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ================================================================
# SARV ACOUSTIC GESTURE ANALYZER V3
# ================================================================
#
# Analyzes the dataset collected by:
#
#   acoustic_gesture_data_v3.py
#
# Goal:
#
#   Determine whether acoustic features can distinguish:
#
#       idle
#       swipe_left
#       swipe_right
#       approach
#       away
#       hand_up
#       hand_down
#
# ================================================================


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "gesture_dataset_v3"
)

AUDIO_DIR = os.path.join(
    DATASET_DIR,
    "audio"
)

CSV_FILE = os.path.join(
    DATASET_DIR,
    "metadata.csv"
)


SAMPLE_RATE = 44100

CHIRP_START = 7500
CHIRP_END = 8500


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("SARV ACOUSTIC GESTURE ANALYZER V3")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Audio:")
print(AUDIO_DIR)

print()


# ================================================================
# LOAD METADATA
# ================================================================

if not os.path.exists(CSV_FILE):

    raise FileNotFoundError(
        f"Metadata file not found:\n{CSV_FILE}"
    )


rows = []

with open(
    CSV_FILE,
    "r",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        rows.append(row)


print(
    f"Metadata records: {len(rows)}"
)


# ================================================================
# BANDPASS FILTER
# ================================================================

LOW_FREQ = CHIRP_START - 1200
HIGH_FREQ = CHIRP_END + 1200


sos = signal.butter(
    4,
    [
        LOW_FREQ,
        HIGH_FREQ
    ],
    btype="bandpass",
    fs=SAMPLE_RATE,
    output="sos"
)


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def extract_features(audio):

    features = []

    # ------------------------------------------------------------
    # Basic amplitude features
    # ------------------------------------------------------------

    rms = np.sqrt(
        np.mean(
            audio ** 2
        )
    )

    peak = np.max(
        np.abs(audio)
    )

    mean_abs = np.mean(
        np.abs(audio)
    )

    std = np.std(audio)

    features.extend([
        rms,
        peak,
        mean_abs,
        std
    ])


    # ------------------------------------------------------------
    # Bandpass acoustic response
    # ------------------------------------------------------------

    filtered = signal.sosfilt(
        sos,
        audio
    )

    filtered_rms = np.sqrt(
        np.mean(
            filtered ** 2
        )
    )

    filtered_peak = np.max(
        np.abs(filtered)
    )

    features.extend([
        filtered_rms,
        filtered_peak
    ])


    # ------------------------------------------------------------
    # Envelope
    # ------------------------------------------------------------

    analytic = signal.hilbert(
        filtered
    )

    envelope = np.abs(
        analytic
    )

    envelope_mean = np.mean(
        envelope
    )

    envelope_std = np.std(
        envelope
    )

    envelope_peak = np.max(
        envelope
    )

    features.extend([
        envelope_mean,
        envelope_std,
        envelope_peak
    ])


    # ------------------------------------------------------------
    # Divide recording into temporal regions
    # ------------------------------------------------------------

    n = len(filtered)

    sections = np.array_split(
        filtered,
        8
    )

    for section in sections:

        section_rms = np.sqrt(
            np.mean(
                section ** 2
            )
        )

        section_peak = np.max(
            np.abs(section)
        )

        features.extend([
            section_rms,
            section_peak
        ])


    # ------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------

    window = np.hanning(
        len(filtered)
    )

    spectrum = np.abs(
        np.fft.rfft(
            filtered * window
        )
    )

    frequencies = np.fft.rfftfreq(
        len(filtered),
        1 / SAMPLE_RATE
    )


    band_mask = (
        (frequencies >= LOW_FREQ)
        &
        (frequencies <= HIGH_FREQ)
    )


    band_spectrum = spectrum[
        band_mask
    ]

    band_frequencies = frequencies[
        band_mask
    ]


    if len(band_spectrum) > 0:

        total_energy = np.sum(
            band_spectrum ** 2
        ) + 1e-12

        peak_index = np.argmax(
            band_spectrum
        )

        peak_frequency = (
            band_frequencies[
                peak_index
            ]
        )

        spectral_centroid = (
            np.sum(
                band_frequencies *
                band_spectrum
            )
            /
            (
                np.sum(
                    band_spectrum
                )
                + 1e-12
            )
        )

        spectral_spread = np.sqrt(
            np.sum(
                (
                    band_frequencies
                    -
                    spectral_centroid
                ) ** 2
                *
                band_spectrum
            )
            /
            (
                np.sum(
                    band_spectrum
                )
                + 1e-12
            )
        )

        spectral_energy = (
            total_energy
        )

    else:

        peak_frequency = 0
        spectral_centroid = 0
        spectral_spread = 0
        spectral_energy = 0


    features.extend([
        peak_frequency,
        spectral_centroid,
        spectral_spread,
        spectral_energy
    ])


    # ------------------------------------------------------------
    # Frequency-band energies
    # ------------------------------------------------------------

    frequency_bands = [
        (6500, 7000),
        (7000, 7500),
        (7500, 8000),
        (8000, 8500),
        (8500, 9000),
        (9000, 9500)
    ]


    for low, high in frequency_bands:

        mask = (
            (frequencies >= low)
            &
            (frequencies < high)
        )

        if np.any(mask):

            energy = np.mean(
                spectrum[mask] ** 2
            )

        else:

            energy = 0

        features.append(
            energy
        )


    # ------------------------------------------------------------
    # STFT temporal-frequency features
    # ------------------------------------------------------------

    try:

        f, t, zxx = signal.stft(
            filtered,
            fs=SAMPLE_RATE,
            nperseg=1024,
            noverlap=768
        )

        mask = (
            (f >= LOW_FREQ)
            &
            (f <= HIGH_FREQ)
        )

        if np.any(mask):

            stft_energy = np.abs(
                zxx[mask]
            ) ** 2

            temporal_energy = np.mean(
                stft_energy,
                axis=0
            )

            # Resample temporal energy to exactly
            # 12 values.

            if len(temporal_energy) >= 2:

                x_old = np.linspace(
                    0,
                    1,
                    len(temporal_energy)
                )

                x_new = np.linspace(
                    0,
                    1,
                    12
                )

                temporal_profile = np.interp(
                    x_new,
                    x_old,
                    temporal_energy
                )

                # Normalize
                temporal_profile /= (
                    np.mean(
                        temporal_profile
                    )
                    + 1e-12
                )

                features.extend(
                    temporal_profile.tolist()
                )

            else:

                features.extend(
                    [0.0] * 12
                )

        else:

            features.extend(
                [0.0] * 12
            )

    except Exception:

        features.extend(
            [0.0] * 12
        )


    return np.array(
        features,
        dtype=np.float64
    )


# ================================================================
# LOAD WAV
# ================================================================

def load_wav(path):

    with wave.open(
        path,
        "rb"
    ) as wf:

        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(
            wf.getnframes()
        )

    if channels != 1:

        raise ValueError(
            "Expected mono WAV"
        )

    if rate != SAMPLE_RATE:

        raise ValueError(
            f"Unexpected sample rate: {rate}"
        )

    if sample_width == 2:

        audio = np.frombuffer(
            frames,
            dtype=np.int16
        ).astype(
            np.float32
        ) / 32768.0

    else:

        raise ValueError(
            f"Unsupported sample width: "
            f"{sample_width}"
        )

    return audio


# ================================================================
# EXTRACT DATASET
# ================================================================

X = []
y = []
groups = []
filenames = []


print()
print("=" * 70)
print("EXTRACTING FEATURES")
print("=" * 70)


for index, row in enumerate(rows):

    filename = row["wav_file"]
    label = row["label"]

    path = os.path.join(
        AUDIO_DIR,
        filename
    )

    if not os.path.exists(path):

        print(
            f"WARNING: missing {filename}"
        )

        continue

    try:

        audio = load_wav(
            path
        )

        features = extract_features(
            audio
        )

        X.append(
            features
        )

        y.append(
            label
        )

        # Repetition is used as the grouping
        # variable to prevent nearly identical
        # samples from leaking between folds.
        #
        # Timestamp identifies the actual recording.

        groups.append(
            f"{label}_{row['repetition']}"
        )

        filenames.append(
            filename
        )

    except Exception as e:

        print(
            f"ERROR: {filename}: {e}"
        )


X = np.array(X)
y = np.array(y)
groups = np.array(groups)


print()
print(
    f"Usable recordings: {len(X)}"
)

print(
    f"Feature count: {X.shape[1]}"
)


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

classes, counts = np.unique(
    y,
    return_counts=True
)

for cls, count in zip(
    classes,
    counts
):

    print(
        f"{cls:15s}: {count}"
    )


# ================================================================
# BASIC FEATURE STATISTICS
# ================================================================

print()
print("=" * 70)
print("FEATURE VARIATION")
print("=" * 70)


feature_means = np.mean(
    X,
    axis=0
)

feature_stds = np.std(
    X,
    axis=0
)

print(
    f"Average feature std: "
    f"{np.mean(feature_stds):.6f}"
)

print(
    f"Maximum feature std: "
    f"{np.max(feature_stds):.6f}"
)


# ================================================================
# CLASS SEPARATION
# ================================================================

print()
print("=" * 70)
print("CLASS SEPARATION")
print("=" * 70)


for cls in classes:

    class_data = X[
        y == cls
    ]

    centroid = np.mean(
        class_data,
        axis=0
    )

    distance = np.linalg.norm(
        class_data - centroid,
        axis=1
    )

    print(
        f"{cls:15s} "
        f"within-class distance: "
        f"{np.mean(distance):.4f}"
    )


# ================================================================
# RANDOM FOREST
# ================================================================

print()
print("=" * 70)
print("CLASSIFICATION TEST")
print("=" * 70)

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
)


# ---------------------------------------------------------------
# GROUPED CROSS VALIDATION
# ---------------------------------------------------------------

unique_groups = np.unique(
    groups
)

n_splits = min(
    5,
    len(unique_groups)
)


cv = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)


print()
print(
    f"Cross-validation folds: "
    f"{n_splits}"
)

print(
    "Training classifier..."
)


predictions = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=cv,
    n_jobs=1
)


# ================================================================
# RESULTS
# ================================================================

accuracy = accuracy_score(
    y,
    predictions
)


print()
print("=" * 70)
print("ACOUSTIC CLASSIFICATION RESULT")
print("=" * 70)

print()
print(
    f"Overall accuracy: "
    f"{accuracy * 100:.2f}%"
)


print()
print("Classification report:")
print()

print(
    classification_report(
        y,
        predictions,
        digits=3
    )
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

matrix = confusion_matrix(
    y,
    predictions,
    labels=classes
)


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(
    "Actual \\ Predicted"
)

print(
    " " * 18 +
    " ".join(
        f"{c[:8]:>9}"
        for c in classes
    )
)

for cls, row_matrix in zip(
    classes,
    matrix
):

    print(
        f"{cls[:16]:16s} " +
        " ".join(
            f"{value:9d}"
            for value in row_matrix
        )
    )


# ================================================================
# PER-CLASS ACCURACY
# ================================================================

print()
print("=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for i, cls in enumerate(
    classes
):

    total = np.sum(
        matrix[i]
    )

    correct = matrix[i, i]

    if total > 0:

        class_accuracy = (
            correct / total
        ) * 100

    else:

        class_accuracy = 0

    print(
        f"{cls:15s}: "
        f"{class_accuracy:6.2f}%"
    )


# ================================================================
# FINAL INTERPRETATION
# ================================================================

print()
print("=" * 70)
print("SARV V3 INTERPRETATION")
print("=" * 70)

print()

if accuracy >= 0.90:

    print(
        "STRONG RESULT."
    )

    print(
        "The acoustic signal appears to contain"
    )

    print(
        "strongly separable information."
    )

    print()
    print(
        "Continue with acoustic recognition."
    )

elif accuracy >= 0.75:

    print(
        "PROMISING RESULT."
    )

    print(
        "The acoustic signal contains useful"
    )

    print(
        "gesture information, but the system"
    )

    print(
        "needs better features and/or signal"
    )

    print(
        "processing before real-time control."
    )

    print()
    print(
        "Continue with acoustic recognition."
    )

elif accuracy >= 0.60:

    print(
        "WEAK BUT INTERESTING RESULT."
    )

    print(
        "There may be acoustic information,"
    )

    print(
        "but the current feature representation"
    )

    print(
        "is not sufficiently reliable."
    )

    print()
    print(
        "Next step should be improved"
    )

    print(
        "signal processing before abandoning"
    )

    print(
        "the acoustic approach."
    )

else:

    print(
        "POOR SEPARATION."
    )

    print(
        "The current acoustic configuration"
    )

    print(
        "does not reliably distinguish"
    )

    print(
        "the gestures."
    )

    print()
    print(
        "We should inspect the raw recordings"
    )

    print(
        "before deciding whether acoustic"
    )

    print(
        "recognition is viable."
    )


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)