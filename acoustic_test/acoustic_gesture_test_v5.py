import os
import csv
import time
import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from datetime import datetime

# ================================================================
# SARV ACOUSTIC GESTURE TEST V5
# CONTROLLED APPROACH / AWAY EXPERIMENT
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

os.makedirs(AUDIO_DIR, exist_ok=True)

# ================================================================
# AUDIO CONFIGURATION
# ================================================================

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

SAMPLE_RATE = 44100

CHIRP_LOW = 7500
CHIRP_HIGH = 8500

CHIRP_DURATION = 0.100

RECORDING_DURATION = 2.0

# Number of recordings for each gesture
REPETITIONS = 20

# ================================================================
# DEVICE INFORMATION
# ================================================================

print("=" * 70)
print("SARV ACOUSTIC GESTURE TEST V5")
print("CONTROLLED APPROACH / AWAY EXPERIMENT")
print("=" * 70)

print()
print(f"Input device  : {INPUT_DEVICE}")
print(f"Output device : {OUTPUT_DEVICE}")
print(f"Sample rate   : {SAMPLE_RATE}")
print(f"Chirp         : {CHIRP_LOW} - {CHIRP_HIGH} Hz")
print(f"Recording     : {RECORDING_DURATION:.1f} seconds")

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Audio:")
print(AUDIO_DIR)

# ================================================================
# GENERATE CHIRP
# ================================================================

def generate_chirp():

    n = int(CHIRP_DURATION * SAMPLE_RATE)

    t = np.linspace(
        0,
        CHIRP_DURATION,
        n,
        endpoint=False
    )

    signal = chirp(
        t,
        f0=CHIRP_LOW,
        f1=CHIRP_HIGH,
        t1=CHIRP_DURATION,
        method="linear"
    )

    window = np.hanning(n)

    signal *= window

    # Keep speaker output at a safe level
    signal *= 0.25

    return signal.astype(np.float32)


CHIRP_SIGNAL = generate_chirp()

# ================================================================
# AUDIO RECORDING
# ================================================================

def record_trial():

    total_samples = int(
        RECORDING_DURATION * SAMPLE_RATE
    )

    recording = np.zeros(
        total_samples,
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # Start input stream
    # ------------------------------------------------------------

    stream = sd.InputStream(
        device=INPUT_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    stream.start()

    start_time = time.perf_counter()

    next_chirp = 0.15

    chirp_count = 0

    blocks = []

    try:

        while True:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            if elapsed >= RECORDING_DURATION:
                break

            # ----------------------------------------------------
            # Play chirp
            # ----------------------------------------------------

            if elapsed >= next_chirp:

                sd.play(
                    CHIRP_SIGNAL,
                    samplerate=SAMPLE_RATE,
                    device=OUTPUT_DEVICE,
                    blocking=False
                )

                chirp_count += 1

                next_chirp += 0.250

            # ----------------------------------------------------
            # Read microphone
            # ----------------------------------------------------

            remaining = (
                RECORDING_DURATION - elapsed
            )

            block_size = min(
                1024,
                max(
                    1,
                    int(
                        remaining *
                        SAMPLE_RATE
                    )
                )
            )

            data, overflowed = stream.read(
                block_size
            )

            blocks.append(
                data[:, 0].copy()
            )

    finally:

        stream.stop()
        stream.close()

        sd.stop()

    if not blocks:
        return None

    audio = np.concatenate(blocks)

    audio = audio[:total_samples]

    return audio


# ================================================================
# SAVE WAV
# ================================================================

def save_wav(path, audio):

    import wave

    audio_int16 = np.clip(
        audio,
        -1.0,
        1.0
    )

    audio_int16 = (
        audio_int16 * 32767
    ).astype(np.int16)

    with wave.open(path, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)

        wav.writeframes(
            audio_int16.tobytes()
        )


# ================================================================
# BASIC FEATURES
# ================================================================

def calculate_basic_features(audio):

    rms = float(
        np.sqrt(
            np.mean(audio ** 2)
        )
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    # Divide recording into 8 temporal sections
    sections = np.array_split(
        audio,
        8
    )

    section_rms = []

    for section in sections:

        value = np.sqrt(
            np.mean(
                section ** 2
            )
        )

        section_rms.append(
            float(value)
        )

    return {
        "rms": rms,
        "peak": peak,
        "section_rms": section_rms
    }


# ================================================================
# METADATA
# ================================================================

CSV_FIELDS = [
    "timestamp",
    "label",
    "repetition",
    "wav_file",
    "rms",
    "raw_peak",
    "section_rms_1",
    "section_rms_2",
    "section_rms_3",
    "section_rms_4",
    "section_rms_5",
    "section_rms_6",
    "section_rms_7",
    "section_rms_8"
]


if not os.path.exists(METADATA_FILE):

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()


# ================================================================
# USER INSTRUCTIONS
# ================================================================

print()
print("=" * 70)
print("IMPORTANT EXPERIMENT RULES")
print("=" * 70)

print()
print("Keep the laptop completely stationary.")
print()
print("Keep your body as still as possible.")
print()
print("Only the hand should move.")
print()
print("For APPROACH:")
print("Move your hand slowly toward the laptop.")
print()
print("For AWAY:")
print("Move your hand slowly away from the laptop.")
print()
print("For IDLE:")
print("Keep your hand completely still.")
print()

input(
    "Press ENTER when you are ready to begin..."
)


# ================================================================
# EXPERIMENT
# ================================================================

GESTURES = [
    "idle",
    "approach",
    "away"
]


for label in GESTURES:

    print()
    print("=" * 70)
    print(f"GESTURE: {label.upper()}")
    print("=" * 70)

    if label == "idle":

        print()
        print("Place your hand in the test position.")
        print("Keep it COMPLETELY STILL.")

    elif label == "approach":

        print()
        print("Start with your hand approximately 60 cm away.")
        print("Slowly move toward approximately 20 cm.")

    elif label == "away":

        print()
        print("Start with your hand approximately 20 cm away.")
        print("Slowly move toward approximately 60 cm.")

    print()

    input(
        "Press ENTER when ready..."
    )

    for repetition in range(
        1,
        REPETITIONS + 1
    ):

        print()
        print(
            f"{label.upper():<10} "
            f"{repetition:02d}/{REPETITIONS}"
        )

        print(
            "Recording in 1 second..."
        )

        time.sleep(1)

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        print(
            "RECORDING..."
        )

        audio = record_trial()

        if audio is None:

            print(
                "ERROR: recording failed"
            )

            continue

        features = calculate_basic_features(
            audio
        )

        filename = (
            f"{label}_"
            f"{repetition:03d}_"
            f"{timestamp}.wav"
        )

        wav_path = os.path.join(
            AUDIO_DIR,
            filename
        )

        save_wav(
            wav_path,
            audio
        )

        with open(
            METADATA_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=CSV_FIELDS
            )

            row = {
                "timestamp": timestamp,
                "label": label,
                "repetition": repetition,
                "wav_file": filename,
                "rms": features["rms"],
                "raw_peak": features["peak"]
            }

            for i, value in enumerate(
                features["section_rms"],
                start=1
            ):

                row[
                    f"section_rms_{i}"
                ] = value

            writer.writerow(row)

        print(
            f"Saved: {filename}"
        )

        print(
            f"RMS: {features['rms']:.6f} | "
            f"Peak: {features['peak']:.6f}"
        )

        time.sleep(0.5)


# ================================================================
# COMPLETE
# ================================================================

print()
print("=" * 70)
print("V5 DATA COLLECTION COMPLETE")
print("=" * 70)

print()
print(
    f"Audio directory:"
)

print(
    AUDIO_DIR
)

print()
print(
    f"Metadata:"
)

print(
    METADATA_FILE
)

print()
print(
    "Expected recordings:"
)

print(
    len(GESTURES) * REPETITIONS
)

print()
print(
    "Next step:"
)

print(
    "Run the V5 acoustic analyzer."
)

print(
    "We will analyze echo timing, Doppler, "
    "frequency shift and temporal motion."
)

print()
print("=" * 70)