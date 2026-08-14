import os
import csv
import time
import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from datetime import datetime

# ================================================================
# SARV CONTROLLED-DISTANCE REFLECTION VALIDATION v1
# RECORDING TOOL
#
# This reuses the EXACT acoustic signal chain from
#   acoustic_gesture_test_v5.py
# so that recordings are directly comparable to V5/V6/V7/V8.
#
# It does NOT collect gestures. It records a STATIONARY hand at
# known distances, plus a no-hand baseline.
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_ROOT = os.path.join(
    BASE_DIR,
    "audio"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "metadata.csv"
)

# ================================================================
# AUDIO CONFIGURATION — PRESERVED FROM V5
# ================================================================

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

SAMPLE_RATE = 44100

CHIRP_LOW = 7500
CHIRP_HIGH = 8500

CHIRP_DURATION = 0.100

RECORDING_DURATION = 2.0

# Repetitions at each condition. More than V5 to evaluate stability.
REPETITIONS = 10

# Valid conditions
CONDITIONS = [
    "baseline",
    "10cm",
    "20cm",
    "30cm",
    "40cm",
    "50cm"
]

# ================================================================
# DEVICE INFORMATION
# ================================================================

print("=" * 70)
print("SARV CONTROLLED-DISTANCE REFLECTION VALIDATION v1")
print("RECORDING TOOL")
print("=" * 70)

print()
print(f"Input device  : {INPUT_DEVICE}")
print(f"Output device : {OUTPUT_DEVICE}")
print(f"Sample rate   : {SAMPLE_RATE}")
print(f"Chirp         : {CHIRP_LOW} - {CHIRP_HIGH} Hz")
print(f"Chirp duration: {CHIRP_DURATION:.3f} s")
print(f"Recording     : {RECORDING_DURATION:.1f} seconds")
print(f"Repetitions   : {REPETITIONS} per condition")
print(f"Conditions    : {', '.join(CONDITIONS)}")

print()
print("Audio root:")
print(AUDIO_ROOT)
print()
print("Metadata:")
print(METADATA_FILE)

for cond in CONDITIONS:
    os.makedirs(
        os.path.join(
            AUDIO_ROOT,
            cond
        ),
        exist_ok=True
    )

# ================================================================
# GENERATE CHIRP — IDENTICAL TO V5
# ================================================================

def generate_chirp():

    n = int(
        CHIRP_DURATION * SAMPLE_RATE
    )

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

    signal *= 0.25

    return signal.astype(np.float32)


CHIRP_SIGNAL = generate_chirp()

# ================================================================
# AUDIO RECORDING — IDENTICAL TO V5
# ================================================================

def record_trial():

    total_samples = int(
        RECORDING_DURATION * SAMPLE_RATE
    )

    recording = np.zeros(
        total_samples,
        dtype=np.float32
    )

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

            if elapsed >= next_chirp:

                sd.play(
                    CHIRP_SIGNAL,
                    samplerate=SAMPLE_RATE,
                    device=OUTPUT_DEVICE,
                    blocking=False
                )

                chirp_count += 1

                next_chirp += 0.250

            remaining = (
                RECORDING_DURATION - elapsed
            )

            block_size = min(
                1024,
                max(
                    1,
                    int(
                        remaining
                        * SAMPLE_RATE
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
        return None, 0

    audio = np.concatenate(blocks)

    audio = audio[:total_samples]

    return audio, chirp_count


# ================================================================
# SAVE WAV — IDENTICAL TO V5
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
# BASIC FEATURES (reproducibility)
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

    return {
        "rms": rms,
        "peak": peak
    }


# ================================================================
# METADATA
# ================================================================

CSV_FIELDS = [
    "timestamp",
    "condition",
    "distance_cm",
    "repetition",
    "wav_file",
    "sample_rate",
    "chirp_low",
    "chirp_high",
    "chirp_duration",
    "recording_duration",
    "expected_chirps",
    "chirps_played",
    "rms",
    "raw_peak"
]


def write_header_if_needed():

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


CONDITION_DISTANCE = {
    "baseline": 0,
    "10cm": 10,
    "20cm": 20,
    "30cm": 30,
    "40cm": 40,
    "50cm": 50
}

# ================================================================
# USER INSTRUCTIONS
# ================================================================

print()
print("=" * 70)
print("IMPORTANT EXPERIMENT RULES")
print("=" * 70)

print()
print("Keep the laptop COMPLETELY stationary.")
print()
print("Keep your body as still as possible.")
print()
print("The hand must be opened flat, palm facing the laptop.")
print()
print("At each distance the hand must be COMPLETELY STILL.")
print()
print("Measure distance with a ruler from the laptop speaker/"
      "microphone area to the palm.")

print()
print("Conditions:")
print("  baseline : NO hand in front of the laptop.")
print("  10cm     : palm 10 cm from the laptop.")
print("  20cm     : palm 20 cm from the laptop.")
print("  30cm     : palm 30 cm from the laptop.")
print("  40cm     : palm 40 cm from the laptop.")
print("  50cm     : palm 50 cm from the laptop.")

print()
print(
    "Expected round-trip delays for reference:"
)

speed_of_sound = 343.0

for dist_cm in [
    10,
    20,
    30,
    40,
    50
]:

    delay_ms = (
        2.0 * dist_cm / 100.0
        / speed_of_sound
        * 1000.0
    )

    print(
        f"  {dist_cm:>2d} cm ~ "
        f"{delay_ms:.3f} ms"
    )

print()
input(
    "Press ENTER when you are ready to begin..."
)


# ================================================================
# CONDITION SELECTION LOOP
# ================================================================

print()
print("=" * 70)
print("CONDITION FLOW")
print("=" * 70)

print()
print(
    "You will record baseline first, then 10, 20, 30, 40, 50 cm "
    "in increasing order."
)

print()

write_header_if_needed()

for condition in CONDITIONS:

    dist_cm = CONDITION_DISTANCE[condition]

    print()
    print("#" * 70)
    print(
        f"CONDITION: {condition.upper()}"
    )
    print("#" * 70)

    print()

    if condition == "baseline":

        print(
            "Remove your hand from in front of the laptop."
        )

        print(
            "Keep the area clear and stay still."
        )

    else:

        print(
            f"Place your open palm exactly "
            f"{dist_cm} cm from the laptop "
            f"(speaker/microphone area)."
        )

        print(
            "Palm faces the laptop, fingers together, "
            "hand completely still."
        )

    print()
    print(
        "Setup:"
    )

    print(
        f"  Sample rate       : {SAMPLE_RATE}"
    )

    print(
        f"  Chirp             : {CHIRP_LOW}-{CHIRP_HIGH} Hz, "
        f"{CHIRP_DURATION:.3f}s"
    )

    print(
        f"  Repetitions       : {REPETITIONS}"
    )

    input(
        "Press ENTER when ready to begin this condition..."
    )

    for repetition in range(
        1,
        REPETITIONS + 1
    ):

        print()

        print(
            f"{condition.upper():<9} "
            f"{repetition:02d}/{REPETITIONS}"
        )

        print(
            "Keep hand STILL. Recording in 1 second..."
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

        audio, chirps_played = record_trial()

        if audio is None:

            print(
                "ERROR: recording failed"
            )

            continue

        features = calculate_basic_features(
            audio
        )

        filename = (
            f"{condition}_"
            f"{repetition:03d}_"
            f"{timestamp}.wav"
        )

        wav_path = os.path.join(
            AUDIO_ROOT,
            condition,
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
                "condition": condition,
                "distance_cm": dist_cm,
                "repetition": repetition,
                "wav_file": filename,
                "sample_rate": SAMPLE_RATE,
                "chirp_low": CHIRP_LOW,
                "chirp_high": CHIRP_HIGH,
                "chirp_duration": CHIRP_DURATION,
                "recording_duration": RECORDING_DURATION,
                "expected_chirps": 7,
                "chirps_played": chirps_played,
                "rms": features["rms"],
                "raw_peak": features["peak"]
            }

            writer.writerow(row)

        print(
            f"Saved: {filename}"
        )

        print(
            f"Chirps played: {chirps_played} | "
            f"RMS: {features['rms']:.6f} | "
            f"Peak: {features['peak']:.6f}"
        )

        time.sleep(0.5)

    print()
    print(
        f"Condition {condition} complete."
    )

    if condition != "50cm":

        input(
            "Press ENTER to continue to the next condition..."
        )


# ================================================================
# COMPLETE
# ================================================================

print()
print("=" * 70)
print("DISTANCE VALIDATION RECORDING COMPLETE")
print("=" * 70)

print()
print("Audio:")
print(AUDIO_ROOT)

print()
print("Metadata:")
print(METADATA_FILE)

print()
print(
    "Expected recordings: "
    f"{len(CONDITIONS) * REPETITIONS}"
)

print()
print(
    "Next step:"
)

print(
    "Run the distance validation analysis tool."
)

print()
print("=" * 70)