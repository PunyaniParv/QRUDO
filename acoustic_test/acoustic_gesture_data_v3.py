import os
import csv
import time
import wave
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy import signal


# ================================================================
# SARV ACOUSTIC GESTURE DATA COLLECTOR V3
# ================================================================
#
# Purpose:
#   Collect labeled acoustic recordings for hand-gesture research.
#
# This version does NOT:
#   - estimate distance
#   - classify gestures
#   - control Windows
#
# It DOES:
#   - transmit an ultrasonic-ish chirp
#   - record microphone response
#   - save raw WAV recordings
#   - calculate basic signal features
#   - save metadata to CSV
#
# ================================================================


# ---------------- CONFIGURATION ----------------

INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

SAMPLE_RATE = 44100

CHIRP_START = 7500
CHIRP_END = 8500
CHIRP_DURATION = 0.100

# Amount of audio recorded for every sample
RECORD_DURATION = 0.80

# Silence before recording starts
PRE_RECORD_DELAY = 0.20

# Number of repetitions per gesture
REPETITIONS = 20

# Folder containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "gesture_dataset_v3")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")

CSV_FILE = os.path.join(DATASET_DIR, "metadata.csv")


# ---------------------------------------------------------------
# CREATE DIRECTORIES
# ---------------------------------------------------------------

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


# ---------------------------------------------------------------
# DEVICE INFORMATION
# ---------------------------------------------------------------

print("=" * 70)
print("SARV ACOUSTIC GESTURE DATA COLLECTOR V3")
print("=" * 70)

print()
print(f"Input device  : {INPUT_DEVICE}")
print(f"Output device : {OUTPUT_DEVICE}")
print(f"Sample rate   : {SAMPLE_RATE}")
print(f"Chirp         : {CHIRP_START} - {CHIRP_END} Hz")
print(f"Chirp duration: {CHIRP_DURATION * 1000:.0f} ms")
print(f"Record length : {RECORD_DURATION:.2f} sec")
print(f"Repetitions   : {REPETITIONS}")

print()
print("Dataset directory:")
print(DATASET_DIR)


# ---------------------------------------------------------------
# GENERATE CHIRP
# ---------------------------------------------------------------

def create_chirp():

    samples = int(SAMPLE_RATE * CHIRP_DURATION)

    t = np.linspace(
        0,
        CHIRP_DURATION,
        samples,
        endpoint=False
    )

    chirp = signal.chirp(
        t,
        f0=CHIRP_START,
        f1=CHIRP_END,
        t1=CHIRP_DURATION,
        method="linear"
    )

    # Fade in/out to reduce clicks
    fade_samples = int(0.01 * SAMPLE_RATE)

    fade = np.linspace(
        0,
        1,
        fade_samples
    )

    chirp[:fade_samples] *= fade
    chirp[-fade_samples:] *= fade[::-1]

    # Conservative amplitude
    chirp *= 0.25

    return chirp.astype(np.float32)


CHIRP = create_chirp()


# ---------------------------------------------------------------
# PLAY CHIRP + RECORD MICROPHONE
# ---------------------------------------------------------------

def record_sample():

    total_samples = int(
        SAMPLE_RATE * RECORD_DURATION
    )

    recording = np.zeros(
        total_samples,
        dtype=np.float32
    )

    recording_started = threading.Event()

    def callback(indata, frames, time_info, status):

        if status:
            print(
                f"\nAudio status: {status}",
                flush=True
            )

        if not recording_started.is_set():
            recording_started.set()

        start = callback.position
        end = min(
            start + frames,
            total_samples
        )

        if start < total_samples:

            recording[start:end] = (
                indata[:end - start, 0]
            )

        callback.position += frames

        if callback.position >= total_samples:
            raise sd.CallbackStop()

    callback.position = 0

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=INPUT_DEVICE,
            callback=callback,
            blocksize=1024
        ):

            # Start playback shortly after recording starts
            time.sleep(PRE_RECORD_DELAY)

            sd.play(
                CHIRP,
                samplerate=SAMPLE_RATE,
                device=OUTPUT_DEVICE,
                blocking=False
            )

            while callback.position < total_samples:
                time.sleep(0.005)

            sd.stop()

    except Exception as e:

        sd.stop()

        print()
        print("ERROR during recording:")
        print(e)

        return None

    return recording


# ---------------------------------------------------------------
# SAVE WAV
# ---------------------------------------------------------------

def save_wav(filename, audio):

    # Normalize only for storage if necessary.
    # We intentionally keep the normalization very conservative.
    max_value = np.max(np.abs(audio))

    if max_value > 0.999:

        audio_to_save = (
            audio / max_value * 0.95
        )

    else:

        audio_to_save = audio

    pcm = np.int16(
        np.clip(
            audio_to_save,
            -1,
            1
        ) * 32767
    )

    with wave.open(filename, "wb") as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------

def calculate_features(audio):

    if audio is None or len(audio) == 0:
        return {}

    rms = float(
        np.sqrt(
            np.mean(audio ** 2)
        )
    )

    raw_peak = float(
        np.max(np.abs(audio))
    )

    # Band-pass filter around transmitted chirp
    low = CHIRP_START - 1000
    high = CHIRP_END + 1000

    sos = signal.butter(
        4,
        [low, high],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos"
    )

    filtered = signal.sosfilt(
        sos,
        audio
    )

    filtered_rms = float(
        np.sqrt(
            np.mean(filtered ** 2)
        )
    )

    # Analytic signal
    analytic = signal.hilbert(
        filtered
    )

    envelope = np.abs(
        analytic
    )

    envelope_mean = float(
        np.mean(envelope)
    )

    envelope_peak = float(
        np.max(envelope)
    )

    # Frequency spectrum
    spectrum = np.abs(
        np.fft.rfft(filtered)
    )

    frequencies = np.fft.rfftfreq(
        len(filtered),
        1 / SAMPLE_RATE
    )

    mask = (
        (frequencies >= CHIRP_START - 1000)
        &
        (frequencies <= CHIRP_END + 1000)
    )

    if np.any(mask):

        band_freqs = frequencies[mask]
        band_spec = spectrum[mask]

        peak_frequency = float(
            band_freqs[
                np.argmax(band_spec)
            ]
        )

        spectral_energy = float(
            np.mean(
                band_spec ** 2
            )
        )

    else:

        peak_frequency = 0.0
        spectral_energy = 0.0

    # Time-frequency energy
    nperseg = min(
        1024,
        len(filtered)
    )

    try:

        f, t, Zxx = signal.stft(
            filtered,
            fs=SAMPLE_RATE,
            nperseg=nperseg,
            noverlap=nperseg // 2
        )

        stft_mask = (
            (f >= CHIRP_START - 1000)
            &
            (f <= CHIRP_END + 1000)
        )

        if np.any(stft_mask):

            tf_energy = np.mean(
                np.abs(
                    Zxx[stft_mask]
                ) ** 2,
                axis=0
            )

            temporal_variation = float(
                np.std(tf_energy)
            )

        else:

            temporal_variation = 0.0

    except Exception:

        temporal_variation = 0.0

    return {
        "rms": rms,
        "raw_peak": raw_peak,
        "filtered_rms": filtered_rms,
        "envelope_mean": envelope_mean,
        "envelope_peak": envelope_peak,
        "peak_frequency": peak_frequency,
        "spectral_energy": spectral_energy,
        "temporal_variation": temporal_variation
    }


# ---------------------------------------------------------------
# CSV INITIALIZATION
# ---------------------------------------------------------------

CSV_FIELDS = [
    "timestamp",
    "label",
    "repetition",
    "wav_file",
    "rms",
    "raw_peak",
    "filtered_rms",
    "envelope_mean",
    "envelope_peak",
    "peak_frequency",
    "spectral_energy",
    "temporal_variation"
]


if not os.path.exists(CSV_FILE):

    with open(
        CSV_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()


# ---------------------------------------------------------------
# GESTURE INSTRUCTIONS
# ---------------------------------------------------------------

GESTURES = {

    "idle":
        "Keep your hand completely still / remove hand.",

    "swipe_left":
        "Move your hand smoothly from RIGHT to LEFT.",

    "swipe_right":
        "Move your hand smoothly from LEFT to RIGHT.",

    "approach":
        "Move your hand smoothly TOWARD the laptop.",

    "away":
        "Move your hand smoothly AWAY from the laptop.",

    "hand_up":
        "Move your hand smoothly UP.",

    "hand_down":
        "Move your hand smoothly DOWN."
}


# ---------------------------------------------------------------
# RECORD ONE SAMPLE
# ---------------------------------------------------------------

def record_labeled_sample(
    label,
    repetition
):

    print()
    print(
        f"Recording {label.upper()} "
        f"{repetition}/{REPETITIONS}"
    )

    print(
        "Get ready..."
    )

    time.sleep(1.0)

    print(
        "3..."
    )

    time.sleep(0.5)

    print(
        "2..."
    )

    time.sleep(0.5)

    print(
        "1..."
    )

    time.sleep(0.5)

    print(
        "RECORDING"
    )

    audio = record_sample()

    if audio is None:

        print(
            "Recording failed."
        )

        return False

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"{label}_"
        f"{repetition:03d}_"
        f"{timestamp}.wav"
    )

    filepath = os.path.join(
        AUDIO_DIR,
        filename
    )

    save_wav(
        filepath,
        audio
    )

    features = calculate_features(
        audio
    )

    row = {
        "timestamp": timestamp,
        "label": label,
        "repetition": repetition,
        "wav_file": filename,
        **features
    }

    with open(
        CSV_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS
        )

        writer.writerow(row)

    print(
        f"Saved: {filename}"
    )

    print(
        f"RMS: {features['rms']:.6f} | "
        f"Peak Freq: {features['peak_frequency']:.1f} Hz | "
        f"Variation: {features['temporal_variation']:.6e}"
    )

    return True


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("GESTURES TO COLLECT")
    print("=" * 70)

    for gesture, instruction in GESTURES.items():

        print()
        print(
            f"{gesture:15s} -> {instruction}"
        )

    print()
    print("=" * 70)

    print(
        "IMPORTANT:"
    )

    print(
        "1. Keep the laptop in the same position."
    )

    print(
        "2. Keep the room reasonably quiet."
    )

    print(
        "3. Keep your body position consistent."
    )

    print(
        "4. Use approximately the same hand distance."
    )

    print(
        "5. Perform movements smoothly."
    )

    print()
    print(
        "We are collecting DATA, not testing accuracy yet."
    )

    input(
        "\nPress ENTER to begin..."
    )

    total = len(GESTURES) * REPETITIONS
    completed = 0

    for label, instruction in GESTURES.items():

        print()
        print("=" * 70)
        print(
            f"GESTURE: {label.upper()}"
        )
        print(
            instruction
        )
        print(
            "=" * 70
        )

        input(
            "\nPress ENTER when ready..."
        )

        for repetition in range(
            1,
            REPETITIONS + 1
        ):

            success = record_labeled_sample(
                label,
                repetition
            )

            if success:
                completed += 1

            print()
            print(
                f"Progress: "
                f"{completed}/{total}"
            )

    print()
    print("=" * 70)
    print("V3 DATA COLLECTION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Recordings collected: {completed}"
    )

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
        CSV_FILE
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Analyze the recordings and determine whether"
    )

    print(
        "LEFT / RIGHT / APPROACH / AWAY / UP / DOWN"
    )

    print(
        "have separable acoustic signatures."
    )


if __name__ == "__main__":
    main()