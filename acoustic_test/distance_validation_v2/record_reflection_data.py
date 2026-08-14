"""SARV distance validation v2 -- recording tool.

Records interleaved baseline (B) and reflector (R) trials at each
distance. The speaker-to-microphone separation is measured and stored in
experiment_geometry.json before recording begins.

Protocol per distance (10 + 10 = 20 trials, interleaved B,R,B,R,...):
    trial  1: baseline (no reflector)
    trial  2: reflector at distance d
    trial  3: baseline
    trial  4: reflector
    ...
    trial 20: reflector

Total: 20 trials x 5 distances = 100 recordings.

Usage:
    python record_reflection_data.py
"""

import os
import csv
import json
import time
import wave
import sys

import numpy as np
import sounddevice as sd
from scipy.signal import chirp
from datetime import datetime

import config

# ------------------------------------------------------------------
# Chirp generation (identical to v1 for comparability)
# ------------------------------------------------------------------
def generate_chirp():
    n = int(config.CHIRP_DURATION * config.SAMPLE_RATE)
    t = np.linspace(0, config.CHIRP_DURATION, n, endpoint=False)
    signal = chirp(t, f0=config.CHIRP_LOW, f1=config.CHIRP_HIGH,
                   t1=config.CHIRP_DURATION, method="linear")
    signal *= np.hanning(n)
    signal *= 0.25
    return signal.astype(np.float32)


CHIRP_SIGNAL = generate_chirp()


# ------------------------------------------------------------------
# Recording (identical to v1)
# ------------------------------------------------------------------
def record_trial():
    total_samples = int(config.RECORDING_DURATION * config.SAMPLE_RATE)
    recording = np.zeros(total_samples, dtype=np.float32)

    stream = sd.InputStream(
        device=config.INPUT_DEVICE,
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    stream.start()

    start_time = time.perf_counter()
    next_chirp = config.FIRST_CHIRP_TIME
    chirp_count = 0
    blocks = []

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= config.RECORDING_DURATION:
                break
            if elapsed >= next_chirp:
                sd.play(CHIRP_SIGNAL, samplerate=config.SAMPLE_RATE,
                        device=config.OUTPUT_DEVICE, blocking=False)
                chirp_count += 1
                next_chirp += config.CHIRP_INTERVAL
            remaining = config.RECORDING_DURATION - elapsed
            block_size = min(1024, max(1, int(remaining * config.SAMPLE_RATE)))
            data, overflowed = stream.read(block_size)
            blocks.append(data[:, 0].copy())
    finally:
        stream.stop()
        stream.close()
        sd.stop()

    if not blocks:
        return None, 0
    audio = np.concatenate(blocks)[:total_samples]
    return audio, chirp_count


# ------------------------------------------------------------------
# Save WAV
# ------------------------------------------------------------------
def save_wav(path, audio):
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(config.SAMPLE_RATE)
        wav.writeframes(audio_int16.tobytes())


def calculate_basic_features(audio):
    return {
        "rms": float(np.sqrt(np.mean(audio ** 2))),
        "peak": float(np.max(np.abs(audio))),
    }


# ------------------------------------------------------------------
# Geometry setup
# ------------------------------------------------------------------
def setup_geometry():
    print("\n" + "=" * 72)
    print("GEOMETRY SETUP (required before recording)")
    print("=" * 72)
    print("\nMeasure the distance between the CENTER of the laptop's")
    print("speaker and the CENTER of the microphone with a ruler.")
    print("This is the speaker-to-microphone separation s (cm).")
    print("The reflector will be placed at a perpendicular distance d")
    print("from the laptop, centered between speaker and microphone.")

    while True:
        raw = input("\nSpeaker-to-microphone separation s (cm): ").strip()
        try:
            s = float(raw)
            if s <= 0:
                print("  Must be a positive number.")
                continue
            break
        except ValueError:
            print("  Invalid number. Try again.")

    while True:
        raw = input("Speed of sound (m/s) [default 343.0]: ").strip()
        if raw == "":
            c = 343.0
            break
        try:
            c = float(raw)
            if c <= 0:
                print("  Must be positive.")
                continue
            break
        except ValueError:
            print("  Invalid number. Try again.")

    reflector = input("Reflector description (e.g. 'A4 clipboard, rigid flat'): ").strip()
    if not reflector:
        reflector = "rigid flat reflector"

    geometry = {
        "speaker_mic_separation_cm": s,
        "speed_of_sound": c,
        "reflector_description": reflector,
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config.GEOMETRY_FILE, "w", encoding="utf-8") as f:
        json.dump(geometry, f, indent=2)

    print("\nGeometry saved to:")
    print(config.GEOMETRY_FILE)
    print(f"  s = {s} cm, c = {c} m/s, reflector = {reflector}")
    return geometry


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------
CSV_FIELDS = [
    "timestamp",
    "distance_cm",
    "condition",          # baseline | reflector
    "trial_index",        # 1..20 within distance
    "trial_number",       # 1..10 within condition
    "wav_file",
    "sample_rate",
    "chirp_low",
    "chirp_high",
    "chirp_duration",
    "recording_duration",
    "expected_chirps",
    "chirps_played",
    "rms",
    "raw_peak",
    "speaker_mic_separation_cm",
    "speed_of_sound",
    "expected_bistatic_delay_ms",
    "reflector_description",
]


def write_header_if_needed():
    if not os.path.exists(config.METADATA_FILE):
        with open(config.METADATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 72)
    print("SARV DISTANCE VALIDATION v2 -- RECORDING TOOL")
    print("=" * 72)

    print(f"\nInput device  : {config.INPUT_DEVICE}")
    print(f"Output device : {config.OUTPUT_DEVICE}")
    print(f"Sample rate   : {config.SAMPLE_RATE}")
    print(f"Chirp         : {config.CHIRP_LOW}-{config.CHIRP_HIGH} Hz, "
          f"{config.CHIRP_DURATION*1000:.0f} ms")
    print(f"Recording     : {config.RECORDING_DURATION:.1f} s")
    print(f"Chirps/trial  : {config.EXPECTED_CHIRPS}")
    print(f"Distances     : {config.DISTANCES_CM} cm")
    print(f"Trials/distance: {2 * config.TRIALS_PER_CONDITION} "
          f"({config.TRIALS_PER_CONDITION} baseline + "
          f"{config.TRIALS_PER_CONDITION} reflector)")
    print(f"Total         : {2 * config.TRIALS_PER_CONDITION * len(config.DISTANCES_CM)} recordings")

    os.makedirs(config.AUDIO_ROOT, exist_ok=True)

    geometry = setup_geometry()
    s = geometry["speaker_mic_separation_cm"]
    c = geometry["speed_of_sound"]

    print("\n" + "=" * 72)
    print("EXPECTED BISTATIC DELAYS")
    print("=" * 72)
    for d in config.DISTANCES_CM:
        ms = config.bistatic_delay_ms(d, s, c)
        print(f"  {d:>3} cm -> {ms:.3f} ms")

    print("\n" + "=" * 72)
    print("EXPERIMENT RULES")
    print("=" * 72)
    print("\n- Keep the laptop COMPLETELY stationary.")
    print("- Keep your body as still as possible.")
    print("- Use a RIGID FLAT reflector (book, clipboard, etc.).")
    print("- Measure distance d with a ruler from the laptop surface")
    print("  (speaker/mic area) to the reflector face.")
    print("- Center the reflector between speaker and microphone.")
    print("- Reflector face must be parallel to the laptop surface.")
    print("- Baseline trials: remove the reflector entirely.")
    print("- Reflector trials: hold the reflector STILL at distance d.")

    write_header_if_needed()

    input("\nPress ENTER when you are ready to begin...")

    for dist_cm in config.DISTANCES_CM:
        expected_ms = config.bistatic_delay_ms(dist_cm, s, c)
        print("\n" + "#" * 72)
        print(f"DISTANCE: {dist_cm} cm  (expected delay {expected_ms:.3f} ms)")
        print("#" * 72)

        for trial_index in range(1, 2 * config.TRIALS_PER_CONDITION + 1):
            # Interleave B, R, B, R, ...
            is_reflector = (trial_index % 2 == 0)
            condition = "reflector" if is_reflector else "baseline"
            trial_number = trial_index // 2 if is_reflector else (trial_index + 1) // 2

            print("\n" + "-" * 72)
            print(f"Trial {trial_index:02d}/20 | {condition.upper()} "
                  f"{trial_number:02d}/10 | distance {dist_cm} cm")

            if condition == "baseline":
                print("  REMOVE the reflector. Keep the area clear.")
            else:
                print(f"  Place the rigid flat reflector at {dist_cm} cm,")
                print("  centered between speaker and mic, face parallel,")
                print("  completely STILL.")

            input("  Press ENTER when ready...")
            print("  Recording in 1 second...")
            time.sleep(1)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            print("  RECORDING...")
            audio, chirps_played = record_trial()
            if audio is None:
                print("  ERROR: recording failed")
                continue

            features = calculate_basic_features(audio)
            filename = (f"d{dist_cm:02d}_{condition}_{trial_number:02d}_"
                        f"{timestamp}.wav")
            wav_path = os.path.join(config.AUDIO_ROOT, filename)
            save_wav(wav_path, audio)

            with open(config.METADATA_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow({
                    "timestamp": timestamp,
                    "distance_cm": dist_cm,
                    "condition": condition,
                    "trial_index": trial_index,
                    "trial_number": trial_number,
                    "wav_file": filename,
                    "sample_rate": config.SAMPLE_RATE,
                    "chirp_low": config.CHIRP_LOW,
                    "chirp_high": config.CHIRP_HIGH,
                    "chirp_duration": config.CHIRP_DURATION,
                    "recording_duration": config.RECORDING_DURATION,
                    "expected_chirps": config.EXPECTED_CHIRPS,
                    "chirps_played": chirps_played,
                    "rms": features["rms"],
                    "raw_peak": features["peak"],
                    "speaker_mic_separation_cm": s,
                    "speed_of_sound": c,
                    "expected_bistatic_delay_ms": round(expected_ms, 6),
                    "reflector_description": geometry["reflector_description"],
                })

            print(f"  Saved: {filename} | chirps={chirps_played} "
                  f"rms={features['rms']:.6f}")
            time.sleep(0.5)

    print("\n" + "=" * 72)
    print("RECORDING COMPLETE")
    print("=" * 72)
    print(f"\nAudio: {config.AUDIO_ROOT}")
    print(f"Metadata: {config.METADATA_FILE}")
    print(f"Geometry: {config.GEOMETRY_FILE}")
    print("\nNext: run analyze_reflection_data.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nRecording failed: {e}")
        sys.exit(1)