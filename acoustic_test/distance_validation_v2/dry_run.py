"""SARV distance validation v2 -- dry / configuration check.

This script validates the experiment infrastructure WITHOUT recording any
real data. It:

  1. verifies the audio devices are available,
  2. generates the chirp and reference,
  3. builds a synthetic recording with a simulated rigid-reflector echo
     at each distance (using the bistatic model),
  4. runs the full detection algorithm on the synthetic signal,
  5. prints the expected delays, search windows, and detection results.

If the infrastructure is correct, the synthetic reflector echo should be
detected at (or very near) the expected bistatic delay at every distance.

Usage:
    python dry_run.py
"""

import os
import sys

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

import config

# ------------------------------------------------------------------
# Signal chain (identical to the analysis tool)
# ------------------------------------------------------------------
def generate_reference_chirp():
    n = int(config.CHIRP_DURATION * config.SAMPLE_RATE)
    t = np.arange(n) / config.SAMPLE_RATE
    k = (config.CHIRP_HIGH - config.CHIRP_LOW) / config.CHIRP_DURATION
    phase = 2.0 * np.pi * (config.CHIRP_LOW * t + 0.5 * k * t * t)
    signal = np.sin(phase)
    signal *= np.hanning(n)
    signal = signal.astype(np.float64)
    norm = np.linalg.norm(signal)
    if norm > 0:
        signal /= norm
    return signal


def bandpass(signal):
    from scipy.signal import butter, sosfiltfilt
    nyquist = config.SAMPLE_RATE / 2.0
    low = config.FILTER_LOW / nyquist
    high = config.FILTER_HIGH / nyquist
    sos = butter(6, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def matched_filter(signal, reference):
    from scipy.signal import correlate
    reference = reference - np.mean(reference)
    signal = signal - np.mean(signal)
    corr = correlate(signal, reference, mode="valid", method="fft")
    ref_norm = np.linalg.norm(reference)
    ref_len = len(reference)
    squared = signal ** 2
    cum = np.concatenate([[0.0], np.cumsum(squared)])
    window_energy = cum[ref_len:] - cum[:-ref_len]
    denominator = np.sqrt(np.maximum(window_energy, 1e-18)) * ref_norm + 1e-12
    return corr / denominator


def refine_peak_subsample(values, index):
    index = int(index)
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    y1 = float(values[index - 1])
    y2 = float(values[index])
    y3 = float(values[index + 1])
    denom = y1 - 2.0 * y2 + y3
    if abs(denom) < 1e-12:
        return float(index)
    offset = 0.5 * (y1 - y3) / denom
    offset = np.clip(offset, -0.5, 0.5)
    return float(index + offset)


def search_window(reference_index, expected_delay_ms):
    sr = config.SAMPLE_RATE
    hw_samples = config.SEARCH_WINDOW_HALF_WIDTH_MS / 1000.0 * sr
    center = reference_index + expected_delay_ms / 1000.0 * sr
    win_start = int(round(center - hw_samples))
    win_end = int(round(center + hw_samples))
    # Exclude the direct speaker->mic path region.
    exclusion = config.direct_path_exclusion_ms()
    exclusion_samples = int(exclusion / 1000.0 * sr)
    win_start = max(win_start, reference_index + exclusion_samples)
    return win_start, win_end


def analyze_chirp(mf, reference_index, expected_delay_ms):
    sr = config.SAMPLE_RATE
    win_start, win_end = search_window(reference_index, expected_delay_ms)
    win_start = max(0, win_start)
    win_end = min(len(mf), win_end)
    if win_end <= win_start:
        return None
    region = np.abs(mf[win_start:win_end])
    local_idx = int(np.argmax(region))
    abs_idx = win_start + local_idx
    refined = refine_peak_subsample(np.abs(mf), abs_idx)
    delay_ms = (refined - reference_index) / sr * 1000.0
    strength = float(region[local_idx])
    return {
        "delay_ms": delay_ms,
        "strength": strength,
        "window_start_ms": (win_start - reference_index) / sr * 1000.0,
        "window_end_ms": (win_end - reference_index) / sr * 1000.0,
    }


# ------------------------------------------------------------------
# Synthetic recording with a simulated reflector echo
# ------------------------------------------------------------------
def make_synthetic_recording(dist_cm, sep_cm, speed):
    """Build a 2 s recording with 7 chirps and a simulated echo.

    The echo is a delayed, attenuated copy of the chirp, placed at the
    bistatic expected delay. This lets us verify the detection pipeline
    end-to-end without real hardware.
    """
    sr = config.SAMPLE_RATE
    total = int(config.RECORDING_DURATION * sr)
    audio = np.zeros(total, dtype=np.float64)

    # Playback chirp (same as recording tool)
    n = int(config.CHIRP_DURATION * sr)
    t = np.linspace(0, config.CHIRP_DURATION, n, endpoint=False)
    from scipy.signal import chirp as scipy_chirp
    play = scipy_chirp(t, f0=config.CHIRP_LOW, f1=config.CHIRP_HIGH,
                       t1=config.CHIRP_DURATION, method="linear")
    play *= np.hanning(n)
    play *= 0.25

    expected_ms = config.bistatic_delay_ms(dist_cm, sep_cm, speed)
    echo_delay_samples = int(expected_ms / 1000.0 * sr)
    # Direct speaker->mic path arrives at delay s/c.
    direct_delay_samples = int((sep_cm / 100.0) / speed * sr)

    for i in range(config.EXPECTED_CHIRPS):
        start = int((config.FIRST_CHIRP_TIME + i * config.CHIRP_INTERVAL) * sr)
        # Direct acoustic path (speaker->mic), weak coupling.
        direct_start = start + direct_delay_samples
        direct_end = direct_start + n
        if direct_end <= total:
            audio[direct_start:direct_end] += 0.02 * play
        # Reflector echo at bistatic delay. Use a clearly-dominant echo
        # so the dry run verifies the pipeline detects it; the real
        # experiment measures the actual echo strength vs distance.
        echo_start = start + echo_delay_samples
        echo_end = echo_start + n
        if echo_end <= total:
            audio[echo_start:echo_end] += 0.05 * play

    # small noise floor
    rng = np.random.default_rng(42)
    audio += 1e-4 * rng.standard_normal(total)
    return audio


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 72)
    print("SARV DISTANCE VALIDATION v2 -- DRY / CONFIGURATION CHECK")
    print("=" * 72)

    # 1. Devices
    print("\n[1] AUDIO DEVICES")
    if sd is None:
        print("  sounddevice not importable -- device check skipped.")
    else:
        try:
            devs = sd.query_devices()
            print(f"  Default input : {sd.default.device[0]}")
            print(f"  Default output: {sd.default.device[1]}")
            print(f"  Configured input device : {config.INPUT_DEVICE}")
            print(f"  Configured output device: {config.OUTPUT_DEVICE}")
            print("  (Verify these indices match your hardware before recording.)")
        except Exception as e:
            print(f"  WARNING: could not query devices: {e}")

    # 2. Geometry
    print("\n[2] GEOMETRY")
    sep = config.effective_separation_cm()
    speed = config.effective_speed_of_sound()
    if sep is None:
        print("  SPEAKER_MIC_SEPARATION_CM is NOT set.")
        print("  Dry run will use s = 15.0 cm for the synthetic check only.")
        sep = 15.0
    else:
        print(f"  Speaker/mic separation s = {sep} cm")
    print(f"  Speed of sound c = {speed} m/s")

    # 3. Expected delays + effective search windows
    exclusion = config.direct_path_exclusion_ms(sep, speed)
    print("\n[3] EXPECTED BISTATIC DELAYS (t = 2*sqrt(d^2+(s/2)^2)/c)")
    print(f"  Direct-path exclusion: {exclusion:.3f} ms "
          f"(direct path + ~1 ms main lobe + margin)")
    print(f"  {'dist':>5} {'expected':>10} {'effective window':>18} {'resolvable':>10}")
    expected = {}
    for d in config.DISTANCES_CM:
        ms = config.bistatic_delay_ms(d, sep, speed)
        expected[d] = ms
        lo = max(ms - config.SEARCH_WINDOW_HALF_WIDTH_MS, exclusion)
        hi = ms + config.SEARCH_WINDOW_HALF_WIDTH_MS
        resolvable = ms >= exclusion
        print(f"  {d:>4}cm {ms:>9.3f}ms [{lo:>7.3f}, {hi:>7.3f}] "
              f"{'yes' if resolvable else 'NO (below resolution)'}")

    # 4. Signal chain + synthetic detection
    print("\n[4] SYNTHETIC REFLECTOR DETECTION CHECK")
    ref = generate_reference_chirp()
    print(f"  Reference chirp: {len(ref)} samples, "
          f"{config.CHIRP_LOW}-{config.CHIRP_HIGH} Hz, "
          f"{config.CHIRP_DURATION*1000:.0f} ms")

    all_ok = True
    for d in config.DISTANCES_CM:
        audio = make_synthetic_recording(d, sep, speed)
        filtered = bandpass(audio)
        mf = matched_filter(filtered, ref)
        # analyze chirp 0 (first chirp) for the check.
        # The reference index is the chirp START time (where the direct
        # acoustic path peaks in the matched filter). v1 incorrectly used
        # sample_center - reference_length//2, which shifted all delays by
        # half the chirp length (~50 ms).
        reference_index = int(config.FIRST_CHIRP_TIME * config.SAMPLE_RATE)
        exp_ms = expected[d]
        if exp_ms < exclusion:
            print(f"  {d:>4}cm : expected {exp_ms:.3f}ms is BELOW the "
                  f"direct-path exclusion ({exclusion:.3f}ms) -- "
                  f"not resolvable with this chirp")
            continue
        res = analyze_chirp(mf, reference_index, exp_ms)
        if res is None:
            print(f"  {d:>4}cm : NO PEAK FOUND")
            all_ok = False
            continue
        err = res["delay_ms"] - exp_ms
        ok = abs(err) < config.DETECTION_TOLERANCE_MS
        all_ok = all_ok and ok
        print(f"  {d:>4}cm : measured={res['delay_ms']:.3f}ms "
              f"expected={exp_ms:.3f}ms err={err:+.3f}ms "
              f"strength={res['strength']:.4f} "
              f"{'OK' if ok else 'MISMATCH'}")

    # 5. Summary
    print("\n[5] RESULT")
    if all_ok:
        print("  Infrastructure OK: synthetic reflector detected at all "
              "resolvable distances within tolerance.")
    else:
        print("  WARNING: synthetic reflector NOT detected correctly. "
              "Check the signal chain before recording.")

    print("\n" + "=" * 72)
    print("DRY RUN COMPLETE -- no data was recorded.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nDry run failed: {e}")
        sys.exit(1)