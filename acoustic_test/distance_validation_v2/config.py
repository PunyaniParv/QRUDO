"""SARV distance validation v2 -- shared configuration.

This module is the single source of truth for the controlled reflection
experiment. Both the recording tool and the analysis tool import from
here so that recording and analysis always use identical settings.

The v1 investigation (see ../distance_validation_v1/TECHNICAL_REPORT.md)
concluded that the observed distance correlation was likely dominated by
peak-selection bias. v2 therefore:

  * uses the physically-correct BISTATIC delay model
    t = 2 * sqrt(d^2 + (s/2)^2) / c   (s = measured speaker/mic separation)
  * searches only within a narrow, physically-justified window around the
    expected delay (NOT the full 0.6-18 ms response)
  * selects the STRONGEST candidate in that window (not arbitrary peaks)
  * enforces ONE detection per chirp
  * performs proper baseline subtraction with baseline statistics
"""

import os
import json

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# Audio / signal chain (preserved from v1 for comparability)
# ------------------------------------------------------------------
SAMPLE_RATE = 44100
CHIRP_LOW = 7500
CHIRP_HIGH = 8500
CHIRP_DURATION = 0.100
RECORDING_DURATION = 2.0
FIRST_CHIRP_TIME = 0.15
CHIRP_INTERVAL = 0.250
EXPECTED_CHIRPS = 7

# Audio devices (verify with device_test.py / sounddevice)
INPUT_DEVICE = 2
OUTPUT_DEVICE = 4

# Analysis bandpass (preserved from v1)
FILTER_LOW = 6000
FILTER_HIGH = 11000

# ------------------------------------------------------------------
# Physics
# ------------------------------------------------------------------
SPEED_OF_SOUND = 343.0  # m/s at ~20 C

# ------------------------------------------------------------------
# Geometry -- MUST be measured and set by the user before recording.
# s = speaker-to-microphone separation in cm.
# The recording tool prompts for this and stores it in
# experiment_geometry.json; the analysis tool reads it from there.
# ------------------------------------------------------------------
SPEAKER_MIC_SEPARATION_CM = None  # <-- set this (or via recording tool)

# ------------------------------------------------------------------
# Experiment design
# ------------------------------------------------------------------
DISTANCES_CM = [10, 20, 30, 40, 50]
TRIALS_PER_CONDITION = 10  # 10 baseline + 10 reflector per distance

# ------------------------------------------------------------------
# Detection algorithm
# ------------------------------------------------------------------
# Physically-justified search window half-width around the bistatic
# expected delay. Justification:
#   - chirp bandwidth 1 kHz -> matched-filter main lobe ~1 ms wide
#   - distance measurement uncertainty ~+/-1 cm -> ~+/-0.06 ms
#   - reflector centering error -> < 0.1 ms
# A half-width of 1.5 ms captures the reflector peak while excluding
# the dense baseline response that dominated v1.
SEARCH_WINDOW_HALF_WIDTH_MS = 1.5

# Margin added beyond the direct-path main lobe when excluding the direct
# speaker->mic path from the search window.
DIRECT_PATH_MARGIN_MS = 0.1

# Minimum separation between distinct peaks in the full response
PEAK_DISTANCE_MS = 0.25

# Relative threshold for global peak extraction (full response)
PEAK_RELATIVE_THRESHOLD = 0.18

# Full echo search range (for global peak rank)
ECHO_MIN_DELAY_MS = 0.60
ECHO_MAX_DELAY_MS = 18.0

# Delay-error tolerance for a "detection" (one per chirp)
DETECTION_TOLERANCE_MS = 0.50

# ------------------------------------------------------------------
# Validation decision thresholds (initial criteria -- reported raw too)
# ------------------------------------------------------------------
VALIDATION = {
    # fraction of reflector chirps with a positive baseline-subtracted
    # peak near the expected delay
    "min_detection_rate": 0.70,
    # delay RMSE (measured vs bistatic expected) in ms
    "max_delay_rmse_ms": 0.15,
    # mean peak rank of the reflector candidate (1 = global strongest)
    "max_mean_rank": 3.0,
    # Pearson correlation between reflector strength and distance
    # (must be negative: strength decreases with distance)
    "strength_distance_corr_threshold": -0.5,
    # baseline must NOT show the same reflector-specific response
    "baseline_max_detection_rate": 0.30,
    # significance level for reflector-vs-baseline strength test
    "ttest_alpha": 0.05,
}

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
AUDIO_ROOT = os.path.join(BASE_DIR, "audio")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")
GEOMETRY_FILE = os.path.join(BASE_DIR, "experiment_geometry.json")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")
RESULTS_DIR = os.path.join(ANALYSIS_DIR, "results")


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def load_geometry():
    """Load geometry from experiment_geometry.json if present."""
    if os.path.exists(GEOMETRY_FILE):
        try:
            with open(GEOMETRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "speaker_mic_separation_cm": SPEAKER_MIC_SEPARATION_CM,
        "speed_of_sound": SPEED_OF_SOUND,
        "reflector_description": "",
    }


def effective_separation_cm():
    geo = load_geometry()
    return geo.get("speaker_mic_separation_cm", SPEAKER_MIC_SEPARATION_CM)


def effective_speed_of_sound():
    geo = load_geometry()
    return geo.get("speed_of_sound", SPEED_OF_SOUND)


def bistatic_delay_ms(dist_cm, sep_cm=None, speed=None):
    """Bistatic expected delay: t = 2*sqrt(d^2 + (s/2)^2)/c.

    d = reflector distance from laptop (cm)
    s = speaker-microphone separation (cm)
    Returns delay in ms.
    """
    if sep_cm is None:
        sep_cm = effective_separation_cm()
    if speed is None:
        speed = effective_speed_of_sound()
    if sep_cm is None:
        raise ValueError(
            "SPEAKER_MIC_SEPARATION_CM is not set. "
            "Run the recording tool to measure and store the geometry, "
            "or set it in config.py."
        )
    d = dist_cm / 100.0
    s = sep_cm / 100.0
    path = 2.0 * np.sqrt(d * d + (s / 2.0) ** 2)
    return path / speed * 1000.0


def direct_path_exclusion_ms(sep_cm=None, speed=None):
    """Delay (ms) below which the direct speaker->mic path dominates.

    The direct path arrives at the microphone at delay s/c and its
    matched-filter main lobe is ~1/bandwidth wide (1 kHz -> ~1 ms).
    Echoes arriving within this region are not resolvable from the
    direct path, so the search window is clamped to start after it.

    Returns the exclusion delay in ms.
    """
    if sep_cm is None:
        sep_cm = effective_separation_cm()
    if speed is None:
        speed = effective_speed_of_sound()
    if sep_cm is None:
        return 1.5
    direct_delay = (sep_cm / 100.0) / speed * 1000.0
    main_lobe_half = 1.0 / (CHIRP_HIGH - CHIRP_LOW) * 1000.0
    return direct_delay + main_lobe_half + DIRECT_PATH_MARGIN_MS
