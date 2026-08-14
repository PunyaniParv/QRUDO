# SARV Distance Validation v2 — Controlled Reflection Validation

**Purpose:** Determine whether a physical rigid reflector produces a
repeatable acoustic reflection that is distinguishable from the baseline,
using a physically-correct bistatic delay model.

This experiment does **not** modify the gesture-control pipeline, does not
train a classifier, and does not create a V9 gesture system. It is a
standalone controlled experiment under `acoustic_test/distance_validation_v2/`.

---

## 1. Background (why v2 exists)

The v1 investigation (`../distance_validation_v1/TECHNICAL_REPORT.md`)
concluded that the observed distance correlation (r = 0.9629) was likely an
artifact of peak-selection bias:

- The matched-filter response is dense with peaks (always 12/chirp, a cap).
- The baseline contains peaks at **every** expected delay.
- The algorithm always selected a peak near the expected delay, so the
  "measured delay" tracked the expected value regardless of physics.
- The tracked peak was usually weak (median rank 5–7.5), not the dominant
  echo.
- Peak strength did **not** decrease with distance (opposite of a real
  reflection).
- The monostatic `2d/c` model is physically wrong for a laptop with
  separated speaker and microphone.

v2 fixes these by using the bistatic model, a narrow physically-justified
search window, strongest-candidate selection, one detection per chirp, and
proper baseline subtraction.

---

## 2. Experiment design

### Distances
10, 20, 30, 40, 50 cm.

### Trials per distance
- 10 baseline recordings (no reflector)
- 10 reflector recordings (rigid flat reflector present)

Interleaved as `B, R, B, R, ...` (20 trials per distance) to cancel slow
environmental/driver drift.

### Total
20 trials × 5 distances = **100 recordings**.

---

## 3. Geometry

The laptop has **separated speaker and microphone** (bistatic geometry).
The speaker-to-microphone separation `s` must be **measured with a ruler**
and recorded. It is stored in `experiment_geometry.json` by the recording
tool.

### Bistatic expected-delay model

```
t = 2 * sqrt(d^2 + (s/2)^2) / c
```

- `d` = reflector distance from the laptop (perpendicular)
- `s` = measured speaker/microphone separation
- `c` = speed of sound (default 343.0 m/s)

The reflector is placed at perpendicular distance `d`, centered between the
speaker and microphone, face parallel to the laptop surface.

### Example expected delays (s = 15 cm, c = 343 m/s)

| Distance | Bistatic delay |
|----------|----------------|
| 10 cm    | 0.729 ms       |
| 20 cm    | 1.245 ms       |
| 30 cm    | 1.803 ms       |
| 40 cm    | 2.374 ms       |
| 50 cm    | 2.948 ms       |

(These are computed at runtime from the measured `s`.)

---

## 4. Reflector

Use a **rigid, flat reflector** — a book, clipboard, or similar rigid flat
object. The goal is to validate reflection physics before introducing
hand-shape variability. The reflector description is recorded in the
geometry metadata.

---

## 5. Detection algorithm (strict, not v1)

1. **Matched-filter response** — normalized matched filter of the
   bandpassed recording against the reference chirp.
2. **Physically-justified search window** — a narrow window around the
   bistatic expected delay (`SEARCH_WINDOW_HALF_WIDTH_MS = 1.5 ms`).
   Justification: chirp bandwidth 1 kHz → main lobe ~1 ms wide; distance
   measurement uncertainty ~±1 cm → ~±0.06 ms; centering error < 0.1 ms.
   This excludes the dense baseline response that dominated v1.
3. **Direct-path exclusion** — the window's lower bound is clamped to
   exclude the direct speaker→mic path (delay `s/c` plus its ~1 ms
   matched-filter main lobe plus a small margin). Echoes arriving within
   this region are not resolvable from the direct path.
4. **Strongest/global candidate** — the maximum of |matched-filter| within
   the window is selected (not arbitrary peaks near the expected delay).
5. **One detection per chirp** — exactly one candidate per chirp.
6. **Bistatic expected delay** — computed from measured `s`.
7. **Baseline response statistics** — mean and std of the baseline
   matched-filter response per distance.
8. **Baseline subtraction** — reflector strength minus baseline mean
   strength at the same delay.
9. **Residual peak amplitude** near the expected delay.
10. **Global maximum** of the whole response and its delay.
11. **Peak rank/strength/delay** recorded for every chirp.

### Resolution limit (important)

With the 1 kHz-bandwidth chirp (7.5–8.5 kHz), the matched-filter main lobe
is ~1 ms wide. Combined with the direct-path delay (`s/c`), the direct-path
exclusion is ~1.5 ms (for s = 15 cm). This means:

- **10 cm** (expected ~0.73 ms) and **20 cm** (expected ~1.25 ms) are
  **below the resolution limit** — their echoes overlap the direct path
  and cannot be resolved with this chirp.
- **30–50 cm** (expected ~1.8–2.9 ms) are resolvable.

The dry-run check confirms this: the synthetic reflector is detected at
30/40/50 cm but not at 10/20 cm. The experiment still records all five
distances; the analysis reports the raw measurements and the validation
logic decides whether the reflector is validated. If 10/20 cm cannot be
resolved, the validation will report `ACOUSTIC REFLECTION NOT VALIDATED`
for the full 10–50 cm range — an honest physical result, not a tuned pass.

---

## 6. Output metrics (machine-readable)

Per-chirp CSV (`reflection_chirp_report.csv`):

- `recording_id`
- `condition` (baseline | reflector)
- `distance_cm`
- `chirp_number`
- `expected_delay_ms` (bistatic)
- `measured_delay_ms`
- `delay_error_ms`
- `peak_strength`
- `baseline_strength`
- `baseline_subtracted_strength`
- `peak_rank`
- `global_max_strength`
- `global_max_delay_ms`
- `residual_peak_strength`
- `residual_peak_delay_ms`
- `detection_status`
- `final_detection`

Also produced:

- `reflection_recording_summary.csv` — per-recording aggregates
- `reflection_distance_metrics.csv` — per-distance metrics + t-tests
- `validation_summary.json` — validation decision and criteria

---

## 7. Plots (7)

1. Expected vs measured delay
2. Delay error vs distance
3. Baseline vs reflector matched-filter response
4. Baseline-subtracted response
5. Reflection strength vs distance
6. Detection consistency across chirps
7. Peak-delay distribution

---

## 8. Validation logic

The system is **VALIDATED** only if the evidence collectively supports:

1. a **positive baseline-subtracted peak** at the expected delay
   (detection rate ≥ 0.70);
2. the reflector peak is **consistently stronger than baseline**
   (t-test p < 0.05 at every distance, mean subtraction > 0);
3. the reflector peak is **among the strongest/global peaks**
   (mean rank ≤ 3.0);
4. **strength decreases with distance** (Pearson r ≤ −0.5);
5. **delay RMSE < 0.15 ms**;
6. the **baseline does not show the same reflector-specific response**
   (baseline detection rate ≤ 0.30).

If any criterion fails, the result is reported as:

```
ACOUSTIC REFLECTION NOT VALIDATED
```

Raw measurements are always reported. Thresholds are initial criteria and
are **not** tuned to force a pass.

---

## 9. Files

| File | Purpose |
|------|---------|
| `config.py` | Shared configuration (single source of truth) |
| `dry_run.py` | Dry/configuration check (no recording) |
| `record_reflection_data.py` | Recording tool (interleaved B/R) |
| `analyze_reflection_data.py` | Analysis + validation + plots |
| `EXPERIMENT_PROTOCOL.md` | This document |

---

## 10. Commands

```bash
# 1. Dry / configuration check (no data recorded)
python acoustic_test/distance_validation_v2/dry_run.py

# 2. Record the 100-recording dataset (only after approval)
python acoustic_test/distance_validation_v2/record_reflection_data.py

# 3. Analyze + validate
python acoustic_test/distance_validation_v2/analyze_reflection_data.py
```

---

## 11. Physical setup required

- Laptop with working speaker + microphone (verify device indices in
  `config.py`).
- A ruler.
- A rigid flat reflector (book, clipboard, etc.).
- A quiet room; keep the laptop and your body still.