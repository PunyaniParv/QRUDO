# SARV Distance Validation v1 — Technical Investigation Report

**Date:** 2026-08-14
**Scope:** Analysis of existing `distance_validation_v1` data only. No new data collected, no V9 created, no classifier trained, no hardware changed.

---

## 1. Executive Summary

The measured delay-vs-distance correlation (r = 0.9629) is **real but almost certainly an artifact of the peak-selection algorithm**, not evidence of a physical hand reflection. The analysis pipeline always selects a peak near the expected delay because the matched-filter response is dense with candidate peaks (always exactly 12 per chirp, the cap), and the baseline contains peaks at **every** expected delay. A stricter, physically-justified peak-selection method reduces the correlation to r ≈ 0.95 with much higher error (RMSE 0.28 ms vs 0.078 ms), and the tracked peak is **not** the same physical component across distances.

---

## 2. What Is Definitely Real

### 2.1 The measured delay does increase with distance
Per-recording median delay vs distance:
| Distance | Expected (2d/c) | Measured median | Measured mean | Std |
|----------|----------------|-----------------|---------------|-----|
| 10 cm | 0.583 ms | 0.797 ms | 0.807 ms | 0.090 |
| 20 cm | 1.166 ms | 1.101 ms | 1.093 ms | 0.124 |
| 30 cm | 1.749 ms | 1.759 ms | 1.687 ms | 0.279 |
| 40 cm | 2.332 ms | 2.220 ms | 2.346 ms | 0.211 |
| 50 cm | 2.915 ms | 2.852 ms | 2.816 ms | 0.199 |

- Per-recording correlation (distance vs median delay): **r = 0.9629** (n=47)
- Per-condition median correlation: **r = 0.9945**
- Monostatic linear fit: slope = 0.0523 ms/cm (ideal = 0.0583), intercept = 0.177 ms, RMSE = 0.078 ms

**This correlation is statistically real** — it is not noise. The question is whether it reflects a physical hand reflection or an algorithmic selection bias.

### 2.2 The matched-filter response is dense with peaks
- Every chirp produces **exactly 12 candidate peaks** (the `MAX_CANDIDATE_PEAKS` cap). This is true for **all 427 chirps** across all conditions.
- The search window is 0.6–18.0 ms (17.4 ms wide). With 0.25 ms minimum peak separation, up to 70 peaks are theoretically possible. `find_peaks` always finds ≥12 peaks above the 18% relative threshold.
- **Conclusion:** The matched-filter output is not a clean impulse response with a few distinct echoes. It is a dense, noisy response with many comparable peaks. The "12 peaks" is a truncation artifact, not a physical count.

### 2.3 The baseline contains peaks at every expected delay
| Expected delay | Baseline peaks within 0.5 ms | Baseline peaks within 1.0 ms |
|----------------|------------------------------|------------------------------|
| 0.583 ms (10cm) | 33 | 64 |
| 1.166 ms (20cm) | 61 | 96 |
| 1.749 ms (30cm) | 57 | 116 |
| 2.332 ms (40cm) | 53 | 107 |
| 2.915 ms (50cm) | 49 | 101 |

**Every unique delay found in hand conditions also appears in the baseline** (10/10, 17/17, 11/11, 15/15, 19/19 within 0.25 ms). The baseline is not "clean" — it has peaks everywhere.

### 2.4 The full candidate-peak distributions are nearly identical between baseline and hand
Histogram overlap (0.25 ms bins, 0–18 ms):
| Condition | Overlap with baseline |
|-----------|----------------------|
| 10 cm | 0.897 |
| 20 cm | 0.878 |
| 30 cm | 0.847 |
| 40 cm | 0.887 |
| 50 cm | 0.872 |

The overall delay distribution barely changes when the hand is present. This is **not** what a strong hand reflection would produce.

---

## 3. What Is Likely an Artifact

### 3.1 The 0.9629 correlation is inflated by the peak-selection method
The original analysis computes `median_delay_ms` from **all** peaks within ±0.5 ms of the expected delay, across all chirps. Because:
1. The response is dense (12 peaks/chirp),
2. The baseline has peaks at every expected delay,
3. The tolerance window (±0.5 ms) is wide relative to the peak spacing (0.25 ms),

...the algorithm **always finds a peak near the expected delay** and reports its delay. The "measured delay" is essentially "the delay of whatever peak happens to be closest to the expected value," which naturally tracks the expected value as distance changes.

### 3.2 The tracked peak is NOT the same physical component
The strongest peak near the expected delay has a **median rank of 5–7.5** among the 12 candidates (rank 1 = strongest overall):
| Condition | Rank distribution | Median rank |
|-----------|-------------------|-------------|
| 10 cm | 5×rank1, rest rank 6–12 | 7.5 |
| 20 cm | 2×rank1, rest rank 2–11 | 5.0 |
| 30 cm | 3×rank1, rest rank 5–12 | 7.0 |
| 40 cm | 3×rank1, rest rank 2–12 | 7.0 |
| 50 cm | 3×rank1, rest rank 2–12 | 7.0 |

The "selected" peak is usually a **weak, low-ranked peak**, not the dominant reflection. The algorithm is not tracking a consistent physical echo — it is picking whichever weak peak happens to fall near the expected delay.

### 3.3 The peak strength does NOT decrease with distance
If the hand reflection were real, its strength should decrease with distance (inverse-square law). Instead:
| Condition | Mean strength of tracked peak |
|-----------|------------------------------|
| 10 cm | 0.0500 |
| 20 cm | 0.0591 |
| 30 cm | 0.0623 |
| 40 cm | 0.0615 |
| 50 cm | 0.0650 |

Strength **increases** slightly with distance — the opposite of a physical reflection. This is consistent with the algorithm selecting different (random) peaks at different distances.

### 3.4 Baseline subtraction shows no consistent hand peak
The strongest baseline-subtracted peak (|hand| − |baseline|) near the expected delay is **not at the expected delay**:
| Condition | Expected | Strongest diff peak | Value |
|-----------|----------|---------------------|-------|
| 10 cm | 0.583 ms | 0.000 ms | 0.011 |
| 20 cm | 1.166 ms | 0.272 ms | 0.024 |
| 30 cm | 1.749 ms | 2.585 ms | 0.041 |
| 40 cm | 2.332 ms | 1.361 ms | 0.034 |
| 50 cm | 2.915 ms | 2.109 ms | 0.030 |

There is no consistent positive peak at the expected delay that appears only when the hand is present.

### 3.5 The `near_expected > 7` bug
The analysis counts **all** peaks within the tolerance window across **all** chirps, not one per chirp. Multiple peaks per chirp can fall within ±0.5 ms, so `near_expected` can exceed 7 (the number of chirps). 8 recordings show this (e.g., 20cm_001: 10/7, 30cm_009: 11/7, 50cm_002: 10/7). The `detection_rate` can therefore exceed 1.0, which is logically impossible for one-detection-per-chirp. This inflates the apparent detection rate.

### 3.6 The 2d/c model is not the right physical model
The laptop has **separated speaker and microphone**. The actual acoustic path is **bistatic**: speaker → hand → microphone, not a monostatic round-trip.

- Monostatic: delay = 2d/c
- Bistatic (hand centered between speaker & mic, separation s): delay = 2·√(d² + (s/2)²)/c

For a typical laptop (s ≈ 10–15 cm), the bistatic delay is **larger** than monostatic, especially at short distances:
| Distance | Mono | Bistatic s=10cm | Bistatic s=15cm |
|----------|------|-----------------|-----------------|
| 10 cm | 0.583 | 0.652 | 0.729 |
| 20 cm | 1.166 | 1.202 | 1.245 |
| 30 cm | 1.749 | 1.773 | 1.803 |
| 50 cm | 2.915 | 2.930 | 2.948 |

The measured data has a **positive intercept** (0.177 ms) and a **slope below ideal** (0.0523 vs 0.0583 ms/cm). A bistatic fit with s = 11.3 cm gives RMSE = 0.104 ms (worse than monostatic+offset at 0.078 ms). A bistatic+offset fit gives s = 25.9 cm (unrealistically large for a laptop) with RMSE = 0.055 ms.

**Conclusion:** The 2d/c model is physically wrong for this geometry, but the bistatic correction alone does not explain the data. The systematic offset and reduced slope suggest the measured delays are not a clean physical path at all.

---

## 4. What Cannot Currently Be Determined

1. **Whether a hand reflection exists at all.** The data cannot distinguish a weak hand reflection from the dense baseline response. The baseline has peaks at every expected delay, so the presence of a peak near the expected delay is not evidence of a hand reflection.

2. **The actual speaker/microphone separation.** The recording script does not record the laptop model or the speaker/mic positions. The bistatic fit suggests s ≈ 11–26 cm, but this is unconstrained and inconsistent.

3. **Whether the hand reflection, if it exists, is strong enough to be useful.** The matched-filter SNR near the expected delay is only 9–14 dB, and the tracked peak is usually rank 5–12 (weak). Even if a hand reflection exists, it may be too weak to detect reliably.

4. **The exact cause of the systematic delay offset.** The measured delays are offset from both monostatic and bistatic models. This could be due to:
   - Speaker/mic driver latency (fixed offset)
   - The chirp reference alignment (the `expected_index` uses `sample_center - reference_length//2`, which may not align to the actual chirp start)
   - The hand not being exactly at the measured distance
   - The hand not being centered between speaker and mic

---

## 5. Minimum Next Controlled Experiment

To conclusively validate or reject the hand reflection, the following controlled experiment is required. **This is a proposal only — do not implement until approved.**

### Protocol: "Hand-Present vs Hand-Absent at Fixed Distance, with Baseline Subtraction"

**Goal:** Determine whether placing a hand at a known distance produces a **repeatable, distance-dependent peak** that is **absent** in the baseline, using a physically-correct bistatic model.

**Setup:**
1. **Fix the geometry.** Measure and record the actual speaker and microphone positions on the laptop (distance between them, s). Place the hand at a known perpendicular distance d from the laptop surface, centered between speaker and mic.
2. **Use a rigid reflector.** Replace the hand with a flat rigid board (e.g., a book or clipboard) of known size, to eliminate hand-shape variability. This isolates the reflection physics from hand geometry.
3. **Record interleaved baseline and hand trials.** For each distance (10, 20, 30, 40, 50 cm), record:
   - 10 baseline trials (no reflector)
   - 10 hand trials (reflector present)
   Interleave them (B, H, B, H, ...) to cancel slow drift.

**Analysis (strict, physically-justified):**
1. **Use the bistatic model.** Expected delay = 2·√(d² + (s/2)²)/c, with s measured, not assumed.
2. **Baseline subtraction.** Compute the average baseline matched-filter response and subtract it from each hand trial. Look for a **positive residual peak** at the bistatic expected delay.
3. **One detection per chirp.** Select only the **strongest** peak per chirp (rank 1), not all peaks within tolerance.
4. **Require the peak to be the global maximum** in the search window, not just any peak near expected.
5. **Check strength vs distance.** A real reflection must show strength decreasing with distance (inverse-square). If strength is flat or increasing, reject.
6. **Check delay consistency.** The tracked peak delay must be consistent across chirps (low std) and match the bistatic model within the chirp bandwidth resolution (~0.1 ms).

**Decision criteria:**
- **Validated** if: (a) a positive baseline-subtracted peak exists at the bistatic expected delay, (b) it is the strongest peak in the window, (c) its strength decreases with distance, (d) its delay matches the bistatic model with RMSE < 0.15 ms, and (e) it is absent in baseline.
- **Rejected** if: the baseline-subtracted peak is not at the expected delay, or strength does not decrease with distance, or the peak is not consistently the strongest.

**Minimum data:** 10 baseline + 10 hand trials × 5 distances = 100 recordings, ~15 minutes.

---

## 6. Summary Table

| Question | Answer |
|----------|--------|
| Why exactly 12 peaks/chirp? | `MAX_CANDIDATE_PEAKS=12` cap always hit; response is dense with ≥12 peaks above 18% threshold. Artifact. |
| Why near_expected > 7? | Code counts all peaks within ±0.5 ms across all chirps, not one per chirp. Bug. |
| Same physical peak tracked? | No. Median rank 5–7.5; algorithm switches between weak peaks. |
| Baseline vs hand distributions? | Nearly identical (85–90% overlap). Baseline has peaks at every expected delay. |
| Raw WAV/correlation around expected? | Dense, noisy response; no clean echo at expected delay. SNR 9–14 dB. |
| 2d/c model valid? | No. Bistatic path is physically correct but doesn't fully explain data. |
| Does 0.9629 survive stricter selection? | No. Drops to r ≈ 0.95 with RMSE 0.28 ms (vs 0.078 ms). |
| Is the hand reflection real? | **Cannot be determined from current data.** Likely an artifact of peak-selection bias. |

---

## 7. Files Created During Investigation

- `investigate_distance_data.py` — main investigation script
- `deep_peak_analysis.py` — peak identity & baseline subtraction analysis
- `compute_correlations.py` — correlation computation
- `investigation/` — output plots (correlation response, full MF response, baseline-subtracted response)