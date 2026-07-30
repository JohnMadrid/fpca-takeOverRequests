# Car-Frame Correction of Gaze and Head Direction — Problem, Fix, Verification

## The problem

Gaze (`EyeDirWorldCombined`) and head (`NoseVector`) were recorded as **unit
direction vectors in the world frame** (norm = 1.000, SD 0.0001). The analysis
used the `.x` and `.y` components as "horizontal" and "vertical". This is
incorrect: the world frame is a fixed compass, but the car drives along
different world directions in different events and **turns within the event
window**. Consequences:

1. **The "forward" axis flips between events.** Mean `NoseVector.x` was 0.84
   (Stag) but 0.46 / 0.40 (FallingRocks / Motorcyclist); mean `NoseVector.z`
   was the mirror image (0.27 / 0.85 / 0.90). The road points along world-x in
   Stag and world-z in the others, so "forward" lands on a different axis per
   event.

2. **Cosine compression where forward ≈ axis.** In Stag, `.x` sits at 0.84
   (near the unit-vector ceiling), where the cosine slope is 112°/unit vs the
   57° minimum. Small `.x` variance there hides ~14° of real angular movement.

3. **In-window car turn warps the trajectory.** The car turns **+66° within the
   Stag window** (HMD-derived heading). A driver looking steadily forward
   produces an `.x` trajectory that ramps 0.95 → 0.60 purely from the car
   turning — not from behaviour. Horizontal gaze leaks from world-z into world-x
   as the car rotates, so a constant gaze appears as a changing multi-channel
   pattern. This distorts the per-timepoint covariance and the FPCA/MFPCA
   eigenfunctions, **even within a single event**, in the post-onset window that
   contains the turn.

This explained the anomalous and event-inconsistent ED values (e.g. Stag
pre-ED ≈ 2.97, post-ED ≈ 8.19, opposite direction to the other events).

## Why this was not a data error

Same session, same tracker, never re-mounted: the hardware frame is identical
across events. The shift is therefore real — the **world frame faithfully
reports head/gaze direction in absolute coordinates**, and the car simply heads
in different world directions per event and turns mid-event. Confirmed: pre-onset
(self-drive) between-driver SD of `NoseVector.x` = 0.020 — every driver moves
together, i.e. the variation is the shared car heading, not individual behaviour.

## The fix

Express gaze and head **relative to the car**, per driver:

1. Derive each driver's car heading θ(t) from the **direction of their own HMD
   motion** in the x-z plane (`arctan2(dz, dx)` of `HmdPosition`).
2. Low-pass the HMD trajectory with a **Butterworth filter (0.8 Hz, order 3,
   zero-phase)** to remove crash spikes and head-jolt noise while keeping the
   slow road bend. (0.8 Hz chosen by cutoff sweep: 0.3–0.5 Hz over-smoothed the
   Motorcyclist heading; ≥0.8 Hz preserved its real ±10° structure.)
3. Rotate each driver's world (x, z) of gaze and head by −θ(t) →
   **lateral** (left/right relative to car). Keep `.y` (vertical) unchanged.
   Drop forward/z (it is ~constant, see T1) and the old world x/z.

HMD motion is dominated by car translation, not head movement (x-z step median
0.31 vs y wobble 0.029 = **10.8×**), and the HMD-derived heading matched the
mean-nose heading to **2.5°** — so HMD trajectory is a clean car-heading source.

New columns written to `data_segment/car_reference/car_reference_<event>.csv`:
`NoseVectorCar.x` (head lateral), `NoseVectorCar.y` (= `NoseVector.y`),
`EyeDirCar.x` (eye lateral), `EyeDirCar.y` (= `EyeDirWorldCombined.y`).

## Verification (on the saved per-driver-corrected files)

**T1 — forward becomes constant & event-consistent.** Car-frame forward mean =
0.985 / 0.980 / 0.986 across the three events (was world `.x` = 0.84 / 0.46 /
0.40). The event-dependence is removed; forward is ~1 everywhere, so it carries
no variance and is correctly dropped.

**T2 — lateral pre-onset ≈ 0, event-consistent.** Car-frame head-lateral pre-mean
= −0.004 / −0.062 / −0.043 (world `.x` pre was 0.96 / 0.35 / 0.39). Drivers look
forward (lateral ≈ 0) while passively cruising, in every event.

**T3 — lateral gaze now tracks steering (behavioural ground truth).**
Median per-driver correlation of eye-lateral with `SteeringInput`, post-onset:
car-frame = −0.24 / −0.44 / −0.67 (consistent across events); world `.x` was
0.07 / 0.30 / 0.43 (inconsistent, ~0 in Stag). The consistent sign means
"look where you steer" is now recovered in every event.

**T4 — directional ground truth (hazard side).** Stag and FallingRocks hazards
are on the right. Car-frame mean eye-lateral in the 1–2 s window: Stag = **+0.14**
(clear look-right, correctly timed). FallingRocks = −0.002 (null — consistent
with the rocks being static and visible before onset, so no post-onset saccade).

**T5 — geometry.** Unit norm preserved after rotation: 1.0000 (SD ≤ 0.0006). No
NaNs in any car-frame column.

**T6 — individual behaviour preserved.** Per-driver inspection (10 drivers/event):
pre-onset trajectories converge to baseline (shared heading removed); post-onset
they fan into distinct individual responses (not flattened to the mean).
Post-onset between-driver SD *increased* after de-rotation (head 0.059 → 0.093,
eye 0.092 → 0.144) — the correction revealed behavioural variance previously
masked by the shared turn, rather than removing it.

## Status / scope

- Car-reference files created and verified for Stag, FallingRocks, Motorcyclist.
- A preprocessing step performs this after event segmentation for future runs.
- All five notebooks have an unmissable per-cell banner on every cell that
  accesses world-frame `EyeDirWorldCombined.x/z` or `NoseVector.x/z`.
- MFPCA is being switched to the car-frame channels (head-lateral,
  head-vertical, eye-lateral, eye-vertical, steering). PCA / LDA pipelines to be
  migrated later.
