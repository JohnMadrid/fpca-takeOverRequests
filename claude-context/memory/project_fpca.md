---
name: fpca-project-knowledge
description: "Crucial parameters, design choices, logic, results, and ongoing decisions for the fpca-takeOverRequests driving simulation analysis project"
metadata: 
  node_type: memory
  type: project
  originSessionId: 391ce050-8ead-4f72-a500-c1d510fad3e8
---

## Project overview

PhD-level driving simulation study. Participants completed driving events (FallingRocks, StagEvent, MotorcyclistEvent, etc.) under different experimental conditions. The repo is `C:/Users/willi/Desktop/fpca-takeOverRequests/`. There is no single main notebook; work spans several active top-level notebooks depending on the task: `Analysis_ED_PCA.ipynb`, `Analysis_ED_PCA_temp_16.6.ipynb`, `Analysis_ED_PCA.backup_premerge.ipynb`, `Analysis_ED_PCA_warnings.ipynb`, `Analysis_ED_PCA_warnings_09.02.26.ipynb`, `pca_lda_5var.ipynb`, `pca_lda_5var_temp.ipynb`, `preprocessing.ipynb`, `Descriptives_Warnings.ipynb`, `RT&crash_analysis.ipynb`, `ellipsoid_animation.ipynb`. The old `pca_lda.ipynb` is now in `trash/` (superseded). Pick/confirm the target notebook per task rather than assuming one. Data CSVs are in `data/cleaned_data/data_segment/outliers_removed_5features/steeringRemoved/`.

Variables of interest: `EyeDirWorldCombined.x` (horizontal gaze), `HeadRotation.x` (head), `SteeringInput`, `time_from_event`, `uid`, `SuccessfulCompletionState` (1=success/S, 0=fail/F), `ExperimentalCondition`.

**Why:** Investigates anticipatory visuomotor coordination under cognitive load. Success vs failure differences, and condition (load) effects on eye-head-steering coupling.

---

## Event descriptions and task characteristics

**StagEvent**: Hazard avoidance. The stag appears around the time of the warning. The visual warning outlines the stag, so participants see it faster. Warning therefore increases reaction time but also accelerates visual detection. Success depends on reaction quality.

**FallingRocks (FallenRocks)**: Static obstacle avoidance. Rocks are visible the entire approach. Not moving. Participant steers around them. Straightforward, visually guided.

**MotorcyclistEvent**: Very sudden hazard appearing after warning. Two motorcycles appear abruptly. Success may occur because the participant reacted very well and fast, OR because they were already out of the motorcycles' path by chance. Less dependent on reaction quality than the other events because of the speed of onset.

---

## Key analysis cells in pca_lda.ipynb

| Cell marker | What it does |
|---|---|
| `tlag_xcorr` (cell ~100) | Eye-head xcorr. One-sided window [0, 500ms], peak on positive xcorr only. |
| Cell 105 (`_xs_eye_col`) | Eye-steering xcorr. Two-sided +-2000ms, peak on absolute xcorr, mag>=0.05 filter. |
| Cell 106 (inserted) | Raw signal plots for xcorr inputs (eye + steering, z-scored). |

---

## Eye-steering xcorr: design choices

- **Normalization**: `/(N * sa * sb)` (biased, consistent with Matlab xcorr default)
- **Lag window**: primary +-1500ms; sensitivity at +-2000, +-1000, +-500ms. Rationale: literature reports lags up to ~1s, +-1500ms captures those with 500ms buffer without inviting noise from 2s range.
- **Peak**: argmax(|xcorr|); sign recorded separately
- **Magnitude filter**: `peak_mag >= 0.10` (primary); sensitivity at 0.05
- **Time window**: `[0.0, 4.0]` seconds post-event
- **Stratification**: S/F balanced within each condition for success comparisons; condition comparisons use full unbalanced data
- **Seeds tested**: [42, 123, 456, 789, 999]; plot seed = 999
- **ANCOVA**: additive `peak ~ C(success) + C(condition)` and interaction `~ C(success) * C(condition)` (OLS, type-II SS)
- **Each OLS in its own try/except** to prevent one failure silently zeroing all three p-values

---

## Eye-head xcorr: design choices

- **Window**: [0, 500ms] (one-sided, anticipatory direction only)
- **Peak**: on positive xcorr only
- **Plots**: 3-figure structure: raincloud S vs F, violin/strip by condition with FDR pairwise brackets, condition x outcome interaction lines

---

## Sensitivity tables (both xcorr cells)

Window sensitivity and seed sensitivity print **every p-value**: t-test, MWU, ANCOVA success, ANCOVA condition, ANCOVA interaction, Kruskal-Wallis. Split into two sub-tables: one for `peak_lag`, one for `peak_mag`. Format: `f'{p:10.4f}{"***"/"** "/"*  "/"   "}` left-aligned stars in fixed-width field to prevent column drift.

ANCOVA and KW use **full unstratified** data; t-test and MWU use **stratified** subset. In seed sensitivity, ANCOVA/KW are computed once on full data and are **identical across seeds** (noted in header).

---

## Plot structure (3-figure pattern used in both xcorr cells)

1. Mean xcorr curve + raincloud `peak_mag` (S vs F) + raincloud `peak_lag` (S vs F)
2. Violin/strip `peak_mag` and `peak_lag` by condition, with FDR-corrected pairwise MWU significance brackets
3. Condition x outcome interaction line plots for `peak_mag` and `peak_lag`

---

## Known results and interpretations

- **Condition effects on FallingRocks and Motorcyclist**: significant (different visual scene complexity changes tangent-point strategy). Expected from literature.
- **Null success effects**: consistent with Chattington 2007 (no individual differences in time lead across drivers). Wilson 2008 predicts coupling coefficient reduction under cognitive load; Figure 1 raincloud (peak_mag S vs F) is the most theoretically motivated panel.
- **Eye leads steering** at positive lags (anticipatory); negative lag = steering precedes eye (reactive).
- **Literature baseline**: anticipatory visuomotor cascade: eyes lead head lead steering ~0.75-1s.

---

## Bug fixes to remember

- **Mag interaction silent failure**: all three mag OLS p-values were inside a single try/except; if additive model failed, all became nan. Fix: each OLS in its own try/except with variables pre-initialized to np.nan.
- **Column drift in tables**: f-string with variable-width star suffix shifts columns. Fix: `f'{p:10.4f}{star:<3}'` with left-aligned fixed-width star field.
- **seed sensitivity revert**: a wide-table / two-table format was applied and then reverted (user request). Current format is two sub-tables (lag, then mag) with every p-value.

---

## Patch script convention

Changes to `pca_lda.ipynb` are applied via standalone Python patch scripts (e.g., `patch_xs_plots.py`). Each script loads the notebook JSON, modifies the target cell's source, and writes it back. Cell is found by searching for a unique string in its source, not by index.

---

## Files to know

- `pca_lda.ipynb` - main analysis notebook
- `data/cleaned_data/data_segment/outliers_removed_5features/steeringRemoved/` - CSVs
- `saved_results.joblib` - cached results
- `FIXES_SUMMARY.md` - historical fix log
- Memory: `C:/Users/willi/.claude/projects/C--Users-willi/memory/`
