---
name: Shinya Project Knowledge
description: Cataldo/Erener error-attribution self-touch adaptation study. Robotic self-reaching adaptation with two unimodal readouts (tactile localisation, visual reaching). Analysis in Shinya_Safak_20112025.Rmd.
type: project
originSessionId: b10b5509-e929-43ac-8d4e-3a7bc39a2891
---

## Overview

Postdoc project with Antonio Cataldo. Manuscript title (in prep): "Error attribution during somato-motor adaptation" (Cataldo*, Erener* et al.). Investigates whether prediction errors during self-touch sensorimotor adaptation are attributed to the tactile or motor modality.

Main analysis file: `C:/Users/erene/OneDrive/Desktop/Shinya_Safak_20112025.Rmd`.
Data: `TRY_Safak_Labelled_Data_04092025.xlsx`.
Pre-reg: `Safak_Preregistration_22092024.pdf` (Desktop).
Cataldo slides: `Cataldo_NTT_CINET_09092025.pdf` (Desktop).

Framework: extends Cataldo's prior work (Curr Biol 2022; Neuropsychologia 2021) on asymmetric bidirectional integration of motor and tactile signals in self-touch. Central question: what error signals lead to adaptation during self-touch, and how is the error attributed across modalities?

## Apparatus and task

Robotic leader-follower setup. Right hand holds leader robot, ballistic proximo-distal reaching movement toward tactile target delivered on unseen left forearm by third (target) robot. Follower robot on left arm delivers tactile feedback whose location is governed by the M:T gain. Vision of both arms occluded (mirror screen). Wrist splint, moulded left hand support, articulated right armrest, start point not on skin. Beeps sequence enforces ballistic movement.

Three tasks interleaved:

1. **Self-touch reaching (main adaptation task)**: reach with right hand to felt tactile target on left arm. Follower position depends on M:T gain.
2. **Tactile localisation readout** (attribution to left arm / tactile system): tactile target delivered, participant uses pedals to move an orange line to the felt position. This is the "Touch" DV.
3. **Visual reaching readout** (attribution to right arm / motor system): blue line appears in front of right hand, participant reaches with leader robot to align with the blue line (no vision of arm, proprioceptive). This is the "Movement" DV.

Readouts interleaved after every 7 trials of the main task.

## Design

Three blocks (M:T gains): 1.5:1 (proximally deviated tactile), 1:1 (control), 1:1.5 (distally deviated tactile). Order counterbalanced.

Each block: Pre (49 trials, M:T=1:1), Adaptation (98 trials, gradual gain change ~1 mm/trial over first 50 trials to keep implicit, then plateau), Post (49 trials, M:T=1:1). Total 196–197 trials/block in 7-trial cycles.

Phases used in current analysis:
- **Pre** = trialDist 43 (end of pre-baseline)
- **Late_Adap** = trialDist 141 (end of adaptation plateau)
- **First_Post** = trialDist 148 (immediate aftereffect)

N = 18 (multiple of 6 for full counterbalancing).

## DVs and preprocessing

Main task: reaching error (right-hand endpoint relative to felt tactile target).
Readouts: signed difference between target and response. In Rmd:
- `TRY_rightErr` = visual reaching = **Movement/motor** readout
- `TRY_leftErr` = tactile localisation = **Touch/tactile** readout
- `Master_Max_Error`, `Slave_Max_Error` = kinematic measures from main adaptation task

Preprocessing pipeline (`Corr_*` → `flip_Corr_*` → `ave_Flip_Corr_*`):
1. **Baseline correction** per (sbjN, blkN): subtract subject's mean Pre value from every trial. Pre becomes ~0 for every subject by construction.
2. **Sign-flip main1.5** so both 0.7 and 1.5 point in adaptation direction.
3. **Pool across gains**: within (sbjN, trialN), mean of flip_Corr across main0.7 and main1.5. Assign to main0.7 rows. main1.0 keeps raw non-flipped value. main1.5 set to NA.

Downstream analyses filter to `blkN == "main0.7"` = pooled-across-adapting-gains. main1.0 (control) excluded.

## Pre-registration

Original planned analysis:
- 3 (gain) x 3 (phase) rmANOVA on reaching errors per task
- One-sample t-tests of adaptation/post-adaptation slopes vs 0
- Regression between adaptation and readout slopes to integrate across gains
- Belief-updating model
- Explicitly anticipates alternative analyses "with the same hypotheses and overall inferential structure but different assumptions and power"

Current deviates by pooling gains via flip-and-average, running 2x2 rmANOVAs on Pre-vs-LateAdap and LateAdap-vs-FirstPost, and adding MANOVA + LDA on (Movement, Touch).

## Analyses in current Rmd

**Two 2x2 rmANOVAs** (`afex::aov_ez`), Phase x Readout (Movement, Touch), pooled main0.7:
- ANOVA 1: Pre vs Late_Adap. Touch sign-flipped.
- ANOVA 2: Late_Adap vs First_Post. Touch NOT sign-flipped.
- Each followed by 4 paired t-tests as simple-effect follow-ups.

**Two MANOVAs + LDAs** (added at professor's suggestion; MANOVA/LDA are same underlying math but different R functions expose different pieces):
- `car::Anova(lm(cbind(Movement, Touch) ~ phase + sbjN), type = 3, "Pillai")` for omnibus test.
- `MASS::lda(phase ~ Movement + Touch)` for canonical axis.
- Report: raw canonical coefs, standardized canonical coefs (raw × within-group SD), structure coefficients (cor with canonical variate), |Movement|/|Touch| ratio, subject-level bootstrap 95% CI on ratio (2000 reps, resample subjects).
- Touch flipped in contrast 1, unflipped in contrast 2 (matches ANOVA convention).

Sign-flip: MANOVA/LDA test statistics and coefficient magnitudes are sign-invariant (only Touch sign flips). ANOVA is NOT sign-invariant on Readout main effect or Phase x Readout interaction.

## Results

### Pre vs Late Adaptation (Touch flipped, pooled main0.7 + flipped main1.5)

- N = 18/18
- MANOVA: intercept Pillai 0.269, F(2,16)=2.94, p=0.082. Phase Pillai 0.343, F(2,16)=4.18, p=0.035. sbjN Pillai 0.861, F(34,34)=0.76, p=0.790.
- Raw canonical: Movement 0.779, Touch 0.124
- Standardized: Movement 0.966, Touch 0.254
- Structure: Movement r=0.972, Touch r=0.281
- |Movement|/|Touch|=3.81, bootstrap 95% CI [0.64, 71.5]

### Late Adaptation vs First Post (Touch unflipped)

- N = 18/18
- MANOVA: intercept Pillai 0.626, F(2,16)=13.39, p=0.0004. Phase Pillai 0.372, F(2,16)=4.74, p=0.024. sbjN Pillai 1.354, F(34,34)=2.10, p=0.017.
- Raw canonical: Movement 0.274, Touch −0.208
- Standardized: Movement 0.390, Touch −0.788
- Structure: Movement r=0.60, Touch r=−0.93
- |Movement|/|Touch|=0.495, bootstrap 95% CI [0.025, 6.79]

## Substantive interpretation (Cataldo's slide framing)

- **During adaptation** (implicit, gradual gain change): sensory error attributed to MOVEMENT only, driving implicit motor learning.
- **First post trial** (explicit/perceptual error at abrupt gain reversal): error attributed to BOTH touch and movement, ~2:1 favouring touch.
- Motor attribution present in both T<M and T>M gain conditions.
- Tactile attribution in first post strongly driven by T>M gain (larger/more salient tactile stimuli, consistent with "recalibration only when Touch>Movement" from earlier 8-experiment series in Cataldo 2022 Curr Biol).

MANOVA/LDA dissociation matches: motor (proprioceptive) readout carries adaptation buildup; tactile readout carries abrupt de-adaptation.

## Caveats

- **Single-trial snapshots** (trialDist 43, 141, 148). Noisy. Windowed sensitivity analysis worth doing.
- **Pooling collapses two gain conditions**; main1.0 control excluded from MANOVA.
- **Bootstrap CIs on ratio extremely wide** because Touch (contrast 1) or Movement (contrast 2) coef is small, blowing up denominator. Direction reliable, magnitude not. **Structure coefficients** (bounded [-1,1]) are the stable summary and tell same story - recommend leading with these.
- **Explicit vs implicit** framing is not clean; both transitions are implicit in standard motor-learning sense. Prefer "gradual buildup vs abrupt release" for precision.

## Comparison paper: Tsay et al. 2024 J Neurophysiol

- Tested chronically deafferented adults on clamped visuomotor rotation. Found implicit adaptation and perceived movement outcome minimally impacted by deafferentation, challenging proprioception-dependent models.
- Their perceptual re-alignment model: proprioception + vision as parallel inputs to multimodal percept of hand position.
- Prediction that deafferentation should amplify visual dominance and heighten perceived error was NOT borne out.
- Our data add within-somatosensation dissociation in intact participants: proprioceptive (motor) and tactile channels are not interchangeable; they dominate at different phases.

## Related Cataldo publications (background lit)

1. Cataldo et al., 2021 Neuropsychologia: disruption of normal self-touch correspondence
2. Cataldo et al., 2022 Curr Biol: interplay of sensorimotor signals in space perception; asymmetric bidirectional integration
3. Kong*, Cataldo* et al., 2022 Proc Roy Soc B: interhemispheric communication during self-touch
4. Cataldo*, Crivelli* et al., 2023 Proc Roy Soc B: active self-touch restores bodily self-awareness

In prep companion projects: Béres (model-learning), Felsenheimer (vision-supervised), Sellitto (unilateral stroke), Coelho (early/late blind).

## Workflow

User runs chunks interactively in RStudio and pastes errors here. Fix targeted issues; do not refactor unrelated code. R packages: ggplot2, tidyr, data.table, readxl, writexl, gridExtra, dplyr, afex, emmeans, clipr, MASS, car, boot.
