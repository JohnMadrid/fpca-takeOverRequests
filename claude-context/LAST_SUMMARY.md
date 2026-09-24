# Last session handoff

**Date:** 2026-09-24
**Note:** This shutdown was run WITHOUT compaction (user was about to update Claude Code for VS Code and just wanted progress saved). So this summary is written from the live session, not from a /compact digest.

## What we did this session (decisions + why)

### FPCA supplement to the ED/PCA paper (main technical work)
- Confirmed and used "Route A" MFPCA = the Happ & Greven (2016) two-step. Proved it is identical to concatenate-standardised-channels-and-PCA (Prop. 5: keeping all univariate components is an orthonormal rotation, eigenvalues invariant). The receipt (max eigenvalue diff ~1e-13) is printed by the code. This is the user's "I told you so" card for the professor (they proposed the concatenation idea 2 years ago and the supervisor disliked it).
- Moved everything into the notebook (user wanted it all in one place, not in fpca_stats.py). `fpca_stats.mfpca_routeA` is now redundant but still present.
- Added TWO new subsections at the end of the notebook's MFPCA section (before the `# PCA` markdown):
  1. "BETWEEN-DRIVER FUNCTIONAL ED (MFPCA Route A two-step)" - self-contained, GOLD footing (car frame, Operation C + pooled per-channel SD), keeps all univariate components, prints functional ED + Route A=B receipt, plots top-3 eigenfunctions + temporal energy profiles + a funcPR(t) vs snapshot ED overlay. Channels relabelled head x/head y/eye x/eye y/steer. "energy" label kept (user reverted my "variance" change).
  2. "DECODING PERFORMANCE FROM THE COVARIANCE OF THE TRAJECTORY" - top-5 MFPCA scores -> 5-fold CV LDA -> success/failure AUC + 2000-perm label-shuffle null, per event, self-contained numpy LDA.
- KEY RESULT (verified on real data): trajectory-shape decoding of success. Stag AUC 0.92 (p<.001), Motorcyclist 0.75 (p<.001), Falling rocks 0.57 (p=.011). The gradient (strong -> moderate -> near-chance) is the interesting finding, not the bare decodability. All three are essentially ONE-dimensional: mode 1 alone gives the whole effect (cumulative CV AUC flat as K grows; per-mode folded AUC concentrated in mode 1). Timing: the discriminant weight (d(t)=sum w_m psi_m(t)) is late, 3.4-3.9s, i.e. decided during action, not ahead of onset.
- Interpretation agreed: decodability measures whether a task funnels behaviour toward one shared low-dimensional solution. Stag funnels hard, Motorcyclist weakly, Falling rocks not at all (failures idiosyncratic, no shared signature). This is the "fills the space vs collapses onto a manifold" question answered per task.
- Decision deferred to Friday meeting: is FPCA "candy" (short coda to this paper) or a second paper. Findings written to work either way. NO MFPCA content on slides yet.

### Notebook cell 191 (cos2 during TFCE-agreement) - made plot-only
- Added a load-or-compute cache guard: first run (with the TFCE globals _AX/_AUACC/_T1_SIG/_TA_SIG alive) saves derived arrays to `cos2_tfce_plotcache.joblib`; every later run loads it and only plots. Lets the user restart the kernel and re-plot with NO TFCE recompute. `_REBUILD=True` forces refresh. USER MUST RUN IT ONCE while the TFCE globals are still in the kernel to seed the cache.
- Changed the cross-event (2/3-agreement) plot x-axis to label "TFCE agreement region" with ticks ">=2/3 agreement (late action)" and "<2 levels".

### Slides (extensive wording work; user copies verbatim)
- Established a TWO-REGISTER wording convention: "predict" = conceptual, "decode" = data. Do not use "separate" (user dislikes it). Nouns: "decodability" (readout) vs "separability" (geometry).
- "ahead of time" is only legitimate relative to the CRASH marker, not onset. Anchor any temporal-lead flex to mean-crash-timing; otherwise drop to "decode" with no temporal claim.
- "low-dimensional collapse" reworded to result-focused "collapse onto a low-dimensional manifold" (matches the yellow recap box).
- Important accuracy split: the MANIFOLD is low-dimensional (not single); the SUCCESS readout is one-dimensional (single dominant axis / PC1).
- Recap (final) slide reviewed; proposed register fixes, spelling consistency (UK behaviour/manoeuvre), and box grammar. Final yellow box lines: "Complex, controlled sensorimotor action collapses onto a surprisingly low-dimensional manifold. That manifold even predicts the outcome, decodable from a single dominant axis."
- On punctuation: user avoids em dashes, colons and semicolons on slides (reads as AI). Comma splices are fine on slides; or use a trailing modifier clause.

### Stats tutoring - 3-phase contrasts
- Of the 4 contrasts (Rise, Collapse, Peak, Decline) only 2 are independent: three phase means carry 2 shape degrees of freedom. Peak (quadratic/hump) and Decline (linear) are the orthogonal pair; Rise and Collapse are recombinations. Peak = Rise + Collapse, Decline = Rise - Collapse; so Collapse = (Peak - Decline)/2. Correct over 2 comparisons per scenario, not 4 (this leaves all stars intact except the marginal "+").
- Confirmed user's contrast definitions are valid, but flagged a SIGN inconsistency: they wrote collapse = Action - TOR (negative during a fall) while rise = TOR - Baseline (positive during a rise). Recommended collapse = TOR - Action for sign-consistency. Their Peak = TOR - (Base+Action)/2 is the standard quadratic contrast at half scale (fine).

### Concept figure (Desktop/epochs_concept.png)
- Three epoch marionettes (profile driver, sketch style): Passive (grey wandering gaze, slack strings), Take-over (orange head+gaze scan fan + arc centred on the eye), Active control (blue: gaze+steering FAN across a lateral road-plane perspective line at eye level, label riding on the line). Fixed: wheel spokes were drawn to double the rim radius (protruding) -> now 0.025/0.075 half-axes. Script: scratchpad/concept_epochs.py.

### Housekeeping
- Found and saved the /shutdown and /start procedure to memory (reference_session_sync.md + MEMORY.md pointer). It was not in memory before, hence I could not recall it earlier.
- Software-used answer for the talk: Python (matplotlib for figures/animations, numpy/scipy/scikit-learn for analysis, Jupyter). Not "AI".

## Current state (done / mid-flight / verified)
- Notebook edits INSERTED and syntax-checked (ast.parse), but NOT run in the notebook kernel. The user runs them. So the two new MFPCA subsections and the cell-191 cache are on disk, unexecuted.
- Decoding AUCs, functional ED, per-mode/cumulative AUC, timing: VERIFIED by running standalone scripts on the real car-frame CSVs (python has pandas but NOT dask/sklearn in the base interpreter; the notebook kernel has them).
- Concept figure: rendered and on Desktop, user broadly happy after many iterations.
- Slides: wording drafted/advised; the user is assembling the actual deck (Google Slides) themselves.

## Open threads / next steps
- Friday meeting decides FPCA candy vs second paper. Adjust the closing framing accordingly.
- Verify the channel-per-phase claim (interim-summary last line and recap panel 3: "distributed at baseline / gaze-head pitch+yaw at take-over / steering+gaze-head yaw at action") against the ACTUAL snapshot PCA loading figures. Our FPCA showed it is event-dependent (steering dominates action in Motorcyclist/Falling rocks, head-eye yaw in Stag). If it doesn't hold on average, hedge or cut it.
- Optional build: base-vs-visual TRAJECTORY decoder (finding 3', same machinery, BaseCondition vs HUDOnly, N~80, lower power). Not built.
- User must run cell 191 once with TFCE globals alive to seed cos2_tfce_plotcache.joblib.
- Deferred deck fixes still pending (see project_fpca_pending_fixes.md): slide 29 slower->faster / more->fewer collisions / effect->affect, N consistency 160/157/143, slide 31 extinction wording with two-sided p=.700, spelling behaviour/behavior, "small but significant".

## Dead ends ruled out (do not re-explore)
- Per-epoch functional ED: confounded by epoch length (more timepoints -> more modes); length-matched it is noisy and event-dependent. Do not lead with it.
- funcPR(t) vs snapshot ED correlation (r=0.74-0.93): CIRCULAR. Both summarise the same post-onset covariance; snapshot ED is exactly recoverable from the Route-A output. It is a consistency check, not independent confirmation. Frame as "the trajectory decomposition traces the same rise-fall", never "a second method confirms it".
- Global-FPCA instantaneous dimensionality over the FULL window: numerically unstable on the quiet baseline (participation ratio of near-zero modal energies whips 1.8 -> 35 between adjacent points). This justifies fitting functional ED POST-ONSET only (not post-hoc: the modes are event-evoked and snapshot ED already supplies the baseline floor).
- "Motorcyclist is 2-3 dimensional" was WRONG (came from an overfit K=20 run). It is 1-dimensional like the others.

## Files touched this session
- `Analysis_ED_PCA.ipynb` (edited via JSON, it is 1.2MB > read cap): added MFPCA "BETWEEN-DRIVER FUNCTIONAL ED" subsection, added "DECODING PERFORMANCE FROM THE COVARIANCE OF THE TRAJECTORY" subsection, rewrote cell 191 (cos2 TFCE) with cache + x-axis change.
- `fpca_stats.py`: `mfpca_routeA` present from earlier (now redundant vs the in-notebook cells).
- `C:\Users\willi\Desktop\epochs_concept.png` (new 3-epoch marionette figure).
- `C:\Users\willi\.claude\projects\C--Users-willi\memory\reference_session_sync.md` (new) + `MEMORY.md` (pointer added).
- Scratchpad (not in project): concept_epochs.py, test_fed.py, test_decode.py, fe_cell.py, decode_cell.py, cos2cell_new.py.
- New cache file the user will create by running cell 191: `cos2_tfce_plotcache.joblib`.
