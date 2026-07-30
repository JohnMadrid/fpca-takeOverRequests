---
name: fpca-pending-fixes
description: Deferred edits to apply to Analysis_ED_PCA.ipynb next time the user asks to change the main notebook (do NOT touch unprompted)
metadata: 
  node_type: memory
  type: project
  originSessionId: 156e862b-04ec-41f9-9f77-9165a7e1d931
---

Deferred fixes the user wants applied LATER, only when they next ask to edit the main notebook `Analysis_ED_PCA.ipynb`. Do not edit unprompted (file was running on 2026-06-18).

## Cell `# GOLD ED -- PERMUTATION TEST` (time-reversal ED pre/post, ~cell 141)

1. **Literal `\n` bug in printout.** The last line of `_gp_plot` is
   `print(f"  p-value (one-tailed)     : {p:.3f}\\n")` (stored as `\\\\n` in the
   .ipynb JSON), which prints a literal backslash-n instead of a newline. Change
   the `\\n` to a single real `\n`.

2. **p-value estimator.** Uses raw `np.mean(null<=obs)` (can return exactly 0).
   Change to the floored unbiased estimator `(np.sum(null<=obs)+1)/(_GP_B+1)` for
   consistency with the LDA cell and the region-label cell, and so it can't print
   p=0. One-tailed direction stays post<pre (`<=`).

3. **Averaged-across-events variant** averages three separate per-event null arrays
   by index (`nl_avg = mean of stacked nulls`), which is ad hoc (different W per
   event, same seed, index correspondence arbitrary). The pooled-participants
   variant is the clean combined test. If the user wants the averaged variant
   reported, rebuild it as a proper joint null: one flip set per permutation
   applied across all events, average the resulting curves within each draw, then
   post-pre.

**Why:** User caught the `\n` in output and confirmed it's a bug; the other two
are rigor issues I flagged. User trusts this time-reversal test over the
region-label test (cell 137) for pre/post because it shuffles participants (the
independent unit) and preserves within-driver autocorrelation. See [[project_fpca]].
