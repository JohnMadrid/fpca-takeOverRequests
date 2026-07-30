---
name: progress-bar-for-long-tasks
description: Always add a tqdm progress bar to any code that takes more than 1 minute to run
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 156e862b-04ec-41f9-9f77-9165a7e1d931
---

Whenever I write code that will take more than 1 minute to run (perm tests, CV loops, multi-event computations, ellipsoid fits over many timepoints, etc.), add a `tqdm.auto.tqdm` progress bar to the outer loop. Print elapsed/ETA. Per-iteration prints inside long loops are also fine when tqdm is not appropriate.

**Why:** User has been bitten multiple times by runs going hours with zero feedback. Without a progress bar, they cannot tell if the cell is making progress, stuck, or silently broken. Has caused wasted ~50-60 min runs.

**How to apply:**
- Default to `from tqdm.auto import tqdm` with a try/except fallback to identity
- Wrap the outermost loop that runs > 1 min total
- For nested loops where each iteration is long, also print or use tqdm per inner step
- Plot/output incremental results when feasible (per-contrast, per-event) instead of only at the end
- For permutation tests specifically: tqdm over the perm count, plot per (event, condition) as each finishes
