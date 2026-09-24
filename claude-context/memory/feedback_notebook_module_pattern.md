---
name: feedback_notebook_module_pattern
description: "Iterate analysis code in fpca_stats.py + autoreload, not inside notebook cells, to avoid Revert File"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 391ce050-8ead-4f72-a500-c1d510fad3e8
  modified: 2026-08-17T15:47:28.160Z
---

For the FPCA project ([[project_fpca]]), the user was tired of clicking "Revert File" in VS Code every time I edited a cell on disk. Root cause: VS Code holds the open notebook's in-memory model (with run outputs), so any on-disk `.ipynb` edit is a conflict the user must resolve manually.

**Why:** editing code that lives inside notebook cells forces a revert every iteration.

**How to apply:** put code we iterate on into a plain module (`fpca_stats.py` in the project root) and make the notebook cell a thin wrapper:
```python
%reload_ext autoreload
%autoreload 2
import fpca_stats
fpca_stats.some_fn(<kernel objects passed as args>)
```
Then I edit `fpca_stats.py` (not the open notebook, so no conflict) and the user just re-runs the wrapper cell; autoreload re-imports. No Revert File, ever, for that code. Module functions must take kernel objects (arrays, ED-curve fn, RT boundaries) as ARGUMENTS, not read notebook globals.

First function moved: `jackknife_3phase` (per-event gold-ED 3-phase jackknife + inverse-variance event-mean). Wrapper cell sits after the PHASE-MEAN PLOT cell.

Caveat: this only removes reverts for analysis/plotting code. Structural notebook edits (adding/reordering cells) still write the `.ipynb` and still need one revert. Default to the module pattern for new iterative code unless the user says otherwise.

**CRITICAL refinement (user got frustrated 2026-08-17): the WRAPPER CELL MUST BE A FIXED ONE-LINER.** The module trick only avoids reverts when the cell itself never changes. I kept causing reverts by editing the cell's CALL each time the user wanted a tweak (function name, then arguments like `stars=False, star_size=12`, then capturing the return). Every change to the cell's call signature is a notebook edit = a revert. Fix: the cell calls ONE stable entry-point function with NO tunable arguments (e.g. `fpca_stats.phase_plots(_cs_res)`); ALL styling/options/which-versions logic lives INSIDE that module function. When the user wants any change (sizes, versions, colors, stars, adding a plot), edit the module function, never the cell. Also: module plot fns that `plt.show()` AND `return fig` cause double-display if called as a bare last cell line — either capture the return or have the stable wrapper fn return None.
