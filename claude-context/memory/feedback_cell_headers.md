---
name: feedback_cell_headers
description: "Always identify notebook cells by their header comment/title, never by numeric index, because the user's VS Code view does not show cell indexes"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 391ce050-8ead-4f72-a500-c1d510fad3e8
---

When referring to any notebook cell, identify it by its **header** (the first meaningful comment line, or the markdown title), never by numeric index. The user reads notebooks in VS Code, where cell numbers are not displayed, so "cell 176" is meaningless and forces manual counting.

**Why:** Index references are unusable in the user's editor and change every time a cell is inserted, so they go stale immediately.

**How to apply:**
- Say: the cell headed `SIGNED PC1 LOADINGS over time, per event`
- Not: "cell 176"
- When inserting a cell, report both its header and the header of the cell it now sits after (e.g., "inserted `X` right after `Y`").
- When listing several cells, use a table of headers, not indexes.
- Indexes may only appear as a secondary detail alongside the header (never alone), and only if genuinely useful (e.g., explaining a script's insertion logic).
- Locate cells in scripts by matching header text (`"HEADER" in "".join(cell["source"])`), which is what the notebook-edit scripts already do, so nothing changes on the implementation side.

Related: [[feedback_plot_titles]], [[project_fpca]].
