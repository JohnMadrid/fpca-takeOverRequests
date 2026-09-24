---
name: feedback_add_to_notebook
description: Deliver analysis as notebook cells by default; only run computations standalone when explicitly asked
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 391ce050-8ead-4f72-a500-c1d510fad3e8
  modified: 2026-08-26T15:33:50.175Z
---

For this project, when the user asks for an analysis/plot/test, **add the code to the notebook (`Analysis_ED_PCA.ipynb`) as a cell** by default. Do NOT run the computation yourself in a scratchpad script unless the user specifically asks you to run it / produce the numbers.

**Why:** the user runs cells in their own kernel (which already holds the expensive loaded data and intermediate results); running standalone re-loads data, can diverge from the kernel state, and wastes time. They want the code in place to run themselves.

**How to apply:** write the cell to a scratchpad `.py`, `ast.parse` check it, insert/replace in the notebook via json. Skip the standalone validation run unless asked. If a quick sanity check is genuinely needed, ask first. Relates to [[feedback_notebook_module_pattern]] and [[feedback_execute_dont_ask]] (execute-by-default still holds, but "execute" here means put it in the notebook, not run the heavy compute).
