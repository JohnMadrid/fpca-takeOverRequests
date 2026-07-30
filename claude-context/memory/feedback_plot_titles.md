---
name: feedback_plot_titles
description: Plots must never carry a title inside the figure/PNG; put the title as printed text before the plot in the cell output instead
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 391ce050-8ead-4f72-a500-c1d510fad3e8
---

When generating any plot (matplotlib or otherwise), do NOT put a title inside the figure: no `ax.set_title(...)`, no `fig.suptitle(...)`, no `plt.title(...)`, and no title baked into a saved PNG. Instead, emit the title/description as **printed text before the plot** in the cell output (a `print(...)` immediately before `plt.show()` / before saving).

**Why:** The user copies figures verbatim into papers and slides (see [[feedback_writing_style.md]]) and adds their own captions/titles there; an embedded title clutters the clean figure and duplicates the caption.

**How to apply:**
- Keep axis labels (`set_xlabel`/`set_ylabel`), legends, and annotations, only the title is removed.
- Per-panel: instead of `ax.set_title("X")`, print the panel name/description as text above the plot.
- For a multi-panel figure, print a single descriptive line (and any per-panel labels) before `plt.show()`.
- For saved PNGs (no inline output), simply omit the title from the PNG.
