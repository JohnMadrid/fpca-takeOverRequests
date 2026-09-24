---
name: feedback_slide_background
description: "All figures/illustrations meant for the presentation use the slide background color #E9EBE8, never white"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 391ce050-8ead-4f72-a500-c1d510fad3e8
  modified: 2026-08-03T15:39:06.379Z
---

Figures intended for the user's slide deck must use the presentation's background color, **#E9EBE8** (rgb 233,235,232, a light warm grey), not the matplotlib default white. Sampled as the modal color of every content page of `Downloads/project tea-time presentation.pdf`.

**Why:** A white-background PNG shows an obvious white rectangle floating on the grey slide. Matching the background makes the figure sit seamlessly on the slide.

**How to apply:**
- Top of any figure script for slides:
  `plt.rcParams.update({"figure.facecolor":"#E9EBE8","axes.facecolor":"#E9EBE8","savefig.facecolor":"#E9EBE8"})`
- Applies to figure, axes, and the saved PNG at once.
- Blob/imshow figures should use a colormap transparent at the low end so the grey shows through.
- This is the default for slide figures going forward; if the user changes the deck's background, re-sample and update this value.

Related: [[feedback_plot_titles]] (no in-figure titles), [[project_fpca]].
