---
name: Always Rules
description: Rules the user has explicitly marked as permanent preferences - apply in every response without exception
type: feedback
originSessionId: 156e862b-04ec-41f9-9f77-9165a7e1d931
---
## Sources

When citing any internet source, always provide:
- Full URL
- Author/organization
- A direct verbatim quote from the page (not a paraphrase) confirming the specific claim
- If the page could not be fetched or verbatim text could not be extracted, state this explicitly rather than asserting the claim

Exception: claims framed explicitly as "general consensus in the field" with no specific source attached do not require citation, but must be flagged as such.

**Why:** User does not have time to verify sources themselves and requires that citations be real and checkable. Fabricated or paraphrased citations are not acceptable.

## Notebook cell references

Never refer to a cell by its index number. Always identify a cell by its first non-empty line of source code (e.g. `# Run PCA loadings plot` or `def plot_pca_loadings_over_time`).

**Why:** Cell index numbers in the notebook do not match Jupyter execution counts or the user's view in the IDE. First lines are stable and unambiguous.
