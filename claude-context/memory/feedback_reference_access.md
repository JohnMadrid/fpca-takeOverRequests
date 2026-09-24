---
name: feedback-reference-access
description: Never rely on abstracts/summaries for a needed reference; if full text is inaccessible, tell the user and log it in the pending-references memory
metadata:
  type: feedback
---

When a reference is needed to support a claim, read the full text. Abstracts, search-engine summaries, and WebFetch model summaries are not enough to characterise what a study did or found.

If the full text cannot be accessed (paywall, 403, captcha, corrupted PDF), do not silently fall back on the abstract. Tell the user explicitly, and add the reference to [[project-fbi-affordance-pending-refs]] (or an equivalent pending list for the relevant project) so they can obtain and feed it.

**Why:** User stated (2026-09-19): "If you failed to access any of these papers, dont trust just abstracts or summaries. Always let me know (add to memory) if you decided that a reference is needed, but could not access so gave up." Extends [[always_rules]] citation requirements.

**How to apply:** Before characterising a paper's design or findings, confirm full-text access. Useful access routes: user-supplied PDFs, Europe PMC REST API, PMC, publisher open access, Unpaywall/OpenAlex (check oa_status first). If none work, state it and log it.
