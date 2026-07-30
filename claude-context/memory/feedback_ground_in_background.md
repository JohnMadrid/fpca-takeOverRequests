---
name: feedback_ground_in_background
description: "For questions about data/code/presentation, ground answers in existing project background and ask when info is missing rather than assuming or treating as fresh"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 391ce050-8ead-4f72-a500-c1d510fad3e8
---

When the user asks about the data, code, or the presentation, ground the answer in the established background rather than treating the question as fresh: the project memory ([[project_fpca]], [[project_fpca_pending_fixes]]), decisions already made in-session, and the actual repo code/data (read notebooks/CSVs on demand). If the needed information is not in memory or the code, ASK a targeted question instead of assuming or inventing.

**Why:** The FPCA project carries extensive accumulated context and hard-won decisions. Treating each question as fresh loses that history and risks contradicting established choices (e.g. the 5 car-frame channels, the pandas-2.2.2 substitution, the HitObjectName dtype fix).

**How to apply:**
- Before answering, recall relevant memory and check the real code/data; cite what you find (see [[always_rules]]).
- The ~897 MB home transcript (`156e862b...jsonl`) is on disk and `--resume`-visible but NOT loaded in-session; do not assume its contents. Grep it for a specific point if a question depends on it.
- When background is insufficient, ask rather than guess. Consistent with [[feedback_advisor_style]] and [[feedback_execute_dont_ask]].
