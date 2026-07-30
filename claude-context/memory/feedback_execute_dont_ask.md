---
name: execute-dont-ask
description: "User wants execution by default, not permission-seeking; report what you did, not whether to do it"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 156e862b-04ec-41f9-9f77-9165a7e1d931
---

The user does not want confirm-the-plan questions before acting. Execute using best judgment and report WHAT was done, not whether it should be done.

**Why:** Repeated "confirm A/B/C before I write it" questions slowed the work down; the user found the constant permission-seeking frustrating across a long analysis session.

**How to apply:**
- For analysis, plotting, cell edits, notebook patches, refactors: just do it, then summarize what changed. Do not ask which option to pick when a sensible default exists — pick it, state the choice, proceed.
- STILL ask before: deleting files/data, overwriting files the user didn't create or that contradict their description, and other hard-to-reverse or outward-facing actions.
- This overrides the general tendency to surface options first. State recommendations as decisions made, not questions posed.

See also [[feedback_permissions]] (auto-approve PowerShell), [[feedback_response_length]] (cut ~30%, no recaps).
