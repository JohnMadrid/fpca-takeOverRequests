---
name: session-sync-commands
description: The /shutdown and /start slash commands that git-sync this project across machines (mobile <-> home)
metadata: 
  node_type: memory
  type: reference
  originSessionId: 391ce050-8ead-4f72-a500-c1d510fad3e8
  modified: 2026-09-24T18:02:11.699Z
---

The user works across machines ("on mobile basis" so work goes home) and has built project slash commands to carry continuity via git. When they say /shutdown or /start, follow the command file verbatim; do not invent the procedure. The command files are authoritative.

Location (in the fpca project): `C:\Users\willi\Desktop\fpca-takeOverRequests\.claude\commands\shutdown.md` and `start.md`. Engine: `scripts/claude_sync.py` (also `scripts/claude_onboard.py` for first-time onboarding of the big transcript from USB). Run from the project root.

**/shutdown** (end of session): 0) run `/compact` FIRST. 1) write a thorough handoff to `claude-context/LAST_SUMMARY.md` (overwrite): date, what we did + why, current state (done/mid-flight/verified), open threads, dead ends ruled out, files touched, contents of any skipped large transcript. Plain prose, no em dashes. 2) `python scripts/claude_sync.py list-images`, show the user, ASK keep vs discard, then `keep-images` or `discard-images`. 3) `python scripts/claude_sync.py shutdown` (copies memory .md + global CLAUDE.md into claude-context/, gzips last 3 transcripts skipping any >~150MB, stages, commits dated, pushes). 4) report engine output verbatim-ish (files carried, transcripts carried vs skipped w/ sizes, commit/push result); if push failed, say so, do not claim success.

**/start** (resume on a machine): `python scripts/claude_sync.py start` (pulls, re-homes memory + CLAUDE.md, gunzips transcripts), then read `LAST_SUMMARY.md` + memory and brief the user (last date, where we left off, next steps, any summary-only transcript). Offer `claude --resume` for the full thread.

Note: working dir may be `C:\Users\willi`; cd to the project root first. Related: [[fpca-project-knowledge]].
