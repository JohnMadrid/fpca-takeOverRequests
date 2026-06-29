---
description: End-of-day sync. Compact, write summary, push memory + recent transcripts to git.
---

You are ending the work session and syncing continuity to git so the user can
resume on another machine. Do these steps IN ORDER. Do not skip the summary.

## 0. Compact first
Your FIRST action is to compact the conversation by running the `/compact`
command. This frees the context and produces the model's own summary of the
session. Only after compaction completes do you proceed to step 1.

## 1. Write the handoff summary
Write a thorough handoff to `claude-context/LAST_SUMMARY.md` (overwrite it). This
is the file the next machine reads first, so make it complete, not terse. Include:
- **Date** (absolute, today's date).
- **What we did this session** (bullet list, decisions + why, not just actions).
- **Current state** of the work (what's done, what's mid-flight, what's verified vs not).
- **Open threads / next steps** the next session should pick up.
- **Any dead ends** we ruled out, so the next session doesn't re-explore them.
- **Files touched** this session (paths).
- If any large session transcript will be SKIPPED by the size cap, note here what
  was in it so its content isn't lost to the next machine.

Write plainly (the user copies prose into papers): no em dashes, no AI-stereotype
phrasing. Keep it information-dense; length is fine here.

## 2. Session images: ask before committing
List the images produced this session that are staged in
`claude-context/session-images/`:
```
python scripts/claude_sync.py list-images
```
Show the user the list and ASK: "Add these session images to plots/keep/ (they get
committed), or discard them?" Then run exactly one of:
```
python scripts/claude_sync.py keep-images      # user said yes -> moves to plots/keep, warns if >500 MB
python scripts/claude_sync.py discard-images   # user said no  -> deletes staged images
```
If there are no staged images, skip this step. Relay any size warning verbatim.

## 3. Run the sync engine
Run from the project root:
```
python scripts/claude_sync.py shutdown
```
This copies the memory `.md` files + global `CLAUDE.md` into `claude-context/`,
gzips the last 3 session transcripts (skipping any whose gzip exceeds the ~150 MB
cap and noting it), stages the working tree + context, commits with a dated
message, and pushes.

## 4. Report
Relay the engine's report to the user verbatim-ish: how many memory files were
carried, which transcripts were carried vs skipped (with sizes), and the commit/
push result. If the push failed, say so plainly and show the error; do not claim
success.

## Notes
- Nothing is stripped or reduced. Files commit whole. The ONLY exception: any file
  whose GZIP exceeds ~150 MB is not committed (it stays USB-only and must be
  summarised in `LAST_SUMMARY.md`). Notebooks gzip to tens of MB, so they commit
  normally; this cap only catches genuine monsters like raw 900 MB transcripts.
- The full history (including the big transcript) is carried to new devices once by
  hand via `scripts/claude_onboard.py`. Daily git then carries memory, summary, and
  all sub-cap changes.
- This command compacts FIRST (step 0), then writes the handoff and syncs, so you
  never have to run /compact separately.
