---
description: Start-of-day sync. Pull, restore memory + CLAUDE.md, relink transcripts, brief me.
---

You are starting work on this machine and need to restore continuity from git
(and, on a freshly onboarded machine, from the USB payload). Do these IN ORDER.

## 1. Run the sync engine
Run from the project root:
```
python scripts/claude_sync.py start
```
This pulls git, copies `claude-context/memory/*.md` into THIS machine's memory dir
(re-homing across the different username/drive), restores the global `CLAUDE.md`,
and gunzips any carried transcripts into this machine's project-hash dir so
`claude --resume` can list them.

## 2. Load the handoff
Read `claude-context/LAST_SUMMARY.md` and the restored memory files
(`MEMORY.md`, `always_rules.md`, `inferred_preferences.md`, `project_fpca.md`).
Then give the user a short briefing:
- the date of the last session,
- where we left off (1-3 lines),
- the open next steps from the summary,
- any transcript that was skipped last time (so they know it is summary-only here).

## 3. Offer the resume path
If the engine relinked a transcript, tell the user they can natively continue that
exact thread by quitting and running `claude --resume` (or picking it in the VS
Code session list). Make clear: the in-session briefing gives them the summary +
memory now; `--resume` gives them the full thread. They are two separate things.

## Notes
- First time on a brand-new machine: the big transcript is not in git. Onboard it
  once via `python scripts/claude_onboard.py import <usb_payload>` BEFORE relying
  on `--resume` for the mega-session. Day-to-day after that, this command is enough.
- Do not fabricate continuity. If `LAST_SUMMARY.md` is missing, say so and proceed
  from memory files alone.
