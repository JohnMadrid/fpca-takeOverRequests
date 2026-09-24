---
name: feedback-context-warning
description: Warn the user at the start of a task if it is likely to hit the context limit and force compaction mid-task
metadata:
  type: feedback
---
At the start of any task, estimate whether it will likely fill the context window (many long papers or files, big transcripts, long multi-step work). If it is likely, warn the user before starting. Propose a split into smaller turns or sessions, a leaner scope, or a fresh session. Keep tool outputs small: bounded grep, no full-file dumps. Save intermediate findings to scratchpad files so a compaction loses less.

**Why:** User (2026-09-19): "Compacting mid task is catastrophic and I'd like to avoid it as much as possible." A compaction during the FBI/LRP literature task lost valuable context and cost many tokens. The user is also on a limited Pro plan budget.

**How to apply:** Give a one-line warning with a proposed split before heavy work. Also mention it when usage is low.
