# Claude Code Instructions

## Session start and post-compaction confirmation

At the very start of every session, and after every context compaction, read the following three files and confirm to the user that you have done so:

1. `C:/Users/willi/.claude/projects/C--Users-willi/memory/always_rules.md`
2. `C:/Users/willi/.claude/projects/C--Users-willi/memory/inferred_preferences.md`
3. `C:/Users/willi/.claude/projects/C--Users-willi/memory/project_fpca.md`

Confirmation message format (brief, one line per file):
> Loaded session context:
> - always_rules.md (citation rules)
> - inferred_preferences.md (working preferences)
> - project_fpca.md (FPCA project knowledge)

Do this before responding to the first user message. Do not skip this even if the user's first message is trivial.
