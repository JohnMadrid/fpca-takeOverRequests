# User Preferences

## Writing Style
- NEVER use an em dash (—) or double hyphen (--) in any response, ever
- NEVER use stereotypical AI expressions (see feedback_writing_style.md for full list)
- Write plainly; user copies output verbatim into papers

## Always Rules
- [always_rules.md](always_rules.md) -- permanent citation requirements (verbatim quotes, URLs, explicit fetch failures)
- [inferred_preferences.md](inferred_preferences.md) -- inferred working preferences across sessions

## Memory Files
- [feedback_writing_style.md](feedback_writing_style.md) -- writing and prose style rules
- [feedback_permissions.md](feedback_permissions.md) -- auto-approve all PowerShell calls, no prompting
- [feedback_response_length.md](feedback_response_length.md) -- cut responses ~30% shorter, no recaps
- [feedback_tfce_average_overlay.md](feedback_tfce_average_overlay.md) -- TFCE "average" plot = per-event curves overlaid with significant periods shaded, not pooled-driver matrix
- [feedback_progress_bar.md](feedback_progress_bar.md) -- always add tqdm progress bar to code that runs > 1 min
- [feedback_execute_dont_ask.md](feedback_execute_dont_ask.md) -- execute by default, report what you did not whether to; still ask before deletes/overwrites
- [feedback_advisor_style.md](feedback_advisor_style.md) -- act as advisor: challenge first, rate confidence [Certain/Likely/Guessing], ban filler agreement, hold positions; WINS on conflict
- [feedback_ground_in_background.md](feedback_ground_in_background.md) -- for data/code/presentation questions, ground in existing project background and ask when info is missing, don't treat as fresh
- [feedback_plot_titles.md](feedback_plot_titles.md) -- plots never have a title inside the figure/PNG; put the title as printed text before the plot in the output
- [feedback_cell_headers.md](feedback_cell_headers.md) -- refer to notebook cells by their header, never by index (user's VS Code shows no cell numbers)
- [feedback_slide_background.md](feedback_slide_background.md) -- slide figures use background #E9EBE8 (deck grey), never white
- [feedback_notebook_module_pattern.md](feedback_notebook_module_pattern.md) -- iterate analysis code in fpca_stats.py + autoreload wrapper cell, not inside notebook cells, to avoid Revert File
- [feedback_reference_access.md](feedback_reference_access.md) -- never rely on abstracts; if full text inaccessible, tell user and log in pending-refs list
- [feedback_add_to_notebook.md](feedback_add_to_notebook.md) -- add analysis as notebook cells by default; only run heavy compute standalone when explicitly asked

## Project Knowledge
- [project_fpca.md](project_fpca.md) -- FPCA driving sim project: parameters, design choices, results, bug fixes, cell map
- [project_shinya.md](project_shinya.md) -- Shinya R adaptation study: Master/Slave and TRY left/right reaching errors
- [project_fpca_pending_fixes.md](project_fpca_pending_fixes.md) -- deferred edits to Analysis_ED_PCA.ipynb (literal \n bug, p-value estimator, averaged null); apply only when next asked to edit main notebook
- [project_fbi_affordance_pending_refs.md](project_fbi_affordance_pending_refs.md) -- FBI/affordance/LRP proposal: refs read in full vs still missing (Sommer 1994 wrong PDF, Gordon 1994, Cardellicchio 2013, Lush 2020, Osman 1992)
- [feedback_context_warning.md](feedback_context_warning.md) -- warn at task start if context limit/compaction is likely; propose split, keep outputs small
- [reference_session_sync.md](reference_session_sync.md) -- /shutdown and /start slash commands (scripts/claude_sync.py) that git-sync work across machines; follow the command files verbatim, don't invent
