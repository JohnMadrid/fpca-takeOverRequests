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

## Project Knowledge
- [project_fpca.md](project_fpca.md) -- FPCA driving sim project: parameters, design choices, results, bug fixes, cell map
- [project_shinya.md](project_shinya.md) -- Shinya R adaptation study: Master/Slave and TRY left/right reaching errors
- [project_fpca_pending_fixes.md](project_fpca_pending_fixes.md) -- deferred edits to Analysis_ED_PCA.ipynb (literal \n bug, p-value estimator, averaged null); apply only when next asked to edit main notebook
