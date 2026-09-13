# Context Usage Rules

You are working on a large project.

Your primary goal is to minimize unnecessary file reads and token usage.

Rules:

1. Never scan the entire project unless I explicitly say "Analyze the whole project."

2. Before reading files, identify which files are actually required for the task.

3. Read only the minimum number of files needed.

4. Do not reopen files that have already been analyzed during this conversation unless I mention they have changed.

5. Do not search datasets, documentation, notebooks, configuration files, or unrelated folders unless they are directly relevant.

6. If the request is limited to one feature or module, stay inside that module.

7. If you are unsure which file is needed, ask me instead of searching the whole project.

8. Prefer incremental changes over project-wide analysis.

9. Keep context usage and API/token consumption as low as possible.

These rules should be followed for every prompt unless I explicitly override them.
