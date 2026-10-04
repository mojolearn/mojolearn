# Agent working rules

## Logs and context

- Save complete build, test, and GPU logs to files. Inspect the exit status and
  structured summary first; do not stream or paste entire logs into context.
- Prefer targeted `rg` searches (or `grep` when unavailable), bounded context,
  and short tails. For example:
  `rg -n -i -m 20 -C 3 --max-columns 240 --max-columns-preview 'error|fail|refused|divergent|summary' run.log`
  or `tail -n 20 run.log`. Narrow to relevant files before searching directories.
- Parse large JSON reports for selected fields, counts, and failures rather
  than printing whole documents. Keep full evidence on disk and report its path.
- Expand the relevant excerpt when needed to establish the cause. A filtered
  excerpt or a few passing lines never proves the whole run passed: preserve
  exit codes, expected coverage, failures, and incomplete or skipped work.
- Never read whole subagent transcripts or `tasks/*.output` files. Request a
  concise status or inspect the specific retained artifact instead.

## Lane and subagent prompts

Include this reminder in every lane brief, spawn prompt, and resumed assignment,
and ask delegated agents to propagate it to their own subagents:

> Keep logs out of context: save complete output to files, use targeted rg/grep
> with bounded surrounding lines and short tails, and summarize exit status,
> coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
> never hide failures or infer full success from filtered output.

Final handoffs should give the branch/commit, changes, validation status,
remaining work, and evidence paths without pasting full logs or diffs.
