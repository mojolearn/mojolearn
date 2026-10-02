# Working rules for Claude sessions and subagents

Every Claude session and subagent in this repo reads this file. Lane briefs add to it; they don't repeat it.

## Logs and output

- Never print a whole log, race output or build output. Use `tail -n 20`, `grep -m 20`, or `grep -E 'error|FAIL|DISAGREE|status='`.
- For a build, read only its exit code and the first error: `... > build.log 2>&1; echo rc=$?; grep -m 5 -B 2 -A 8 'error' build.log`.
- For a race directory, read the one-line summary (`ALGOS lane=... status=... median_ms=...`), not the race log.
- Never read a subagent transcript or a `tasks/*.output` file.
- Put long results in a file under `~/mojolearn-evidence/` and report the path plus a few lines.

## Lane subagents

- **Code only.** A lane subagent writes code, compiles, commits and pushes. It never runs tests, timing, identity runs or Metal jobs on the laptop, and never touches a remote box. It returns the runs it needs as **RUN OWED** lines, with exact commands. The orchestrator runs them, one queue per box.
- **Compile through the slot semaphore:** `bash ~/mojolearn-evidence/compile_slot.sh <command>`. It allows 4 compiles machine-wide at `nice -n 19`. Use `-j 1` and `MOJOLEARN_COMPILE_JOBS=1`.
- **One worktree per lane:** `~/mojolearn-wt/<lane>` on branch `lane/<lane>`. Commit after every edit and push often, because a crash or reboot loses anything uncommitted. Never `git stash`, rebase, `reset --hard` or `checkout --` someone else's edits.
- **Nothing in `/private/tmp`.** It's wiped on reboot. Keep briefs, notes and scripts in the worktree or `~/mojolearn-evidence/`.
- **Use bash, not zsh, for loops and variable expansion.** zsh doesn't word-split `$FLAGS`, so defines get dropped silently.
- **Final reply format, short:**
  - branch and head;
  - one line per change, with the cause as `file:line`;
  - bit changes, if any;
  - the A/B define, if any;
  - RUN OWED.

  No narrative, no pasted diffs.

## Briefs

The orchestrator saves every lane brief as `~/mojolearn-evidence/briefs-<date>/<lane>.md` before launching the lane. After a crash, a lane is relaunched from that file plus `git status` and `git log origin/main..HEAD` in its worktree.

## GPU rules (summary; the plans in docs/plans/ have the details)

- The GPU path is GPU only and parallel. No host steps inside a GPU fit, transform or predict, and no serial one-thread, one-block or per-sample default.
- Same bits on NVIDIA, AMD, Apple and the host column.
- Never time a CPU or host route. The CPU is for verification digests, CPU-only installs and inference.
- Never add, rent, extend or release an Apple machine.
