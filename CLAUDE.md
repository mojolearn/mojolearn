# Working rules for Claude sessions and subagents

Every Claude session and subagent in this repo reads this file. Lane briefs add to it; they don't repeat it.

## Logs and output

- Never print a whole log, race output or build output. Use `tail -n 20`, `grep -m 20`, or `grep -E 'error|FAIL|DISAGREE|status='`.
- For a build, read only its exit code and the first error: `... > build.log 2>&1; echo rc=$?; grep -m 5 -B 2 -A 8 'error' build.log`.
- For a race directory, read the one-line summary (`ALGOS lane=... status=... median_ms=...`), not the race log.
- Never read a subagent transcript or a `tasks/*.output` file.
- Put long results in a file under `~/mojolearn-evidence/` and report the path plus a few lines.

## Lane subagents

- **Code, then queue.** A lane subagent writes code, compiles, commits and pushes, then queues its own GPU runs with `~/mojolearn-evidence/lq/lq` (run it with no arguments for usage). It never runs tests, timing, identity runs or Metal jobs on the laptop, and never ssh-es to, rents, extends or releases a box. `lq` is the only way a lane reaches a box:
  - `lq add nv|amd RACE <branch> <lane[,lane]> <ds[,ds]> [ARMS=..] [BUILDS=..] [ENV=V]`: one build covers every lane x dataset, so batch them.
  - `lq add m3|m2 RACE <branch> <lane> <ds> [ENV=V]`: one lane and one dataset per line. Add a second line with `MOJOLEARN_VENDOR=cpu` for the host digest column.
  - `lq add <box> CMD <branch> <tag> '<command>'`: runs in the branch tree. Scripts it calls must be committed in the branch.
  - `lq results <box> [pattern]` and `lq log <box> <id|tag> [pattern]`: grep-sized output only.

  Boxes: nv is the RunPod L40S, amd the DO MI325X, m3 the M3 Ultra, m2 the M2 Pro. Each runs one job at a time. The orchestrator watches the queues and sends results back. After queuing, the lane ends with a reply listing what it queued (box, id), so the orchestrator can match the results.
- **Compile through the slot semaphore:** `bash ~/mojolearn-evidence/compile_slot.sh <command>`. It allows 4 compiles machine-wide at `nice -n 19`. Use `-j 1` and `MOJOLEARN_COMPILE_JOBS=1`.
- **One worktree per lane:** `~/mojolearn-wt/<lane>` on branch `lane/<lane>`. Commit after every edit and push often, because a crash or reboot loses anything uncommitted. Never `git stash`, rebase, `reset --hard` or `checkout --` someone else's edits.
- **Nothing in `/private/tmp`.** It's wiped on reboot. Keep briefs, notes and scripts in the worktree or `~/mojolearn-evidence/`.
- **Use bash, not zsh, for loops and variable expansion.** zsh doesn't word-split `$FLAGS`, so defines get dropped silently.
- **Final reply format, short:**
  - branch and head;
  - one line per change, with the cause as `file:line`;
  - bit changes, if any;
  - the A/B define, if any;
  - what it queued (box and id), plus any RUN OWED that `lq` can't express.

  No narrative, no pasted diffs.

## Briefs

The orchestrator saves every lane brief as `~/mojolearn-evidence/briefs-<date>/<lane>.md` before launching the lane. After a crash, a lane is relaunched from that file plus `git status` and `git log origin/main..HEAD` in its worktree.

## GPU rules (summary; the plans in docs/plans/ have the details)

- The GPU path is GPU only and parallel. No host steps inside a GPU fit, transform or predict, and no serial one-thread, one-block or per-sample default.
- Same bits on NVIDIA, AMD, Apple and the host column.
- Never time a CPU or host route. The CPU is for verification digests, CPU-only installs and inference.
- Never add, rent, extend or release an Apple machine.
