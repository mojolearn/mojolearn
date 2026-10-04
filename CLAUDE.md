# Working rules for Claude sessions and subagents

Every Claude session and subagent in this repo reads this file. Lane briefs add to it; they don't repeat it.

## Logs and output

- These context rules also apply to Codex and all delegated lanes; see `AGENTS.md`.
- Include the log/context reminder from `AGENTS.md` in every lane brief, spawn prompt and resumed assignment, and propagate it to nested subagents.
- Prefer targeted `rg` searches with bounded context and line lengths (`-m 20 -C 3 --max-columns 240 --max-columns-preview`); use `grep` when `rg` is unavailable. Narrow to relevant files first, and parse large JSON reports for selected fields instead of dumping them.
- Never print a whole log, race output or build output. Use `tail -n 20`, `grep -m 20`, or `grep -E 'error|FAIL|DISAGREE|status='`.
- For a build, read only its exit code and the first error: `... > build.log 2>&1; echo rc=$?; grep -m 5 -B 2 -A 8 'error' build.log`.
- For a race directory, read the one-line summary (`ALGOS lane=... status=... median_ms=...`), not the race log.
- Never read a subagent transcript or a `tasks/*.output` file.
- Put long results in a file under `~/mojolearn-evidence/` and report the path plus a few lines.
- Keep the original exit status and expected coverage. Filtered output is not proof of success; report failures, skipped/incomplete work, and expand the relevant diagnostic block when necessary.

## Lane subagents

- **Code, then queue.** A lane subagent writes code, compiles, commits and pushes, then queues its own GPU runs with `~/mojolearn-evidence/lq/lq` (run it with no arguments for usage). It never runs tests, timing, identity runs or Metal jobs on the laptop, and never ssh-es to, rents, extends or releases a box. `lq` is the only way a lane reaches a box:
  - `lq add nv|amd RACE <branch> <lane[,lane]> <ds[,ds]> [ARMS=..] [BUILDS=..] [ENV=V]`: one build covers every lane x dataset, so batch them.
  - Apple (the M2 Pro) is for same-bits ID checks only: no Apple timing for IDENTICAL. The M3 Ultra belongs to the Apple FAST peer (lane/apple-fast only). One lane and one dataset per line. Never RACE with `MOJOLEARN_VENDOR=cpu`: our CPU is never raced or timed; the host digest comes from the ID check.
  - **Same-bits checks use `lq add <box> ID <branch> <lane[,lane]> <ds[,ds]>`, never full board races.** It races the lane once on the device and once on the host column, over a 50k-row copy of the board data (identical on every box). It prints `IDCHECK <lane> <ds> <device>=<d> host=<d> MATCH|DIFFER`. Queue it on nv, amd and apple. `lq` refuses it when the branch changes no `.mojo` or `bindings/` file: then the kernels are main's and no check is needed.
  - Full-size RACE lines are for speed and quality only: NVIDIA and AMD timing and opponent comparisons. IDENTICAL speed work targets NVIDIA and AMD.
  - `lq add <box> CMD <branch> <tag> '<command>'`: runs in the branch tree. Scripts it calls must be committed in the branch.
  - `lq results <box> [pattern]` and `lq log <box> <id|tag> [pattern]`: grep-sized output only.

  Boxes: nv is the RunPod L40S, amd the DO MI325X, and apple is the M2 Pro (the M3 Ultra is the Apple FAST peer's). Each runs one job at a time. The orchestrator watches the queues and sends results back. After queuing, the lane ends with a reply listing what it queued (box, id), so the orchestrator can match the results.
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

## Experiments (Apple FAST and every speed lane)

- Every experiment is a `-D MOJOLEARN_<AREA>_FAST_<NAME>` define, default off, measured by an A/B on the box.
- A winner (faster, quality equal) becomes the FAST default with a `_OFF` define and a code comment citing the A/B numbers.
- A loser (slower, noise, quality or semantics change) never reaches main: delete its code from the lane before merging. It stays recoverable at the lane's recorded sha.
- Every experiment, kept or dropped, gets one row in `docs/apple-fast/EXPERIMENTS.md` (define, algorithm/dataset, branch@sha, A/B tag, before -> after ms, verdict, reason). Search it before writing a new experiment.

- Measurements run ONE run per arm (Andrew, Oct 3). tools/aft_ab.sh, afc_ab.sh and afc_ab_def.sh force it; AB_MULTI_RUN=1 is the only override. Queue A/B lines with reps 1, rounds 1, pairs 1.

## Briefs

The orchestrator saves every lane brief as `~/mojolearn-evidence/briefs-<date>/<lane>.md` before launching the lane. After a crash, a lane is relaunched from that file plus `git status` and `git log origin/main..HEAD` in its worktree.

## GPU rules (summary; the plans in docs/plans/ have the details)

- The GPU path is GPU only and parallel. No host steps inside a GPU fit, transform or predict, and no serial one-thread, one-block or per-sample default.
- Same bits on NVIDIA, AMD, Apple and the host column, within one version. Bits may change between versions: when a parallel kernel needs a different fold order, change the order on every vendor and in the host column together. Never keep a serial chain to preserve old bits.
- Never time a CPU or host route. The CPU is for verification digests, CPU-only installs and inference.
- Never add, rent, extend or release an Apple machine.
- Race and measure tools default to our GPU arm only; opponents are scored once, stored, and run only by an explicit opponent job.

## Wait for Mojo and Modular

If Mojo or Modular does not support something yet, do not build a workaround for it; wait for their support.
Examples: an AMD portable kernel path needs Mojo to keep kernel IR (or bitcode) and to accept generic gfx targets;
until Modular ships that, we do not pursue it (decided 2026-10-04; the parked prototype is on `lane/amd-portable`,
with the evidence in `docs/AMD_PORTABLE_PATH.md` on that branch). Do not hand-rewrite compiler output, patch
toolchain internals, or ship unsupported build modes. Record the ask for Modular instead and move on.
