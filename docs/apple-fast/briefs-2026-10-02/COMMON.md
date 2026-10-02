# Common brief for every Apple FAST family subagent (2026-10-02)

Read first: `/home/user/mojolearn/CLAUDE.md`, then `docs/apple-fast/NEXT_PASS.md` in your worktree (it is on origin/main),
then this file, then your family brief. Follow them exactly.

## Hard rules

- **Code only.** Never build, compile, test, time, race, run mojo/python benches, or ssh anywhere. There is no Mojo toolchain
  here. `lq` does not exist on this box: do not try it. Requests for measurements are lines in `docs/apple-fast/ab/<family>.txt`.
- **Your worktree is `~/mojolearn-wt/<family>` on branch `lane/apple-fast-<family>`.** Work only there. Never touch another
  worktree. Never check out, push to, or merge into `main` or `lane/apple-fast`. Never `--no-verify`, `git stash`, rebase,
  `reset --hard`, or `checkout --` anything you did not write.
- **Commit after every logical edit, push often**: `git push -u origin lane/apple-fast-<family>` (retry 2s/4s/8s/16s on
  network errors). The pre-push hook `tools/hooks/no_host_routes.py --tree` refuses new host routes in GPU code; if it refuses,
  fix the code, never edit `tools/hooks/host_routes_baseline.tsv`. Grep the hook output (`grep -E 'error|refus|NEW|FAIL'`),
  never paste it whole.
- **Merging main:** `git merge origin/main` (no rebase). Resolve conflicts so that main's IDENTICAL behaviour and main's
  host-route removals win, then re-express the FAST change on top. If main deleted a function the FAST change hooked into,
  re-hook into main's replacement; if main's replacement already does what the FAST switch did, drop the switch and say so in
  the commit message. Read the conflict hunks with `git diff --name-only --diff-filter=U` and `sed -n` around `<<<<<<<` markers;
  never cat whole files into your context.
- **Every change is FAST + Apple only and default OFF**, inside
  `comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` (or a comptime alias of it) and behind a
  `-D MOJOLEARN_<NAME>` define read with `is_defined["MOJOLEARN_<NAME>"]()`. Python-side changes only on a FAST-only branch.
  IDENTICAL must compile main's code unchanged. Prefer a `-D` define to an env switch: an env read is a host step. Do not add
  new `os.getenv`/`getenv` reads on fit/transform/predict paths.
- **FAST is GPU only:** no host loops over rows, no NumPy/sklearn math in the Python layer, no device-to-host round trips inside
  an iteration, no one-thread or one-block launches over runtime sizes, no env reads on the hot path. Quality must hold.
- **Mojo pointer types (the M3 compiler rejected the tier branch on this):** a kernel/launch helper that takes
  `MutPointer[T, MutAnyOrigin]` must be passed `buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()`, not a bare
  `buf.unsafe_ptr()` (that is `Pointer[T, origin_of(buf)]` and does not convert). Use the idioms already in the file you edit.
  Read `docs/apple-fast/m3/build-errors.txt` on `origin/lane/apple-fast-results`
  (`git show origin/lane/apple-fast-results:docs/apple-fast/m3/build-errors.txt | grep -m 10 error`) for the compiler's messages.
- **Commit message** states the static same-bits check in one line, e.g. "Static check: every changed line is inside a
  FAST + Apple guard (or the define's comptime branch); IDENTICAL compiles main's code unchanged." Never put a model name in a
  commit message.

## Request files (`docs/apple-fast/ab/<family>.txt`)

Light A/Bs: old FAST vs new FAST, one alternation (`1 2`), board-size data, one dataset per change first (the second dataset
only after the first wins). One tag per lane x dataset (never loop lanes under one tag). No `-ident` lines. Opponents are never
re-run. Forms:

    CMD lane/apple-fast-<family> <tag> AFC_FAMILY=<algos|classical|classical2> bash tools/afc_ab.sh <tag> <lane> <dataset> 1 2 - <ENV=1>
    CMD lane/apple-fast-<family> <tag> AFC_FAMILY=<algos|classical|classical2> bash tools/afc_ab_def.sh <tag> <binding> <lane> <dataset> 1 2 "" "-D <DEFINE>"
    CMD lane/apple-fast-<family> <tag> AFT_OUT=$HOME/aft-ab/<tag> bash tools/aft_ab.sh <binding> <lane> <dataset> 2 "" "-D <DEFINE>"

`afc_ab.sh` and `aft_ab.sh` are on main. `afc_ab_def.sh` (define-based classical A/B) is only on `origin/lane/apple-fast-tier`:
if your lines use it, copy it into your branch first
(`git show origin/lane/apple-fast-tier:tools/afc_ab_def.sh > tools/afc_ab_def.sh && chmod +x tools/afc_ab_def.sh`), commit it,
and read its header comment for the binding names. AFC_FAMILY picks the bench driver: `algos` = tools/bench_board_algos.py,
`classical` = tools/classical_two_datasets.py, `classical2` = tools/bench_board_more.py; grep the driver for your lane name to
pick the right one. Datasets: taxi, istella, istellarank (ranking), taxi-hourly (time series).
Also keep a short `docs/apple-fast/ab/<family>.md` note: switch name, what it changes, the files, and the risky compile sites.

## Final reply (short, no narrative, no diffs)

- branch and head sha;
- one line per change with the cause as `file:line`;
- bit changes, if any (should be none under IDENTICAL);
- the define(s);
- the request lines added (tags) and anything left unfinished.
