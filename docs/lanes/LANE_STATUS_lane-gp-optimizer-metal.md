# Lane status: lane/gp-optimizer-metal (2026-09-16)

One job, done: take the Metal column `lane/gp-optimizer` still owed at its merge
commit, so `gp-normalize-y` RECORDS instead of REFUSING, and merge it. The
feature itself was already on main; this branch carries evidence and docs only,
no arithmetic.

## What was taken

Branch cut from `main` at `2807d4ad7`. The column, through the EXCLUSIVE Metal
slot, one core, `MOJOLEARN_NUMERIC_MODE=identical` set explicitly:

    tools/identity_break.py \
      --lanes gp,gp-matern12,gp-matern32,gp-matern52-ard,gp-normalize-y,gp-optimize,gp-optimize-restarts \
      --repeats 2 --vendor apple-m4 --json apple-m4.json

**`cells=63 stable=63 moved=0 refused=0`**; infer, model and batch 63 stable
each; **no REFUSED line anywhere in the column.** `gp-normalize-y` carries real
hashes on all nine fixtures, which is the whole point of the rerun.

Diffs, all against columns already committed (no new CPU or GPU work):

| check | verdict |
|---|---|
| Metal against the x86 CPU column | IDENTICAL=63 train, 126 infer and model, 63 batch, nothing ONE-COLUMN |
| the four recorded gp lanes against the 166-lane record | IDENTICAL=36 train, 36 infer and model, 36 batch, **no cell moved** |
| owed accounting, `--require-columns 4` | OK, **OWED=144** cell parts to the next release record |
| gradient sabotage against the production CPU column | DIVERGENT=18 train, 36 infer and model, 18 batch; IDENTICAL=36 on recorded lanes |
| per cell, which cells the sabotage moved | `gp-optimize` 9 of 9 moved, `gp-optimize-restarts` 9 of 9 moved, 0 unmoved, 0 missing |

The sabotage was SEEN to move every new cell and no old one. It is compared on
the first hash and first parts dict, never whole values, because the columns
differ in `--repeats` and a whole-value comparison would read every cell moved
and could not fail.

`diff_record_gp.txt` was REGENERATED from the new column rather than left as the
stale artifact of the old one. Its `--require-columns 4` still exits non-zero
with 36 shortfalls; that is pre-existing and not a regression, and it was
re-verified rather than inherited: **all 36 are `model` parts and the non-model
list is empty**, because the 166-lane record predates the GP's save and load and
carries no model hash.

## Two traps worth keeping

1. **The refusal trap.** A fresh worktree has no `python/mojolearn/identical/`
   at all. A column run there reads REFUSED on every cell with "the base binding
   has no `transpose_f32`", which looks like a lane defect and is not. Stage a
   COMPLETE identical set first and check it by IMPORT: Mojo `def_function`
   exports are registered at module init, so `nm` cannot answer the question.
2. **The silent-refusal trap.** `identity_break._run_reference` defaults the
   wanted numeric mode to `fast` when `MOJOLEARN_NUMERIC_MODE` is unset and
   refuses before the first fit, writing NO JSON. A run that produces nothing
   looks exactly like a run that never started. Set the mode explicitly.

## What this session did NOT do, and why

Two other lanes were queued behind this one and both were **dropped without
recording anything.** Each is a legitimate state to sit in, and the reasons are
in their own lane status files so they are not picked up as unfinished business:

- **`lane/gbdt-rest`** (`docs/lanes/LANE_STATUS_lane-gbdt-rest.md`): HELD and
  then DROPPED. Its fixtures are being shrunk: four GBDT lanes are among the ten
  most expensive in the record (`gbdt-parametric-losses` 1701 s, `gbdt-nan-modes`
  969 s, `gbdt-lossguide-newtoncosine` 838 s, `gbdt-pair-logit` 794 s) and are
  being cut to under 30 s each. A column taken before that change is a column for
  fixtures that no longer exist, and **a sabotage arm proven live at one fixture
  size can be inert at another**. Two of the four being cut sit in this lane's own
  ten-lane spot-check set. Nothing was recorded; an orphaned process that
  re-acquired the Metal lock and started recording at pre-shrink sizes was stopped
  and its partial JSON deleted.
- **`lane/neighbors-rest`** (`docs/lanes/LANE_STATUS_lane-neighbors-rest.md`):
  DROPPED. Its x86 CPU column needs a rented pod, which was not permitted, and
  its only GPU-needing piece (item 2's tests) was not worth another lock
  acquisition. Item 2 remains UNMEASURED, and its open "add a lane or decide in
  writing" question is now answered in writing there.

**The governing rule, learned the expensive way today:** Metal is the scarcest
resource we have, one machine and one GPU shared by every agent. It is for
proving a lane's OWN NEW CELLS, never for working through a backlog of owed
columns lane by lane. Owed GPU cells ride the once-per-release GPU record.

## Still owed here

The NVIDIA and AMD cells of `gp-optimize` and `gp-optimize-restarts`, and the
144 cell parts in `owed.json`, all to the next release record. No box was rented
by this branch and nothing is owed on one.
