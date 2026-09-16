# The decode lanes' CPU column, at the shrunk fixtures

`cpu-apple-m4.json` is the CPU host-route column `lane/stateful-cpu-decoding`
took after merging origin/main, preserved here as a RECORD rather than only as
a log. It was produced in that lane's own worktree
(`package.package_dir = /Users/andrewhendel/mojolearn-wt/stateful-cpu-decoding/
python/mojolearn`), so its host bindings are the branch's, and its `commit`
field (`c8615bb71`) names the tree it ran on. The original file is
`~/mojolearn-evidence/stateful-cpu-decoding/19_lane_verification_after_merge.json`.

## Why it is here

Two measured reasons, both checked against the tree rather than taken from a
summary (lane/inference-coverage-complete, 2026-09-16).

1. **Three of its nine lanes have no other reference at the current fixture.**
   `f4e589395` (2026-09-16 06:28) shrank thirteen identity fixtures and gave
   each a `LANE_REVISIONS` entry. `samba` and `samba-untied-dropout-accum`
   moved to `steps-1-1` and `mamba2-dtlimit` to `seqlen-8-1`. Every other
   committed record that carries those three lanes was taken BEFORE that
   commit (`2026-09-15_cpu-samba`, 09:18; `2026-09-15_cpu-mamba`, 08:19), so
   `_verify_reference.build_table` correctly skips their cells. Rebuilding the
   table from the records on main without this file leaves all three with ZERO
   cells; with it they have one each, on the `base` fixture, backed by the CPU
   class.

2. **It is the only committed record with a `stepfull` cell.** Scanned over
   the 323 committed identity_break column files on main: one file carries a
   `stepfull_verdict` that is neither absent nor `N/A`, and before this record
   landed that count was zero. `stepfull` is the decode lanes' analogue of
   batch invariance, one fresh full pass per position against step-by-step
   decoding with a carried state, and the part had lived only in the harness
   and in a log.

## What it carries, by the JSON's own keys

Nine lanes on the `base` fixture: `transformer`, `transformer-window`,
`mamba1`, `mamba2`, `mamba2-dtlimit`, `mamba3`, `samba`,
`samba-untied-dropout-accum`, `mlp`.

| part | verdicts |
|---|---|
| `verdict` (train) | STABLE 9 |
| `infer_verdict` | STABLE 9 |
| `model_verdict` | STABLE 3, N/A 6 |
| `batch_verdict` | STABLE 9 |
| `stepfull_verdict` | STABLE 8, N/A 1 (`mlp`, no decode state) |
| `rlpair_verdict` | STABLE 8 |
| `ragged_verdict` | STABLE 8, N/A 1 |

`admit()` returns None for it and `device_class()` reads `cpu`; its `fixtures`
and `heldout` hashes equal the ones this harness produces for `base`.

## What is still owed, exactly

* The other eight fixtures for these three lanes. The shrink invalidated them
  on every column, so the next release record retakes them anyway.
* The Apple, NVIDIA and AMD columns at the shrunk fixtures. The Apple stepfull
  run this lane took (`16_apple_metal_stepfull.json`) is deliberately NOT
  recorded here: it ran from the SHARED checkout's package and Metal bindings
  while its `commit` field names the branch, which its own lane status says
  plainly. A record's commit field is load-bearing, so that column is owed as
  a rerun, not as a copy.
* `stepfull` is not in `_verify_reference.PARTS` (`train`, `infer`, `model`,
  `batch`), so `python -m mojolearn verify --all` does not compare it and this
  record's stepfull cells are not yet reachable by a user.
