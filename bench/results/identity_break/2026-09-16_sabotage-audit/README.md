# Sabotage audit, the estimators and core CPU columns (2026-09-16)

Branch `lane/sabotage-audit`, commit bfb8f725a. Eleven lanes whose host
sabotage arm existed but which no committed column had ever been seen to move
were run on this Mac's CPU host route, one core, and watched to fail. No box
was rented and no Metal job ran (the 0.8.6 Apple record held the Metal slot
throughout).

## How it ran

Apple M4, one core (`nice -n 19`, thread knobs 1, `MOJOLEARN_BUILD_JOBS=1`,
one process at a time), in a worktree of its own. The `estimators` and `core`
host families were built twice from the same tree, production and
`-D MOJOLEARN_HOST_SABOTAGE=1`, into two fresh directories. The four binaries
have four different sha256 values (`so_sha256.txt`), so the sabotage column
did not load a production binding. Build wall time was 85, 75, 62 and 91
seconds.

| file | what |
|---|---|
| `cpu-prod.json` | the production CPU column, 11 lanes, fixtures base and ties, 2 repeats |
| `cpu-sab.json` | the same lanes through the sabotage builds, 1 repeat |
| `sabotage_moves.txt` | every cell part, production against sabotage |
| `so_sha256.txt` | the four host binaries |
| `cpu-prod.log`, `cpu-sab.log` | the harness output |
| `build.*.log` | the four builds |
| `lane_table.md` | the per-lane audit table (also in `docs/lanes/SABOTAGE_AUDIT_2026-09-16.md`) |

Both columns record `host.column = cpu`, `cpu_model = Apple M4`, commit
bfb8f725ab2b30a94e8596b407775095e37c2508, mode identical, 22 cells, complete.
`cpu-prod.json` records both families with `sabotage: false` and `cpu-sab.json`
records both with `sabotage: true`, which is the column's own witness of which
arm it ran.

## What moved

79 cell parts moved, 9 did not (`sabotage_moves.txt`).

| lane | train | infer | model | batch |
|---|---|---|---|---|
| pca, pca-whiten, ols, ridge, logistic, logistic-multiclass | MOVED | MOVED | MOVED | MOVED |
| tsvd | MOVED | MOVED | MOVED on base, UNMOVED on ties | MOVED |
| kde, knn, knn-clf, knn-reg | MOVED | MOVED | UNMOVED on both fixtures | MOVED |

The production column is 22 of 22 cells stable, with infer, model and batch
stable as well, so the arms above moved a column that was otherwise steady.

## The nine unmoved parts

Eight of them are the `model` part of kde, knn, knn-clf and knn-reg on both
fixtures. That part is `identity_break._hfile(path)`, the sha256 of the saved
file, and for these estimators the file holds the fitted index (the caller's
own rows as fitted) plus scalars. No arithmetic result enters it, and both
host arms perturb distances computed at query time, so no host sabotage arm
can reach it. This is the same shape as the open radius question and is
written up in `docs/lanes/SABOTAGE_AUDIT_2026-09-16.md`.

The ninth is `tsvd/ties model`, which is different. The base fixture moves and
the integer `ties` fixture does not, so that arm is fixture inert rather than
structurally unreachable.
