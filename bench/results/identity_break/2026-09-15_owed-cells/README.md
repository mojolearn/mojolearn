# OWED cells in the CPU gate's four-column diff (2026-09-15)

Run on the Apple M4 on one core (a shared machine), on lane/cpu-gate-owed-cells.
The CPU column came from a fresh core host build, over the lanes kmeans,
kmeans-random, kmeans-array, kmeans-weighted, kmeans-sqrt and
kmeans-classic-pp. It used the base and odd fixtures with two repeats. The
sabotage column used `-D MOJOLEARN_HOST_SABOTAGE=1` with one repeat. The
committed GPU columns were the 166-lane record and the kmeans-sqrt fix record,
trimmed in scratch to the same two fixtures.

- `before_rec.txt` and `before_fix.txt` are from the origin/main
  identity_break. They exit 1 with 20 and 4 short cells. Every short cell is
  a k-means infer or batch part that the GPU records mark n/a:no-predict.
- `after_rec.txt` and `after_fix.txt` use `--owed-json`. They exit 0 with
  train IDENTICAL x4 on all 12 cells and OWED=20 and OWED=4. The owed lists
  are exactly the 24 cell parts that failed before.
- `owed_sabotage_check.txt`: every owed part (20 of 20 and 4 of 4) moved
  under the sabotage host set.
- `corrupt.txt`: one recorded NVIDIA train hash was corrupted, and the diff
  still exits 1 with DIVERGENT=1.
- `sab_diff.txt`: the sabotage column diffed with `--owed-json` exits 1 with
  DIVERGENT=10 and OWED=0, because a single repeat is not STABLE evidence.
