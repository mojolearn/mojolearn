# The ball-cover index, measured. The 37x is gone.

Apple M4. One window, arms INTERLEAVED inside the repeat loop (not all of one
then all of the other), 3 repeats, median reported. scikit-learn re-run in the
same window rather than compared against yesterday's numbers, because this box
drifts two- to threefold between thermal windows.

scikit-learn gets its DEFAULT `algorithm='auto'` — a kd-tree/ball-tree doing
O(n log n) queries. Forcing it to brute force would flatter us.

## DBSCAN, 8 features, eps 0.30

    n        brute ms   rbc ms   vs brute   sklearn ms   vs sklearn   was (am)
    4,000         6.7      6.3      1.07x         11.5        1.83x      0.70x
    16,000       73.2     25.9      2.82x         54.7        2.11x      0.22x
    50,000      827.5    196.3      4.21x        208.6        1.06x      0.077x
    100,000   4,113.2    316.9     12.98x        519.2        1.64x      0.047x
    200,000  17,481.3    632.7     27.63x      1,315.6        2.08x      0.027x

**RBC wins at every measured size and loses at none**, against our own brute
force AND against scikit-learn. 4,000 is INDISTINGUISHABLE against brute (the
ranges overlap) and is reported as a tie rather than as a 1.07x win.

The ours-against-ours column is the one that decides the default. The sklearn
column is a scoreboard.

## THIS TABLE IS THE SECOND MEASUREMENT. THE FIRST WAS TAKEN ON A DEFECT.

The first run gave 27.53x at 200,000 and **0.90x against scikit-learn at
50,000**, and that 50,000 row was treated as a real anomaly worth explaining.
It was not: the RBC arm was passing the WHOLE dataset as the query instead of
the batch's rows, where `algo.cuh:131` and `:146` both build the query view as
`data.x + start_vertex_id * k`. Every batch re-queried rows `0..n_points`.

It was caught by `check_dbscan_batching_agrees` -- 412 of 612 labels differed
between one batch and five -- and only because flipping the default made that
check exercise the RBC path. **The bug was in the tree, passing every other
check, for as long as RBC was opt-in.**

Total work per batch was unchanged, which is why the timings barely moved. But
a number taken on a defect is not a number, so the whole sweep was re-run.
After the fix, 50,000 goes 0.90x -> 1.06x against scikit-learn and 231.7 ->
196.3 ms, and the sublinearity between 50,000 and 100,000 that made the row
suspicious (231.7 -> 323.1 for twice the data) largely dissolves: 196.3 ->
316.9 -> 632.7 is 1.61x then 2.00x. Whatever remains at 50,000 costs a user
nothing, since RBC still beats brute 4.21x there, and it is logged rather than
chased.

**At 200,000 points we went from 37x SLOWER to 2.08x FASTER.** That is a ~77x
swing, and none of it came from tuning a kernel.

RBC is now the DEFAULT (`EPS_NN_RBC`), which is a documented departure from
cuML's `BRUTE_FORCE` and is DEVIATION 35 in `dbscan/ported/dbscan/runner.mojo`.
The argument is entirely in the ours-against-ours column: there is no n at
which brute force is the better choice for a user on this hardware, and the
labels are identical point for point, so the flip cannot change any output --
only a wait. Their design is untouched: their index, their two eps kernels,
their fallback conditions, their batch structure. Only which side of their
`if` is taken by default.

## Where each piece of it came from

Two independent changes, both of them cuML's own code:

1. **The constant.** Brute force itself went from 45,937 ms to 17,243 ms at
   200k (2.7x) when the neighborhood step stopped materializing a distance
   matrix. RAFT's `EpsUnexpL2SqNeighborhood` is a fused Contractions tile that
   keeps `acc[i][j]` in registers and writes one boolean per pair; ours had
   been a GEMM into an `m x N` float32 buffer plus two more passes over it —
   16 bytes per pair against their 1. See `LANE_dbscan-brute_2026-08-19.md`.

2. **The complexity.** `cuvs::neighbors::ball_cover` — the index cuML's DBSCAN
   calls whenever `eps_nn_method == RBC` (`vertexdeg/algo.cuh:226`). Brute is
   still `O(n^2)`; rbc prunes by the triangle inequality over landmark radii.
   That is the 27.5x at 200k and it is the term that keeps growing.

**Neither is an optimization we invented.** Both are branches of one `if` in
their source that we had not ported. `neighbors/UNPORTED.tsv` had said of
ball_cover: "Brute force first, measure where it stops winning." It stopped
winning at about 10,000 points.

## The answer does not change

`check_dbscan_rbc_matches_brute`: 612 points, **labels identical point for
point**, 3 clusters. Not "up to permutation" — `final_relabel` +
`relabelForSkl` (`runner.cuh:410-416`) make the ids canonical, so equality is
literal. The check also asserts the fixture had more than one cluster, because
an all-noise labelling would match trivially.

That is the property that matters: an index which drops a real neighbour is
SILENT. The point still gets a label, just possibly its own cluster instead of
its neighbour's. Counting clusters cannot see it; comparing the partition
point by point can.

## The 50,000 row, as it stood before the re-run

0.90x is the one size where rbc loses to scikit-learn, and it sits between two
sizes where rbc wins. It is not a thermal artifact — the arms were interleaved
inside the same loop. Candidates: the landmark count `sqrt(n)` crossing a
grid-occupancy boundary, or the index build (`O(m^1.5)` in our port, a radix
sort in theirs) being a larger share at that size. **Not investigated. Not
explained away.**

## What is still not ported

The index build's in-group ordering is `O(m^1.5)` where cuVS uses
`thrust::sort_by_key`. Same order as RBC's own query bound, so the asymptotics
hold, but it will dominate the BUILD at large m. `nn.argsort[target="gpu"]`
cannot stand in for it — see `PORTING.md`, it is silently wrong above 256
elements.
