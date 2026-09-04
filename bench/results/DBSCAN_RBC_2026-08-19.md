# The ball-cover index, measured. The 37x is gone.

> **THE sklearn COLUMN IN THIS FILE WAS MEASURED AGAINST A CRIPPLED INCUMBENT.
> CORRECTED 2026-08-19, LATE. The 2.08x is DEAD.**
>
> `DBSCAN(n_jobs=None)` is scikit-learn's default and it means **ONE CORE**.
> DBSCAN's neighbour queries are its entire cost, so every sklearn number below
> was a 10-core M4's GPU racing a single CPU core. With `n_jobs=-1`:
>
>     n         our rbc   skl 1 core   skl -1   skl gain   rbc vs skl
>     4,000         6.3         11.5     17.0      0.68x       2.71x
>     16,000       25.9         54.7     37.1      1.47x       1.43x
>     50,000      196.3        208.6     94.2      2.21x       0.48x
>     100,000     316.9        519.2    227.9      2.28x       0.72x
>     200,000     632.7      1,315.6    570.8      2.30x       0.90x
>
> **We win below roughly 20,000 points and LOSE from 50,000 up.** At 50,000 we
> are about 1.9x SLOWER, not 1.06x faster. Measured to 800,000 below: it does
> not cross back.
>
> This is the same unfairness this harness already refuses in the other
> direction when it declines to hand scikit-learn `algorithm='brute'`. A
> default that runs the incumbent on a tenth of the machine is not their
> default in any sense a user would recognise. It was caught by Andrew, not by
> the harness, and not by me.
>
> **WHAT SURVIVES UNCHANGED:** everything in the ours-against-ours column. RBC
> is 27.63x faster than our own brute force at 200,000 and wins at every size,
> which is what the default flip rested on and it never depended on sklearn.
> Nothing about DEVIATION 35 changes.
>
> **THERE IS NO CROSSOVER. Measured to 800,000, not extrapolated.**
>
>     n         our rbc ms   sklearn ms   ratio    ours vs prev   skl vs prev
>     4,000            6.7         17.3   2.57x
>     16,000          24.2         33.4   1.38x      3.60x          1.93x
>     50,000         172.9         93.1   0.54x      7.15x          2.79x
>     100,000        362.6        207.2   0.57x      2.10x          2.23x
>     200,000        796.5        523.4   0.66x      2.20x          2.53x
>     400,000      3,449.8      1,482.6   0.43x      4.33x          2.83x
>     800,000      7,930.5      4,009.6   0.51x      2.30x          2.70x
>
> The ratio sits between 0.43x and 0.66x from 50,000 to 800,000 with no trend
> toward 1. **We win below roughly 20,000 points and lose by 1.5x to 2.3x
> everywhere above it, out to the largest size measured.**
>
> **The "closing gap" I reported an hour ago was NOISE.** The previous run gave
> 0.48x / 0.72x / 0.90x at 50k / 100k / 200k; this one gives 0.54x / 0.57x /
> 0.66x at the same sizes on the same code. Three-repeat medians on this box
> are not tight enough to support a trend claim, and I made one. The right
> reading is a flat ratio around 0.5x, not a curve heading for a crossover.
>
> **One real lead, and it is not a small-n artifact.** Ours costs 4.33x going
> from 200,000 to 400,000 where scikit-learn costs 2.83x, then returns to 2.30x
> for the next doubling. A single superlinear step that does not repeat looks
> like a threshold being crossed, and the obvious candidate is
> `compute_batch_size` starting to split the fit into batches -- which adds a
> full extra neighbourhood pass per batch beyond the first
> (`runner.cuh:331-341`). Not confirmed.

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
316.9 -> 632.7 is 1.61x then 2.00x. **The rest of that row has since been
chased twice (2026-08-19, LANE_rbc-build and then the evening phase-timer
round), and both sentences that sat here in turn were wrong.** The first said
it was not worth chasing; the second said batch size was a 1.4x-2.0x lever.
The lever was measured on the three-walk runner and DIED with the two-loop
max_k landing (LANE_rbc-maxk): see "The 50,000 row, explained" below.

**At 200,000 points we went from 37x SLOWER to 2.08x FASTER.** That is a ~77x
swing, and none of it came from tuning a kernel.

RBC is now the DEFAULT (`EPS_NN_RBC`), which is a documented departure from
cuML's `BRUTE_FORCE` and is DEVIATION 35 in `dbscan/gbdt/dbscan/runner.mojo`.
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

## The 50,000 row, explained. Both of the candidates were wrong.

The two explanations offered here originally -- the landmark count `sqrt(n)`
crossing a grid-occupancy boundary, and the index build being a larger share at
that size -- are **both false, and the second is backwards.** Measured
2026-08-19, ball cover alone, warm-up discarded, min of 3, re-run in reverse
order and agreeing to 3%:

    n         L    build ms  count ms  fill ms   build share
    16,000   126     1.49      4.25     4.02       15.2%
    50,000   223     3.65     18.53    18.37        9.0%
    100,000  316     8.22     49.75    49.81        7.4%
    200,000  447    20.79    129.08   132.14        7.4%

`ball_cover.hpp:62` is `raft::sqrt(m)` and ours matches it. Cost per (query x
landmark) is monotone DECREASING through 50,000 -- 2.11 / 1.66 / 1.57 / 1.44 ns
at 16k / 50k / 100k / 200k -- with no step anywhere, so there is no occupancy
boundary. And the build's share of the ball cover FALLS through 50,000 rather
than rising. A third candidate of mine, the dead 2.03 GB `adj` allocation, is
also dead: it allocates in 0.088 ms and holding one live across the query moves
the total 0.06%.

**It is not in the ball cover at all.** The same query work at n = 50,000, split
the way DBSCAN splits it, is flat (18.50 / 19.28 / 19.35 / 20.82 ms across
1/2/2/4 batches) -- but a whole DBSCAN RBC fit at that n moves 1.4x to 2.0x
across those same shapes:

    batch size   fit ms
    40,669 (default)   150-196
    25,002             106-130
    12,502              90-112

Two batches of 40,669 and two of 25,002 are the same batch COUNT, so the axis
was batch SIZE and the ball cover was insensitive to it. The suspect named
here was `weak_cc_batched`.

**RESOLVED 2026-08-19 EVENING, and the suspect is acquitted.** The table above
was taken on the three-walk runner. After the two-loop max_k dispatch landed
(LANE_rbc-maxk) the phase timer (`dbscan/phase_main.mojo`, cuML's own nvtx
boundaries) attributed the fit at the same fixture:

    50,000, forced batches      phase-sum ms   weak_cc share
    2 (default, 80% budget)         38.5           1.3 ms
    2 (6,251 MB... 2 batches)       46.7           1.2 ms
    4 (3,126 MB)                    65.0           3.1 ms
    16,000: 1 / 2 / 8 batches       8.4 / 15.5 / 43.5

`weak_cc` never exceeds 1.3 ms on a default fit; `vertexdeg` dominates and is
RE-RUN per batch, so on the two-loop runner every added batch is a pure cost.
The old lever is INVERTED: cuML's one-big-batch 80% default is the right
policy on this device too, `max_mbytes_per_batch` stays exposed for memory
control only, and the `neigh_per_row` todo buys nothing here. The 50,000
anomaly itself largely dissolved with the two-loop landing: per-point medians
run 2.34 / 1.96 / 3.14 / 2.95 / 2.94 us at 4k / 16k / 50k / 100k / 200k, so
50,000 now sits in line with its neighbours instead of 1.4x-2.0x above them.

## What is still not ported

The index build's in-group ordering is `O(m^1.5)` where cuVS uses
`thrust::sort_by_key`. Same order as RBC's own query bound, so the asymptotics
hold. **The sentence that used to follow -- "but it will dominate the BUILD at
large m" -- is false and is deleted rather than annotated.** The rank is 13% of
the build, the build is 7.4% of the ball cover, so the rank is 0.4% of a DBSCAN
fit; the 1-nn kernel above it in the same build is 6.4x larger, and both are
`O(m^1.5)`, so the RATIO is fixed and the rank can never overtake the kernel it
sits under. On a clustered fixture (12 blobs, worst group 4.5x the mean) the
imbalance costs only 1.42x in `sum|g|^2`. A perfect radix sort buys under 1%. `nn.argsort[target="gpu"]`
cannot stand in for it — see `archive/reference/PORTING.md`, it is silently wrong above 256
elements.
