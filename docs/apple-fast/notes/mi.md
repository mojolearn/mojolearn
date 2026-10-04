# mutual information (select-mutual-info, select-mutual-info-reg): FAST on Apple, profile

Lane af-mi, 2026-10-03, branch lane/apple-fast-mi (from origin/main 8897404da). Written from the code,
no measurement. Board lanes (tools/bench_board_algos.py:619-623, AFC_FAMILY=algos): `select-mutual-info`
(mutual_info_classif, binary target: istella `y > 0`, taxi multiclass loader's target) and
`select-mutual-info-reg` (mutual_info_regression, real target: istella the relevance label 0-4, taxi the
regression target), both on SUB["mid"] = 100,000 fit rows; istella d = 220, taxi the numeric columns.

## Why the regression variant is 60x the classif time on the same data

Both estimators build the same program (python/mojolearn/_expansion_prep.py `_mutual_info`), one launch
per stage on one stream, the arena up once and back once (x_prep/device.mojo). Per fit, continuous
features only (the board's case), dc = d:

| # | stage (op) | units | what one unit does | device form today |
|---|---|---|---|---|
| 1 | col_stats (1) | d | column moments | FAST: block fold (x_prep/fastred.mojo) |
| 2 | mi_colscale (66) | d | ONE thread walks n rows: sum of abs(x / std) | the unit (serial per column) |
| 3 | mi_noise (67) | n·d | scale + splitmix noise, secondary word | the unit (parallel) |
| 4-6 | col_stats, mi_colscale, mi_noise on y (regression only) | 1, 1, n | as above; `mi_colscale` is ONE thread over n rows | the units |
| 7a | mi_cd (69), classif | n·d | Ross: k-th same-class neighbour, count within | **x_prep/dmi.mojo sorted search**: bitonic sort of every column (64-bit words), one thread per column splits the order by class, one thread per (point, column) walks outward in sorted x and counts by binary search (`_point_ties_kernel`, run-aware) |
| 7b | mi_cc (68), regression | n·d | Kraskov: k-th Chebyshev neighbour in (x_c, y), then nx, ny within | **the brute-force unit** `mi_cc_unit`: every thread scans all n rows twice (k-NN pass: 4 loads + 2 `_dsec` + insertion per pair; count pass: 4 loads + 2 `_within` per pair) |
| 8 | mi_reduce (70) | d | ONE thread sums TERM over n rows, digamma, clip | the unit (serial per column) |

Stage 7b is the gap. `mi_cc_unit` is O(n^2 d): on istella 2 x 10^10 pairs per feature x 220 features =
4.4 x 10^12 pair evaluations, each ~8 loads and a serial compare chain into an `InlineArray[Float32, 32]`
pair of k-best registers (dynamic indexing: spilled). 46.5 s FAST / 69.7 s IDENTICAL on the M3 Ultra is
that stage. The classif stage runs the sorted search at O(n log n) per feature and finishes in 783 ms
including the same sort. Nothing else differs between the two programs except the y scale/noise stages
(three tiny launches) and KIND of `mi_reduce`.

The host binding already has the exact argument for the sorted Kraskov search (x_prep/host/mutual_info.mojo
`_cc_column`, lane prep-cpu): every unit writes digamma of COUNTS; the counts are order statistics of the
distance pairs, so any search that finds the same k-th smallest pair and the same counts within it writes
the same words. On a value-sorted column the primary distance is monotone away from the point, so the
k nearest in the joint Chebyshev metric are found by walking outward in x and stopping a side once
|dx| exceeds the current k-th primary; nx and ny are binary searches on the sorted x and on one sorted y
shared by all features. The device never got that form for `mi_cc`; it got it for `mi_cd` only.

The brief's guesses, checked against the code:
- "per-feature launches / host loop over features / per-feature readback": no. One launch covers n·d
  units; the arena comes back once. FEATBATCH as named already holds, so it is not a candidate.
- "a sort per feature on host": no sort at all in the regression path; that is the problem.
- "tiled brute-force k-NN in threadgroup memory" (KNN_TILE): keeps O(n^2 d) = 4.4 x 10^12 pair
  evaluations on istella; even at a few TFLOP-equivalents that is seconds, while the sorted search is
  O(n log n d). Not pursued; the brute force stays only as the per-column fallback for a column with a
  non-finite value (the host's rule).

## Costs the sorted search must handle (why TIES is its own arm)

Taxi numeric columns include low-cardinality ones (passenger count, hour, codes): a run of equal x of
tens of thousands of points. Istella features are sparse-ish: runs of exact zeros of similar size, and
the regression target has 5 distinct values (runs of ~20k+ equal y). The host's plain walk
(`_cc_column`) visits every member of the own x-run (dx = 0 never exceeds the k-th primary), so it is
O(run) per point, O(run^2) per column: on such columns it is no better than the brute force. The
tie-aware arm sorts each column by (key(x), key(y), key(sx)) so that inside an x-run the members are in
y order (the k nearest in y are an outward walk) and inside an (x, y) stretch in sx order (the k-th
pair among equal (x, y) is a Chebyshev k-NN on the noise words, found by an outward walk in sx with the
stop rule sxd >= bs[k-1]). The counts use dmi.mojo's run counts: nx over a second copy of the column
in (key(x), key(sx)) order, ny over y in (key(y), key(sy)) order. Every distance is the unit's own
function, so the k-th pair and the counts are the unit's.

## Other serial or host-side costs in both estimators

- `mi_colscale` and `mi_reduce`: one thread per column over n rows (d threads busy, the GPU idle),
  two launches each ~n dependent-latency loads per column. FAST may fold them by threadgroup
  (x_prep/fastred.mojo's pattern): the sum order changes, pairwise is never less accurate.
- The arena download: `_Prog.run` returns every non-input arena word. Z, ZS and TERM (3·n·d words,
  265 MB on istella) are `pr.alloc`, so they cross back to the host at ~20 ms per 64 MB and the host
  zeroes them first; `pr.work` (device-only scratch, the default route elsewhere in the file) keeps them
  on the device.
- Classif `_point_ties_kernel` and the new cc point kernel map thread t = i·d + c: adjacent threads
  are adjacent COLUMNS of one row, each walking a different column's sorted array (no two threads of a
  simdgroup share a cache line). Mapping t = c·n + r (r = sorted rank) makes adjacent threads walk
  adjacent sorted positions of one column.
- dmi.mojo `_prep_kernel` (classif class split) is one thread per column with run_block batching; it
  stays as is in this lane (its share is bounded by ~5 batched passes over n per column).

## Per-fit launch and buffer counts (regression, FAST, the sorted arm on)

Launches: col_stats 1 + mi_colscale 1 + mi_noise 1 + y stages 3 + mi_cc (flags 1, loads 2-3, bitonic
~1 + 2·log2(big_n / 1024) + ... per sort, gathers 2-3, points 1) + mi_reduce 1; one synchronize inside
the mi_cc stage (its scratch is allocated and freed inside the FAST branch), one at the end. Live
buffers: the arena, the program, the sort scratch (UInt64: (2 + 2·TIES)·d·big_n + 2·big_n) and the
column scratch (UInt32: (5 + 2·TIES)·n·d + 3n + d + 1); istella: ~0.46-0.92 GB + ~0.6 GB.
