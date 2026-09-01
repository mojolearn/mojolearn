# dbscan: from cuML, with RAFT's neighborhood, weak_cc and merge_labels

Fifth section. **COPY, DO NOT IMPROVE.**

## Status: rebuilt against the upstream checkout, 2026-08-19

Everything below was written the first time from a recollection of cuML's
code. `upstream/cuml`, `upstream/raft` and `upstream/cuvs` landed on disk on
2026-08-19 and six divergences fell out of one read. All six are fixed.

    check_fused_eps_agrees_with_materialized OK: 31950 cells identical to the
      materialized path AND to a float64 host oracle (15298 of them
      neighbours, so the pattern is irregular), 150 degrees identical,
      vd[m] = 15298
    check_dbscan OK: 3/3 blobs each one whole cluster with ids exactly
      {0, 1, 2}, 0 merges, 12/12 isolated points labelled -1 as scikit-learn
      does, converged in 2 propagation passes
    check_dbscan_eps_sensitivity OK: eps=2 gives 3 clusters and eps=12 gives 1
    check_exclusive_scan_beyond_the_old_cap OK: 2000000 entries exact across
      977 blocks, total 12000001
    check_dbscan_batching_agrees OK: 5 batches with 4 merge_labels folds give
      labels identical to one batch, in an adj buffer 4x smaller than the
      unbatched run needs

## Two refused configurations closed, 2026-09-01

`sample_weight` and `metric='manhattan'` were both refused at the Python
surface. Both are now served, and one refusal is DELIBERATELY KEPT.

### `sample_weight` -- PORTED on both arms, and GATED SINCE 2026-09-01

cuML has it and we did not. It enters in EXACTLY ONE PLACE: a point is core
when the SUM OF WEIGHTS in its eps-neighborhood reaches `min_samples` rather
than when the COUNT does. `runner.cuh:300-306` is the whole change and
scikit-learn's `_dbscan.py:451-455` is the same sentence. The neighborhood,
the adjacency, the CSR, `weak_cc`, `MergeLabels` and both relabels are
byte-for-byte the unweighted path.

Both producers of the weighted degree are ported, because `launcher` has two:
`coalescedReduction` over the dense `adj` (`algo.cuh:243-254`) for the brute
arm, and `accumulateWeights` over the CSR (`algo.cuh:62-91`) for the ball
cover, with `need_ja_compute`'s `sample_weight` disjunct (`runner.cuh:257`)
so loop 1 fills `ja` on every weighted RBC batch (DEVIATION 29).

**THE NUMERIC RISK IS THE FOLD, AND ITS GATE IS WRITTEN BUT HAS NEVER
RUN** (`check_dbscan_weighted_fold_is_pinned`; see the gate list below for
why). Both of their
reducers close on a CUB stage that folds at the HARDWARE warp width, 32 on
Apple and NVIDIA and 64 on a CDNA wavefront. A weighted degree is a FLOAT
sum, so a faithful port would put an AMD fit and a CUDA fit on opposite sides
of `>= min_samples` for any point sitting on the threshold, which is a
different MODEL and not a last-bit difference in a reported number. Both
kernels therefore close on `core/pinned_reduce.pinned_block_sum` at
`K_LIB_WEIGHTED_VERTEX_DEG`'s width, which the kernel matrix now lists in
`lib_block_bounds_a_float_fold` (DEVIATION 28).

### `metric='manhattan'` -- ADDED, brute arm, and it is ORIGINAL WORK

There is no upstream to port. cuML's DBSCAN offers euclidean, cosine and
precomputed (`dbscan.pyx:110-115`) and RAFT has no L1 eps-neighborhood
kernel, so no `DERIVATION_MAP.tsv` row points anywhere for this arm. The one
borrowed line is the per-pair op, RAFT's
`distance/detail/distance_ops/l1.cuh:49`, cited in the kernel.

**THE EPS COMPARISON IS SQUARED AND THAT FORCED THE DESIGN.** The ported arm
never takes a square root: it accumulates `sum (x-y)^2` and compares against
`eps * eps`, squared once on the host at `algo.cuh:225`. **An L1 sum has no
squared form.** `sum |x-y|` is already the distance and there is no monotone
rewriting that lets it meet an `eps^2`; comparing against `eps * eps` is not
a slower answer, it is a different radius. So the kernel's threshold argument
stopped being "eps squared" and became `thresh`, whose meaning the comptime
`metric` parameter chooses, and `dbscan_metric_threshold` is the single
function both the host launcher and the checks call. The alternative -- a
runtime metric flag -- was rejected because it would put a branch in the
innermost accumulate loop AND leave the host's squaring a separate decision
from the kernel's arithmetic. DEVIATION 27.

On the BALL COVER the metric is refused by name rather than downgraded, and
that is a scope boundary rather than a property of the algorithm: ball-cover
pruning rests on the triangle inequality, which L1 satisfies, but
`neighbors/impl/neighbors/ball_cover/` computes Euclidean distances for its
landmark radii and its three bounds and is another lane's file.

### `algorithm='kd_tree'` -- STILL REFUSED, on the merits

`algorithm` names the eps-neighborhood SEARCH. A kd-tree eps query is a
recursive, data-dependent descent with a per-thread stack: on a GPU the lanes
of a warp diverge at every node, the traversal serializes, and the loads are
pointer chases rather than coalesced reads -- a structure whose whole
advantage is skipping work, on hardware where a warp only skips work every
lane agrees to skip. Its pruning also decays with dimension and degenerates
into a scan with overhead past roughly ten features, which is why
scikit-learn's own `'auto'` abandons it there.

The RANDOM BALL COVER is the structure that does work on this hardware for
this query: two flat arrays and a triangle-inequality bound, `sqrt(n)`
landmark distances per query in one coalesced pass, whole landmark groups
pruned arithmetically, survivors scanned contiguously. No stack, no divergent
descent. It measured 2.7x-27x over brute force at 16k-200k rows (DEVIATION
35) and `check_dbscan_rbc_matches_brute` holds the two labellings identical
point for point.

So `'kd_tree'` is refused because it is a WORSE STRUCTURE for this query on
this hardware, not because anything upstream lacks one. Offering it would
hand a caller a slower answer under a familiar name.

### The gates, and the sabotage arm each carries

**ALL SEVEN RUN AND PASS as of 2026-09-01. FOUR OF THEM HAD NEVER RUN
BEFORE THAT MORNING, and they are the four weighted ones.** Building them
raised `DeadArgumentElimination surveyUse failed`, an LLVM pass assertion,
and took the whole dbscan build down; the entry point had to comment out
both their calls and their imports, because the import is what pulls a
function into codegen. So `sample_weight` was IMPLEMENTED AND UNGATED from
the day it landed.

**THE CURE IS THE OPTIMIZATION LEVEL, NOT THIS LANE'S SOURCE.** `pixi run
check-dbscan` passes `-O1`. Measured on an Apple M4, one variable: -O3 and
-O2 both assert, -O1 and -O0 both build, and `DeadArgumentElimination` is an
-O2-and-above pass. Four candidate source rewrites were tried FIRST, on the
theory that some construct in the gate surveyed badly, and every one of them
still asserted at -O3; one was reverted afterwards because it had taken
`vertex_deg_dispatch` out from under the gate for no benefit.

**IT IS NOT A WEAKENED GATE, and that is measured rather than argued.** With
the weighted four disabled so that -O3 can build at all, the -O1 and the -O3
binaries print BYTE-IDENTICAL output across all 13 remaining gates,
including the float-heavy `check_fused_eps_agrees_with_materialized` (31,950
cells against a float64 host oracle) and `check_dbscan_manhattan_
neighborhood` (22,500). The level is also only the HOST binary's; the Metal
kernels are compiled by their own path.

The bisect is worth keeping because the handle is reusable: enabling the
gates one at a time showed TWO independent triggers, the fold gate alone and
the three fit gates alone. **Apple M4 only. A three-vendor leg is owed and
nothing weighted has run on an H100 or an MI325X.**

    check_dbscan_weighted_fold_is_pinned          the matrix row
      is NUMERIC; MEASURED 1.0000151 at 128, 1.000015 at 64, 1.0 left to
      right; sabotage: the same values folded at WVD_TPB and at half it
      must differ (1.0 and a run of 2^-24 -- exactly 1.0 left to right,
      strictly greater through the tree)
    check_dbscan_manhattan_neighborhood           every cell against a
      float64 host oracle; sabotage A: the L1 kernel run against a SQUARED
      threshold must give a different adjacency; sabotage B: the L2 kernel
      run against the L1 threshold likewise
    check_dbscan_manhattan_changes_the_labels     one cluster under euclidean
      and two under manhattan on a planted axis-aligned fixture whose only
      cross edge is a diagonal step at L2 0.98995 and L1 1.4, eps 1
    check_dbscan_manhattan_refused_on_the_ball_cover   rbc raises; sabotage:
      the same call on brute must SUCCEED
    check_dbscan_weighted_degree_matches_host_oracle   37 rows, 2231
      edges, both
      kernels against their own pinned host folds, bit for bit under
      IDENTICAL; sabotage: a planted mask on which the two arms' folds
      provably separate, and a planted weight vector on which the pinned
      fold and a left-to-right sum provably separate
    check_dbscan_uniform_weight_matches_unweighted     612 labels, weights of
      1.0 reproduce the unweighted labels exactly on BOTH arms; sabotage:
      weight `min_samples` on the twelve isolated points must turn all
      twelve core
    check_dbscan_duplicate_equals_weight_two      duplicating a
      point equals giving it weight 2, on both arms; sabotage: weight 1.5
      must fall back to noise (1.5 + 1 = 2.5 < 3)

## The one number that explains the rebuild

At 8 features and `eps = 0.30`, measured 2026-08-19 against scikit-learn:

    n         ours(ms)   sklearn(ms)  ratio
    4,000        19.03        13.30   0.70x
    16,000      274.64        59.93   0.22x
    50,000     2666.20       206.62   0.077x
    100,000   10523.76       499.56   0.021x
    200,000   45937.13      1237.99   0.027x

Exactly quadratic, and 37x behind at 200,000. The complexity is inherent to
brute force and is a different lane's problem (the ball-cover index). The
CONSTANT was ours, and it was three separate departures from `runner.cuh`
stacked on top of each other. See the lane file
`bench/results/LANE_dbscan-brute_2026-08-19.md` for the divergence table.

## It was NOT cheap because the expensive half already existed

That sentence used to be here and it was the root of the problem. The claim
was that DBSCAN's distance step is "the same `core/gemm.mojo` plus
`core/expand_distances.mojo` that k-means and k-NN use", differing from k-NN
"only in what it does with the tile".

cuML does not do that. `vertexdeg/algo.cuh:229` calls
`raft::spatial::knn::detail::EpsUnexpL2SqNeighborhood`, a fused Contractions
tile kernel that keeps `acc[i][j]` in registers, applies the radius test in
its `epilog()`, and writes one BOOLEAN per pair plus the degree counts. No
GEMM, no distance matrix, no norms. Reusing the k-NN pipeline turned one
kernel and one byte per pair into three kernels and sixteen.

The fused kernel is now at
`dbscan/impl/neighbors/epsilon_neighborhood.mojo`. The materialized path
survives ONLY as `vertexdeg/algo.mojo::eps_neighborhood_kernel`, which
nothing in `gbdt/` calls: it is the reference the fused kernel is diffed
against, cell by cell, the same role `gemm_nt_kernel` used to play for the
vendor matmul.

## Their L2 is UNEXPANDED

`accumulate()` is `diff = x - y; acc += diff * diff`, straight from the
coordinates. The old path used `||x||^2 + ||y||^2 - 2 x.y` so the cross term
could go through a GEMM. That is an arithmetic change, not a routing one:
when the norms dominate the distances the subtraction cancels in float32.
This tree paid for that once already (`PORTING.md 21`) and the DBSCAN fixture
still carries a comment about keeping coordinates small to dodge it. It does
not need to any more.

## The one thing to understand before changing anything here

**Where the core-point restriction is applied.** It is NOT in the graph. The
CSR contains every edge, including edges out of border points, and the
restriction lives in `weak_cc`'s `filter_op`: a core point may propagate a
label, a border point may receive one but never pass it on.

I got this wrong first time and put a core filter in the scan and the
compaction. It reads like an optimization and it silently changes the answer,
because the labeler still needs to see border edges in order to attach
borders to clusters. The fix was to read `adjgraph/algo.cuh` instead of
describing it from memory: their `thrust::exclusive_scan` runs over the whole
degree array with no mask at all.

## Batching is per-batch LABELLING, not just per-batch distance

`runner.cuh` runs two loops over the batches. The first, in REVERSED order so
batch 0 stays resident, builds each batch's adjacency and degrees and fills in
the core mask. The second labels each batch's sub-graph with
`weak_cc_batched` and folds the result into the running labelling with
`MergeLabels::run`. Their comment at `runner.cuh:392-395` says why the fold
cannot be skipped: seeding `weak_cc` with the previous batch's labels gives
wrong answers, cuML issue #3094.

The previous version here batched only the DISTANCES and kept one `weak_cc`
over a CSR built from every row of the dataset. Correct, and it gave the
memory straight back: a global CSR is the whole adjacency in sparse form.
`col_ind` is now allocated for the largest single batch, which is what
`runner.cuh:317` does.

## Labels now match scikit-learn's

`final_relabel` (monotonic `0..k-1`) and `relabelForSkl` (`MAX_LABEL` becomes
`-1`, everything else loses one) are both ported. `NOT_IMPLEMENTED.tsv` used to
excuse skipping them with "labels are arbitrary up to permutation, so the
check compares the PARTITION". That is true of the check and false of the
API: cuML runs both on every fit, so anyone diffing our labels against cuML's
or sklearn's got different numbers for the same clustering.

## A third documented nondeterminism in these upstreams

RAFT's `adj_to_csr` says so itself: "High performance comes at the cost of
non-deterministic output: the column indices are not guaranteed to be stored
in order." Multiple blocks cooperate on one row through an atomic counter.

That joins CatBoost's float histogram atomics and RAFT's radix-select tie
handling. None of the three changes DBSCAN's answer, because label
propagation converges to the same fixed point whatever order edges are
visited in, but the pattern is worth collecting: the incumbents ship
nondeterminism wherever it buys throughput.

Ours used to emit in ascending order, using a block prefix sum per
256-column window, and that was defended here as worth more than "their last
increment of throughput". It was not: at 200,000 columns it ran 782 block
scans and 1,564 barriers per row against their zero barriers. The compaction
is now theirs -- shared per-row cursor, atomic bump, 16-bool chunk loads,
unordered output. What is still not theirs is the WARP AGGREGATION of that
atomic (`cg::coalesced_threads()`, which Mojo has no counterpart for) and the
multi-block-per-row grid, both priced in deviation 34.
