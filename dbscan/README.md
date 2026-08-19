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
`dbscan/ported/neighbors/epsilon_neighborhood.mojo`. The materialized path
survives ONLY as `vertexdeg/algo.mojo::eps_neighborhood_kernel`, which
nothing in `ported/` calls: it is the reference the fused kernel is diffed
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
`-1`, everything else loses one) are both ported. `UNPORTED.tsv` used to
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
