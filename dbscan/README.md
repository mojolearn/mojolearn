# dbscan: from cuML, with RAFT's weak_cc

Fifth section. **COPY, DO NOT IMPROVE.**

## Status: launched and passing

    check_dbscan OK: 3/3 blobs each one whole cluster, 0 merges, 12/12
      isolated points left as noise, 600 core points, converged in 2
      propagation passes
    check_dbscan_eps_sensitivity OK: eps=2 gives 3 clusters and eps=12 gives
      1, which is impossible without the neighbourhood kernel running on real
      distances

Never benchmarked.

## It was cheap because the expensive half already existed

DBSCAN's distance step is the same `core/gemm.mojo` plus
`core/expand_distances.mojo` that k-means and k-NN use. It differs from k-NN
only in what it does with the tile: a radius test instead of a top-k. The
genuinely new code is the CSR build and the label propagation.

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

## A third documented nondeterminism in these upstreams

RAFT's `adj_to_csr` says so itself: "High performance comes at the cost of
non-deterministic output: the column indices are not guaranteed to be stored
in order." Multiple blocks cooperate on one row through an atomic counter.

That joins CatBoost's float histogram atomics and RAFT's radix-select tie
handling. None of the three changes DBSCAN's answer, because label
propagation converges to the same fixed point whatever order edges are
visited in, but the pattern is worth collecting: the incumbents ship
nondeterminism wherever it buys throughput.

Ours emits in ascending order, one thread per row, which is a recorded
DEVIATION and not a fix. It gives up their multi-block cooperation on a
single wide row.
