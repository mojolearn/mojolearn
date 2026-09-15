# HDBSCAN: dense graph rows

Design note for `mojolearn.parallel_classical.fit_hdbscan`. IDENTICAL mode
only; a multi-device fit must have the exact bits of the one-device fit.

## Where the work is

`hdbscan/impl/runner.mojo::fit_hdbscan` with the dense mutual-reachability
graph (DEVIATION 1600) runs, in order:

1. Core distances: a brute k-NN of `X` against itself through
   `neighbors/estimator.mojo::knn_search_traced` (`min_samples` neighbors per
   query row), then the `min_samples`-th distance per row.
2. The `m x m` pairwise L2 distance matrix through
   `hierarchy/impl/cluster/detail/connectivities.mojo::pairwise_distances`.
3. Mutual reachability per cell, `max(core_i, core_j, d_ij / alpha)`.
4. The Boruvka MST over the dense graph, the dendrogram, the condensed
   hierarchy, stabilities and cluster selection.

Steps 1 and 2 are per query row: a k-NN row reads one query and the whole
index, and a distance row reads one row, every other row and the norms. Both
already have row drivers qualified for other estimators. The k-NN reads
`MOJOLEARN_NEIGHBORS_DEVICE_COUNT` (`neighbors/impl/multi_gpu.mojo`, used by
`ParallelQueries` and the graph driver) and the distance matrix reads
`MOJOLEARN_HIERARCHY_DEVICE_COUNT` (`hierarchy/impl/cluster/detail/
multi_gpu.mojo`, used by `AgglomerativeClustering` through `fit_graph`). Each
owner runs the original pinned tile kernel for its rows against the full
replicated reference and copies its rows back as bytes. Step 3 is one cell
at a time on the root. Step 4 has data-dependent control flow (component
merges, ties broken on the edge order of `hierarchy/checks/edge_order.mojo`)
and stays on the root unchanged; it is not partitioned.

So the driver adds no new arithmetic path: it runs `HDBSCAN.fit` in a
cooperative worker whose pool sets both switches, behind a
`hdbscan_rows_parallel_available` flag.

## Reach, not only output

Equal outputs do not show the rows moved. A check-only build of the HDBSCAN
binding with `-D MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE=1` makes later owners
of the distance rows read their query rows one row early; the public check
must fail against it.

## Gates

`tools/parallel_hdbscan_check.py` fits five configurations (5 to 1024 rows,
`eom` and `leaf`, `alpha`, `allow_single_cluster`, a duplicated point) on one
device in-process and through `fit_hdbscan`, with identity traces on both, and
requires every trace record (input, core distances, every mutual reachability
cell, MST, condensed tree, stabilities, selection, labels) and every fitted
attribute to be equal. The native row gates of the neighbors and hierarchy
drivers (`training/checks/graph_rows_check.mojo`) cover the kernels.

The root keeps `X`, the `m x m` graph and the tree. No speed or capacity
claim; the sparse k-NN graph arm is not implemented in this lane at all.
