"""Host-pointer surface for IVF-FLAT: what `bindings/` will call.

**NOT YET WIRED** into `bindings/_mojolearn*.mojo` or `python/mojolearn/` --
those directories are not this lane's. `ivf/README.md`'s WHAT IS OWED names
the Python surface; this file is the entry it should reach, shaped like
`kde/estimator.mojo::kde_score_samples_host` and
`neighbors/estimator.mojo::knn_search`.

Nothing here is a port. `ivf/ported/` mirrors cuVS and is governed by COPY,
DO NOT IMPROVE; this file is host-side policy cuVS has no counterpart for,
in the same category as `mojo_only/`.

THE POLICY CHOICES
------------------

1. **`n_probes` HAS NO DEFAULT AT THIS BOUNDARY.** cuVS defaults it to 20
   (`ivf_flat.hpp:78`) and this file makes the caller say it, because
   DEVIATION 1787 makes it a NUMERIC parameter: it decides which vectors
   are summed over, so it is part of the answer. A defaulted numeric
   parameter is an answer nobody chose. `IvfFlatSearchParams.default()`
   still carries their 20 for anyone porting against their surface.

2. **`n_probes > n_lists` RAISES; THEIRS CLAMPS.**
   `ivf_flat_search.cuh:331` does `std::min(params.n_probes,
   index.n_lists())`. Under policy 1 a clamp would mean two callers who
   wrote different numbers get one answer and no card can say which
   computation ran. See `ivf_search_params_validate`.

3. **THE CARD IS PER CALL, AND A BUILD PLUS A SEARCH IS ONE CARD.**
   `core/identity_trace.mojo` requires `seq` to increase by one across a
   whole file and `tools/identity_trace_diff.py` refuses a file whose
   sequence restarts. A build and a search are two calls and would be two
   `seq 0`s, so `ivf_flat_build_and_search_host` exists and is what
   `ivf_main.mojo` and every gate use when a card is wanted. This is
   DEVIATION 1795 and it is the same shape as DEVIATION 544 for the k-NN
   classifier.

4. **THE METRIC IS L2, THE DEFAULT IS `L2Expanded`, AND THE ROOT IS THE
   CALLER'S CHOICE.** `METRIC_L2_EXPANDED` returns SQUARED distances (what
   the benchmark and the checks compare) and `METRIC_L2_SQRT_EXPANDED`
   returns Euclidean ones (what scikit-learn's `kneighbors` returns). The
   two differ by one `identical_sqrt` per returned element and `sqrt` is
   monotone, so the SET and the ORDER are the same either way -- which is
   worth writing down, because it means a recall REPORT taken under one is
   a statement about the other.

5. **NO WORKSPACE CAP AND NO QUERY BATCHING.**
   `neighbors/estimator.mojo` caps its distance tile because that tile is
   `query_tile x n_index`; the largest thing here is the candidate
   workspace at `n_rows x dim`, which is the index itself. Their own query
   batching (`ivf_flat_search.cuh:343-353`) comes from
   `get_workspace_free_bytes` and is refused for the reason DEVIATION 1798
   gives: a device memory number that decides how the query set is cut is a
   number an identical column may not read. If a shape ever needs cutting,
   the cut has to be a pure function of the shape and `check_launch_
   invariance`'s batch arm is the gate it would have to pass.
"""

from max.gpu.host import DeviceContext

from cluster.ported.cluster.kmeans_params import METRIC_L2_EXPANDED
from core.identity_trace import IdentityTrace
from ivf.ported.neighbors.ivf_flat.ivf_flat_build import ivf_flat_build
from ivf.ported.neighbors.ivf_flat.ivf_flat_index import (
    IvfFlatIndex,
    IvfFlatIndexParams,
    IvfFlatSearchParams,
    ivf_refuse_algorithm,
)
from ivf.ported.neighbors.ivf_flat.ivf_flat_search import (
    IVF_EXPAND_TPB,
    IvfSearchResult,
    ivf_flat_search_traced,
)
from neighbors.mojo_only.pinned_distance_tile import PINNED_TILE_TPB


def ivf_flat_build_host(
    ctx: DeviceContext,
    x: List[Float32],
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    kmeans_n_iters: Int = 20,
    metric: Int = METRIC_L2_EXPANDED,
    seed: UInt64 = 0,
) raises -> IvfFlatIndex:
    """`ivf_flat::build`, host list in, index out.

    **WITH `MOJOLEARN_IDENTITY_TRACE` SET, PREFER
    `ivf_flat_build_and_search_host`.** This entry constructs its own
    trace, so a build here followed by a search there appends two records
    numbered `seq 0` to one file and the differ refuses it. Policy 3.
    """
    ivf_refuse_algorithm(String("ivf_flat"))
    var params = IvfFlatIndexParams.default()
    params.n_lists = n_lists
    params.kmeans_n_iters = kmeans_n_iters
    params.kmeans_trainset_fraction = Float64(1.0)
    params.metric = metric
    params.seed = seed
    var trace = IdentityTrace()
    return ivf_flat_build(ctx, trace, params, x, n_rows, dim)


def ivf_flat_search_host(
    ctx: DeviceContext,
    index: IvfFlatIndex,
    queries: List[Float32],
    n_queries: Int,
    k: Int,
    n_probes: Int,
    tile_tpb: Int = PINNED_TILE_TPB,
    expand_tpb: Int = IVF_EXPAND_TPB,
) raises -> IvfSearchResult:
    """`ivf_flat::search`, host list in, distances and ORIGINAL ids out.

    `n_probes` is required (policy 1). See `ivf_flat_build_host` for the
    trace caveat.
    """
    var sp = IvfFlatSearchParams(n_probes)
    var trace = IdentityTrace()
    return ivf_flat_search_traced(
        ctx, trace, index, sp, queries, n_queries, k, tile_tpb, expand_tpb
    )


def ivf_flat_build_and_search_host(
    ctx: DeviceContext,
    x: List[Float32],
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    queries: List[Float32],
    n_queries: Int,
    k: Int,
    n_probes: Int,
    kmeans_n_iters: Int = 20,
    metric: Int = METRIC_L2_EXPANDED,
    seed: UInt64 = 0,
    tile_tpb: Int = PINNED_TILE_TPB,
    expand_tpb: Int = IVF_EXPAND_TPB,
) raises -> IvfSearchResult:
    """One build and one search under ONE identity card. Policy 3.

    The card's stages, in order: `ivf.quantizer.*` (the coarse k-means fit,
    written under that prefix by `kmeans_fit_main_traced`), then
    `ivf.centers`, `ivf.center_norms`, `ivf.assign`, `ivf.list_offsets`,
    `ivf.list_indices`, `ivf.list_data` from the build, then
    `ivf.query_norm`, `ivf.coarse_dist`, `ivf.probe_dist`,
    `ivf.probe_lists`, `ivf.cand_counts`, `ivf.cand_idx`, `ivf.cand_dist`,
    `ivf.out_dist`, `ivf.out_idx` from the search.

    A cross-vendor run that diverges therefore has an ADDRESS, and the
    addresses partition the way the lane's identity table does:
    `ivf.quantizer.*` is `cluster/`'s and its status is
    `UNSUPERVISED_IDENTITY.md`'s; `ivf.assign` through `ivf.list_data` is
    hazards 1, 2 and 3; `ivf.probe_lists` is hazard 1 on the query side;
    `ivf.cand_idx` is hazard 3's carry; and `ivf.out_idx` diverging while
    `ivf.out_dist` agrees is the tie class and nothing else -- which is
    exactly the split `knn.out_dist` / `knn.out_idx` was separated for.
    """
    ivf_refuse_algorithm(String("ivf_flat"))
    var params = IvfFlatIndexParams.default()
    params.n_lists = n_lists
    params.kmeans_n_iters = kmeans_n_iters
    params.kmeans_trainset_fraction = Float64(1.0)
    params.metric = metric
    params.seed = seed
    var sp = IvfFlatSearchParams(n_probes)

    var trace = IdentityTrace()
    var index = ivf_flat_build(ctx, trace, params, x, n_rows, dim)
    return ivf_flat_search_traced(
        ctx, trace, index, sp, queries, n_queries, k, tile_tpb, expand_tpb
    )
