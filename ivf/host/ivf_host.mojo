# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""IVF-FLAT's build and search on the host, for a box with no GPU
(lane/cpu-training-embedding-ivf, 2026-09-15; the ivf and ivf-euclidean
lanes of tools/identity_break.py).

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`,
and no GPU binding imports this file. `host_ivf_build_and_search` is
`ivf/estimator.mojo::ivf_flat_build_and_search_host` with every device launch
restated on the host, in the order `ivf_flat_build` and
`ivf_flat_search_traced` make them. The host-side pieces those two already
run on the host are CALLED, not copied: the validators of
`ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo`, the CSR layout and probe
merge of `ivf/checks/list_layout.mojo`, and `calc_chunk_indices`,
`postprocess_neighbors` and `postprocess_distances` of
`ivf/impl/neighbors/ivf_common.mojo`.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS.

  the quantizer            `kmeans_fit_main_traced` under the prefix
                           `ivf.quantizer.`, through
                           `cluster/host/kmeans_oracle.mojo::host_fit_main`
                           (the k-means lane's host restatement, which the
                           CPU gate already reads IDENTICAL x4 on nine k-means
                           lanes), with the build's own parameters:
                           `KMeansParams.default()` (k-means|| at
                           oversampling 2.0, tol 1e-4), `n_init` 1, the
                           caller's metric, `max_iter` and seed, unit
                           weights, `plan_quantizer_scale`'s sum scale
                           (`host_plan_sum_scale`, the same per-column chain)
                           and `choose_scale(n_rows, n_rows)`.
  `compute_row_norms`      `row_norm_kernel` at `take_sqrt = 0`,
                           `host_row_norms(..., False)`: the data norms, the
                           centre norms, the query norms and the list norms,
                           SQUARED under both metrics.
  `predict`                the FRESH assignment against the final centroids,
                           `host_assign` with the squared data norms and the
                           squared centre norms (`compute_centroid_norms`
                           roots only under cosine, which is refused).
  `_expanded_distances`    `pinned_distance_tile_kernel` with `is_sqrt = 0`,
                           one cell: the feature axis ascending through
                           `identical_mul_add` with the partial flushed,
                           `ftz(fma(-2, acc, ftz(ftz(qn) + ftz(yn))))`, the
                           clamp at zero.
  `_select_top_k`          `radix_topk_identical_kernel`'s answer: the `k`
                           smallest of a row under the composite key
                           `(twiddle_in(distance) << 32) | position`, in
                           rank order. Every distance here is clamped at
                           `+0.0` or above and finite (the inputs are
                           refused past 2^63), so the key's order is
                           `(distance, position)` on the floats, which is
                           what the selection below compares. Its two
                           refusals, in their words.
  `sort_slots_by_distance_then_index`
                           the search's host insertion sort, restated
                           because its file imports the device kernels.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` moves every extended
row to the next list (`host_ivf_extend`, stage 2) and walks every CANDIDATE
distance's feature axis DESCENDING (the coarse distances and the quantizer's
own arm, `kmeans_oracle`'s extra unit per quantized centroid-sum cell, which
the same define turns on, are left as they are), so the returned distances
move by the order of a float fold and the ids move wherever that reorders a
near tie. Read back by `ivf_host_sabotage`.

A fold walked in the other order is EXACT on the integer-grid `ties`
fixture: there the two arms above moved no bit of the ivf and ivf-euclidean
saved-model checks (x86 RunPod, 2026-09-15). So the same build also moves a
VALUE (lane/ties-sabotage, 2026-09-15): every distance `host_ivf_search`
returns goes through `ivf_sabotage_value_flip` after the root, so the
reported bits differ on every fixture, a root that rounds a one-unit step
back included.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the ivf and ivf-euclidean lanes is the
measurement.
"""
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.fixed_point import choose_scale
from checks.numerics import ftz, identical_mul_add
from cluster.host.kmeans_oracle import (
    DEFAULT_OVERSAMPLING,
    DEFAULT_TOL,
    INIT_KMEANS_PLUS_PLUS,
    KMeansHostTrace,
    host_assign,
    host_fit_main,
    host_metric_is_sqrt,
    host_plan_sum_scale,
    host_row_norms,
)
from ivf.checks.list_layout import (
    ListLayout,
    build_list_layout,
    extend_list_layout,
    gather_candidate_indices,
    gather_candidate_norms,
    gather_candidate_vectors,
    merge_probed_lists,
)
from ivf.impl.neighbors.ivf_common import (
    calc_chunk_indices,
    n_samples_from_chunks,
    postprocess_distances,
    postprocess_distances_is_identity,
    postprocess_neighbors,
)
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import (
    METRIC_COSINE_EXPANDED,
    IvfFlatIndexParams,
    IvfFlatSearchParams,
    ivf_index_params_validate,
    ivf_refuse_algorithm,
    ivf_search_params_validate,
    ivf_validate_data,
)


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime IVF_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


@always_inline
def ivf_sabotage_value_flip(v: Float32) -> Float32:
    """The value arm of the sabotage build: a float32 whose bits always
    differ from `v`'s. A magnitude below the smallest normal (zero or a
    subnormal) becomes the smallest positive normal; every other value steps
    its mantissa by one unit. Compiled only under IVF_HOST_SABOTAGE."""
    var bits = bitcast[DType.uint32](v)
    if (bits & UInt32(0x7FFFFFFF)) < UInt32(0x00800000):
        return bitcast[DType.float32](UInt32(0x00800000))
    return bitcast[DType.float32](bits + UInt32(1))

#: `ivf_flat_search.mojo`'s `IVF_SELECT_LIMIT` under IDENTICAL, which is
#: `neighbors/checks/select_radix_identical.mojo`'s `IDENTICAL_MAX_K`,
#: restated by value because both files import the device kernels;
#: python/mojolearn/tests/test_cpu_training_embedding_ivf.py holds the two
#: literals equal.
comptime IVF_HOST_SELECT_LIMIT = 1024


@fieldwise_init
struct IvfHostResult(Movable):
    """`IvfSearchResult`: `n_queries x k` distances and ORIGINAL row ids,
    and the candidate count per query."""

    var distances: List[Float32]
    var indices: List[UInt32]
    var n_candidates: List[Int32]


def host_pinned_distance(
    q: List[Float32],
    qi: Int,
    y: List[Float32],
    yi: Int,
    d: Int,
    q_norm: Float32,
    y_norm: Float32,
    descending: Bool,
) -> Float32:
    """`pinned_distance_tile_kernel` at `is_sqrt = 0`, one cell (module
    docstring). `descending` is the sabotage arm's walk and nothing else."""
    var acc = Float32(0.0)
    if descending:
        var f = d - 1
        while f >= 0:
            acc = ftz(identical_mul_add(ftz(q[qi * d + f]), ftz(y[yi * d + f]), acc))
            f -= 1
    else:
        for f in range(d):
            acc = ftz(identical_mul_add(ftz(q[qi * d + f]), ftz(y[yi * d + f]), acc))
    var dist = ftz(identical_mul_add(Float32(-2.0), acc, ftz(ftz(q_norm) + ftz(y_norm))))
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    return dist


def host_select_top_k(
    row: List[Float32], length: Int, k: Int
) raises -> List[UInt32]:
    """`_select_top_k` for one row (module docstring): the positions of the
    `k` smallest under `(distance, position)`, ascending."""
    if k > IVF_HOST_SELECT_LIMIT:
        raise Error(
            "ivf_flat: selection k exceeds the mode's bounded rank capacity "
            + String(IVF_HOST_SELECT_LIMIT)
        )
    if k > length:
        raise Error(
            "ivf_flat: a selection of k = "
            + String(k)
            + " over a row of "
            + String(length)
            + " elements. The implemented radix selector cannot take k > len --"
            " no bucket satisfies `prev_count < k <= cur_count`, every"
            " later pass drops every element, and last_filter reads a"
            " buffer nothing wrote (knn_brute_force.mojo's own note)."
        )
    var taken = List[Bool](length=length, fill=False)
    var out = List[UInt32](capacity=k)
    for _ in range(k):
        var best = -1
        for i in range(length):
            if taken[i]:
                continue
            if best < 0 or row[i] < row[best]:
                best = i
        taken[best] = True
        out.append(UInt32(best))
    return out^


def host_sort_slots_by_distance_then_index(
    mut dist: List[Float32], mut idx: List[UInt32], base: Int, k: Int
):
    """`ivf_flat_search.mojo::sort_slots_by_distance_then_index`, verbatim."""
    for a in range(1, k):
        var dv = dist[base + a]
        var iv = idx[base + a]
        var b = a - 1
        while b >= 0:
            var db = dist[base + b]
            var ib = idx[base + b]
            if db < dv or (db == dv and ib <= iv):
                break
            dist[base + b + 1] = db
            idx[base + b + 1] = ib
            b -= 1
        dist[base + b + 1] = dv
        idx[base + b + 1] = iv


@fieldwise_init
struct IvfHostIndex(Movable):
    """`IvfFlatIndex`'s search half on the host: the centroids, their
    squared norms and the CSR triple (`labels` is build workspace the search
    never reads)."""

    var n_lists: Int
    var dim: Int
    var n_rows: Int
    var metric: Int
    var centers: List[Float32]
    var center_norms: List[Float32]
    var offsets: List[Int32]
    var list_indices: List[UInt32]
    var list_data: List[Float32]


def host_ivf_build(
    x: List[Float32],
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    kmeans_n_iters: Int,
    metric: Int,
    seed: UInt64,
) raises -> IvfHostIndex:
    """`ivf_flat_build_host` on the host (module docstring): the quantizer,
    the squared centre norms, the fresh assignment and the CSR layout."""
    ivf_refuse_algorithm(String("ivf_flat"))
    var params = IvfFlatIndexParams.default()
    params.n_lists = n_lists
    params.kmeans_n_iters = kmeans_n_iters
    params.kmeans_trainset_fraction = Float64(1.0)
    params.metric = metric
    params.seed = seed

    ivf_index_params_validate(params, n_rows, dim)
    ivf_validate_data(x, n_rows, dim, "dataset")
    var sum_scale = host_plan_sum_scale(x, n_rows, dim)
    var weight_scale = choose_scale(Float64(n_rows), n_rows)
    var weights = List[Float32](length=n_rows, fill=Float32(1.0))
    var centers = List[Float32](length=n_lists * dim, fill=Float32(0.0))
    var labels = List[UInt32](length=n_rows, fill=UInt32(0))
    var trace = KMeansHostTrace()
    _ = host_fit_main(
        x, n_rows, dim, weights, n_lists, centers, labels,
        INIT_KMEANS_PLUS_PLUS, seed, 1, kmeans_n_iters, DEFAULT_TOL, metric,
        DEFAULT_OVERSAMPLING, Float32(sum_scale), Float32(weight_scale),
        trace, String("ivf.quantizer."),
    )
    var center_norms = host_row_norms(centers, n_lists, dim, False)
    var x_norm = host_row_norms(x, n_rows, dim, False)
    var predict_c_norm = host_row_norms(centers, n_lists, dim, metric == METRIC_COSINE_EXPANDED)
    var min_dist = List[Float32](length=n_rows, fill=Float32(0.0))
    host_assign(
        x, n_rows, x_norm, centers, n_lists, predict_c_norm, dim,
        host_metric_is_sqrt(metric), labels, min_dist,
    )
    var layout = build_list_layout(labels, x, n_rows, dim, n_lists)
    return IvfHostIndex(
        n_lists, dim, n_rows, metric, centers^, center_norms^,
        layout.offsets.copy(), layout.list_indices.copy(), layout.list_data.copy(),
    )


def host_ivf_search(
    index: IvfHostIndex,
    queries: List[Float32],
    n_queries: Int,
    k: Int,
    n_probes: Int,
    partial_storage: Bool = False,
) raises -> IvfHostResult:
    """`ivf_flat_search_host` on the host (module docstring), over a built
    index: from the build, or from arrays `ivf_validate_index_arrays` has
    admitted.

    `partial_storage` restates the device search's three partial-storage
    arms (`ivf_flat_search.mojo:577,600,640`) for one SHARD of a disjoint
    index whose coarse centres are replicated (`DistributedIVFIndex`,
    python/mojolearn/parallel_ivf.py): a shard may own FEWER than `k`
    candidates for a query, so the short-fill refusal is lifted, only
    `min(k, n_cand)` slots are selected and the remaining slots are padded
    with zeros the driver must ignore (`n_candidates` is what says how many
    are real), and the Euclidean root is NOT taken, because the global order
    is fixed on the squared keys across shards and the root is applied once,
    afterwards, by `ivf_finalize_distances`. Every other statement, in
    particular the coarse probe selection over the REPLICATED centres, is
    the plain search's."""
    var n_lists = index.n_lists
    var dim = index.dim
    var n_rows = index.n_rows
    var metric = index.metric
    var sp = IvfFlatSearchParams(n_probes)
    ivf_search_params_validate(sp, n_lists, n_queries, k)
    ivf_validate_data(queries, n_queries, dim, "queries")
    var dist_is_identity = postprocess_distances_is_identity(metric)
    if n_probes > IVF_HOST_SELECT_LIMIT:
        raise Error(
            "ivf_flat search: n_probes ("
            + String(n_probes)
            + ") exceeds the mode selection limit ("
            + String(IVF_HOST_SELECT_LIMIT)
            + "); the coarse selection runs through the same selector as"
            " the final one and inherits its refusal."
        )
    var q_norm = host_row_norms(queries, n_queries, dim, False)
    var list_norm = host_row_norms(index.list_data, n_rows, dim, False)

    # step 1 and step 2: the coarse distances and the n_probes nearest lists
    var probe_dist = List[Float32](capacity=n_queries * n_probes)
    var probe_ids = List[UInt32](capacity=n_queries * n_probes)
    for q in range(n_queries):
        var coarse = List[Float32](capacity=n_lists)
        for l in range(n_lists):
            coarse.append(host_pinned_distance(queries, q, index.centers, l, dim, q_norm[q], index.center_norms[l], False))
        var picked = host_select_top_k(coarse, n_lists, n_probes)
        for p in range(n_probes):
            probe_dist.append(coarse[Int(picked[p])])
            probe_ids.append(picked[p])
    for q in range(n_queries):
        host_sort_slots_by_distance_then_index(probe_dist, probe_ids, q * n_probes, n_probes)

    # steps 3 to 5: the candidates of each query
    var probe_layout = ListLayout(
        n_lists, n_rows, dim, index.offsets.copy(), index.list_indices.copy(),
        index.list_data.copy(),
    )
    var list_sizes = List[Int32](capacity=n_lists)
    for l in range(n_lists):
        list_sizes.append(index.offsets[l + 1] - index.offsets[l])

    var out_dist = List[Float32](capacity=n_queries * k)
    var out_idx = List[UInt32](capacity=n_queries * k)
    var cand_counts = List[Int32](capacity=n_queries)
    for q in range(n_queries):
        var this_probe = List[UInt32](capacity=n_probes)
        for p in range(n_probes):
            this_probe.append(probe_ids[q * n_probes + p])
        var chunks = calc_chunk_indices(list_sizes, this_probe, n_probes)
        var n_cand = n_samples_from_chunks(chunks, n_probes)
        cand_counts.append(Int32(n_cand))
        if n_cand < k and not partial_storage:
            raise Error(
                "ivf_flat search: query "
                + String(q)
                + " probes "
                + String(n_probes)
                + " lists holding "
                + String(n_cand)
                + " vectors between them, fewer than k = "
                + String(k)
                + ". Their kOutOfBoundsRecord short-fill"
                " (ivf_common.cuh:106-108) is not implemented. Raise n_probes,"
                " or lower k, or rebuild with fewer lists."
            )
        # Partial storage keeps the global coarse probes but may own no
        # candidates at all. Padding has no numerical meaning; `n_candidates`
        # gives min(k, count) valid slots and the driver ignores the rest
        # (`ivf_flat_search.mojo:598-606`).
        var selected = min(k, n_cand)
        if selected == 0:
            for _pad in range(k):
                out_dist.append(Float32(0))
                out_idx.append(UInt32(0))
            continue
        var slots = merge_probed_lists(probe_layout, this_probe, n_probes)
        var cand_vec = gather_candidate_vectors(probe_layout, slots)
        var cand_orig = gather_candidate_indices(probe_layout, slots)
        var cand_norm = gather_candidate_norms(slots, list_norm)
        var row = List[Float32](capacity=n_cand)
        for c in range(n_cand):
            row.append(host_pinned_distance(queries, q, cand_vec, c, dim, q_norm[q], cand_norm[c], IVF_HOST_SABOTAGE))
        var sel_pos = host_select_top_k(row, n_cand, selected)
        var sel_dist = List[Float32](capacity=selected)
        for i in range(selected):
            sel_dist.append(row[Int(sel_pos[i])])
        var sel_orig = postprocess_neighbors(sel_pos, cand_orig, selected)
        host_sort_slots_by_distance_then_index(sel_dist, sel_orig, 0, selected)
        if not dist_is_identity and not partial_storage:
            postprocess_distances(sel_dist, metric)
        for i in range(selected):
            comptime if IVF_HOST_SABOTAGE:
                # THE VALUE ARM (THE NEGATIVE CONTROL above), after the root.
                out_dist.append(ivf_sabotage_value_flip(sel_dist[i]))
            else:
                out_dist.append(sel_dist[i])
            out_idx.append(sel_orig[i])
        for _pad in range(selected, k):
            out_dist.append(Float32(0))
            out_idx.append(UInt32(0))
    return IvfHostResult(out_dist^, out_idx^, cand_counts^)


def host_ivf_build_and_search(
    x: List[Float32],
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    queries: List[Float32],
    n_queries: Int,
    k: Int,
    n_probes: Int,
    kmeans_n_iters: Int,
    metric: Int,
    seed: UInt64,
) raises -> IvfHostResult:
    """`ivf_flat_build_and_search_host` on the host (module docstring): the
    build, then the search over the index it returned. Since
    lane/inference-embedding-ivf-cholesky (2026-09-15) the two halves are
    `host_ivf_build` and `host_ivf_search`, the same statements in the same
    order, so a saved index answers what this call answers."""
    var index = host_ivf_build(x, n_rows, dim, n_lists, kmeans_n_iters, metric, seed)
    return host_ivf_search(index, queries, n_queries, k, n_probes)


def host_ivf_extend(
    index: IvfHostIndex,
    new_x: List[Float32],
    n_new: Int,
    mut new_labels: List[UInt32],
) raises -> IvfHostIndex:
    """`ivf_flat_extend_host` on the host (lane/inference-embedding-ivf-cholesky,
    2026-09-15): the build's own assignment restated (`host_assign` over the
    squared data norms against the FIXED centres, the `(distance, list id)`
    tie rule the build's `ivf.assign` stage uses), then `extend_list_layout`.
    `new_labels` is cleared and receives the list each new row went to."""
    var n_lists = index.n_lists
    var dim = index.dim
    ivf_validate_data(new_x, n_new, dim, "extension rows")
    var x_norm = host_row_norms(new_x, n_new, dim, False)
    var predict_c_norm = host_row_norms(index.centers, n_lists, dim, index.metric == METRIC_COSINE_EXPANDED)
    var labels = List[UInt32](length=n_new, fill=UInt32(0))
    var min_dist = List[Float32](length=n_new, fill=Float32(0.0))
    host_assign(
        new_x, n_new, x_norm, index.centers, n_lists, predict_c_norm, dim,
        host_metric_is_sqrt(index.metric), labels, min_dist,
    )
    comptime if IVF_HOST_SABOTAGE:
        # THE EXTEND SABOTAGE ARM (stage 2, 2026-09-15): every new row goes to
        # the NEXT list, wrong on purpose. The candidate-distance arm below
        # never reaches extend's assignment, and the first CPU leg's owed check
        # showed the new-row lists and the ties fixture's extended layout
        # unmoved under the sabotage set without this.
        for j in range(n_new):
            labels[j] = UInt32((Int(labels[j]) + 1) % n_lists)
    var layout = extend_list_layout(
        index.offsets, index.list_indices, index.list_data, index.n_rows, dim,
        n_lists, labels, new_x, n_new,
    )
    new_labels.clear()
    for j in range(n_new):
        new_labels.append(labels[j])
    return IvfHostIndex(
        n_lists, dim, index.n_rows + n_new, index.metric, index.centers.copy(),
        index.center_norms.copy(), layout.offsets.copy(), layout.list_indices.copy(),
        layout.list_data.copy(),
    )
