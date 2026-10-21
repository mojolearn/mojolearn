# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed bodies of the core host binding (`bindings/_mojolearn_core_host.mojo`).

The binding keeps the Python door: it unpacks PythonObject arguments, calls
one function here under `GILReleased` and converts the typed result back.
Every statement that reads, compares, copies or folds data lives here with
no Python import, word for word as it was in the binding (lane thin-core,
2026-10-21: a pure move, same operations in the same order, no bit change).
"""
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32, u32_ptr
from neighbors.impl.ball_cover.common import rbc_validate_metric
from neighbors.impl.ball_cover.knn import RBC_KNN_MAX_K
from cluster.host.kmeans_oracle import (
    INIT_ARRAY,
    host_kmeans_fit,
    host_kmeans_predict,
    host_kmeans_transform,
    host_kmeans_validate,
)
from neighbors.impl.distance.detail.distance_ops import DIST_INNER_PRODUCT
from core.knn_host_predict import (
    host_rbc_radius_counts,
    host_rbc_radius_fill_rows,
    host_rbc_knn_search,
    KNN_HOST_WEIGHTS_DISTANCE,
    KNN_HOST_WEIGHTS_UNIFORM,
    host_class_probs,
    host_class_vote,
    host_distance_weights,
    host_knn_search,
    host_monotonic,
    host_regress_avg,
    host_unique_labels,
    host_weighted_class_probs,
    host_weighted_regress_avg,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]


def transpose_to_f32[
    S: DType
](
    sp: MutPointer[Scalar[S], MutUntrackedOrigin],
    dp: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int,
    nc: Int,
):
    """`dp[c * nr + r] = Float32(sp[r * nc + c])`, the value
    `_tiled_transpose_to_f32` writes, in plain column order."""
    for c in range(nc):
        var dbase = c * nr
        for r in range(nr):
            dp.unsafe_store(dbase + r, sp.unsafe_load(r * nc + c).cast[DType.float32]())


def nonzero_f64_count(sp: _F64P, count: Int) -> Int:
    """How many of the `count` float64 values at `sp` satisfy `v != 0.0`."""
    var nz = 0
    for i in range(count):
        if sp.unsafe_load(i) != 0.0:
            nz += 1
    return nz


def nonzero_f64_fill(
    sp: _F64P, nr: Int, nc: Int, rp: _I32P, cp: _I32P, vp: _F32P, cap: Int
) raises -> Int:
    """COO triples of every `v != 0.0` of a row-major float64 `[nr, nc]`
    matrix, row-major scan order; raises past `cap` nonzeros."""
    var k = 0
    var overflow = False
    for r in range(nr):
        var base = r * nc
        for c in range(nc):
            var v = sp.unsafe_load(base + c)
            if v != 0.0:
                if k >= cap:
                    overflow = True
                    break
                rp.unsafe_store(k, Int32(r))
                cp.unsafe_store(k, Int32(c))
                vp.unsafe_store(k, v.cast[DType.float32]())
                k += 1
        if overflow:
            break
    if overflow:
        raise Error(
            "nonzero_f64_fill: more than " + String(cap)
            + " nonzero entries; count first with nonzero_f64_count"
        )
    return k


def knn_search_host(
    index_address: Int,
    queries_address: Int,
    dp: _F32P,
    xp: _U32P,
    ni: Int,
    nq: Int,
    nf: Int,
    kk: Int,
    sq: Bool,
    metric: Int,
    metric_arg: Float32,
) raises:
    """Exact k-NN on the host: `nq x kk` float32 distances to `dp` and
    uint32 indices to `xp`, ascending by (distance, index) per row."""
    if ni < 0 or nq < 0 or nf < 0 or kk < 0:
        raise Error("knn_search: a negative dimension was passed")
    var index = read_f32(index_address, ni * nf)
    var queries = read_f32(queries_address, nq * nf)
    var out_dist = List[Float32](length=max(0, nq * kk), fill=Float32(0.0))
    var out_idx = List[UInt32](length=max(0, nq * kk), fill=UInt32(0))
    host_knn_search(
        index, ni, queries, nq, nf, kk, metric, sq, out_dist, out_idx,
        metric_arg,
    )
    # the inner-product search selected the NEGATED products: hand back
    # the products (exact), as the GPU binding's device negation (lane
    # pyglue-numeric: the Python wrapper negated them)
    var neg = metric == DIST_INNER_PRODUCT
    for i in range(nq * kk):
        dp[i] = -out_dist[i] if neg else out_dist[i]
        xp[i] = out_idx[i]


def knn_host_weighted(weights: Int, who: String) raises -> Bool:
    """`knn_classifier_predict` / `knn_regressor_predict`'s weights check,
    same message."""
    var weighted = weights == KNN_HOST_WEIGHTS_DISTANCE
    if weights != KNN_HOST_WEIGHTS_UNIFORM and not weighted:
        raise Error(
            who
            + ": weights value "
            + String(weights)
            + " is neither WEIGHTS_UNIFORM nor WEIGHTS_DISTANCE"
        )
    return weighted


def knn_classify_host(
    index_address: Int,
    queries_address: Int,
    y_address: Int,
    lp: _I32P,
    pp: _F32P,
    up: _I32P,
    ni: Int,
    nq: Int,
    nf: Int,
    kk: Int,
    no: Int,
    want_proba: Bool,
    n_classes: List[Int],
    metric: Int,
    metric_arg: Float32,
    weights: Int,
) raises:
    """The k-NN classifier on the host, a search then a vote or a tally
    (`knn_classifier_predict`)."""
    if no < 1:
        raise Error(
            "knn_classifier_predict: n_outputs must be positive, got "
            + String(no)
        )
    var weighted = knn_host_weighted(weights, "knn_classifier_predict")
    if ni < 0 or nq < 0 or nf < 0 or kk < 0:
        raise Error("knn_classify: a negative dimension was passed")
    var index = read_f32(index_address, ni * nf)
    var queries = read_f32(queries_address, nq * nf)
    var dist = List[Float32](length=max(0, nq * kk), fill=Float32(0.0))
    var idx = List[UInt32](length=max(0, nq * kk), fill=UInt32(0))
    # policy 8: the weighted arm needs the ROOTED distance
    host_knn_search(index, ni, queries, nq, nf, kk, metric, weighted, dist, idx, metric_arg)
    var w = List[Float32]()
    if weighted:
        w = host_distance_weights(dist, nq, kk)
    # The unique sets first, checked against the caller's counts
    # before a byte is written (`_check_class_counts`).
    var ys = List[List[Int32]]()
    var uniqs = List[List[Int32]]()
    for i in range(no):
        ys.append(read_i32(y_address + i * ni * 4, ni))
        uniqs.append(host_unique_labels(ys[i], ni))
        if len(uniqs[i]) != n_classes[i]:
            raise Error(
                "knn_classifier_predict: the implemented getUniquelabels found "
                + String(len(uniqs[i]))
                + " classes for output "
                + String(i)
                + " and the caller sized its buffers for "
                + String(n_classes[i])
                + "; one of the two class sets is wrong and nothing is "
                + "written"
            )
    var labels_out = List[Int32](length=max(0, nq * no), fill=Int32(0))
    var off = 0
    for i in range(no):
        var n_uniq = len(uniqs[i])
        var mono = host_monotonic(ys[i], ni, uniqs[i])
        var proba: List[Float32]
        if weighted:
            proba = host_weighted_class_probs(idx, mono, w, n_uniq, nq, kk)
        else:
            proba = host_class_probs(idx, mono, n_uniq, nq, kk)
        if want_proba:
            for j in range(nq * n_uniq):
                pp[off + j] = proba[j]
            off += nq * n_uniq
        else:
            host_class_vote(proba, uniqs[i], n_uniq, nq, labels_out, no, i)
    if not want_proba:
        for j in range(nq * no):
            lp[j] = labels_out[j]
    var uoff = 0
    for i in range(no):
        for j in range(len(uniqs[i])):
            up[uoff + j] = uniqs[i][j]
        uoff += len(uniqs[i])


def knn_regress_host(
    index_address: Int,
    queries_address: Int,
    y_address: Int,
    op: _F32P,
    ni: Int,
    nq: Int,
    nf: Int,
    kk: Int,
    no: Int,
    metric: Int,
    metric_arg: Float32,
    weights: Int,
) raises:
    """The k-NN regressor on the host, a search then the neighbours' mean
    (`knn_regressor_predict`)."""
    if no < 1:
        raise Error(
            "knn_regressor_predict: n_outputs must be positive, got "
            + String(no)
        )
    var weighted = knn_host_weighted(weights, "knn_regressor_predict")
    if ni < 0 or nq < 0 or nf < 0 or kk < 0:
        raise Error("knn_regress: a negative dimension was passed")
    var index = read_f32(index_address, ni * nf)
    var queries = read_f32(queries_address, nq * nf)
    var dist = List[Float32](length=max(0, nq * kk), fill=Float32(0.0))
    var idx = List[UInt32](length=max(0, nq * kk), fill=UInt32(0))
    host_knn_search(index, ni, queries, nq, nf, kk, metric, weighted, dist, idx, metric_arg)
    var w = List[Float32]()
    if weighted:
        w = host_distance_weights(dist, nq, kk)
    var out = List[Float32](length=max(0, nq * no), fill=Float32(0.0))
    for i in range(no):
        var y = read_f32(y_address + i * ni * 4, ni)
        if weighted:
            host_weighted_regress_avg(idx, y, w, nq, kk, out, no, i)
        else:
            host_regress_avg(idx, y, nq, kk, out, no, i)
    for j in range(nq * no):
        op[j] = out[j]


def host_validate_neighbors(
    idx: List[UInt32], n_index: Int, n_queries: Int, k: Int,
    n_outputs: Int, weights: Int,
) raises:
    """`neighbors/estimator.mojo::_validate_neighbors`, its words."""
    if n_index < 1 or n_queries < 1 or k < 1 or k > n_index or n_outputs < 1:
        raise Error("precomputed neighbors: invalid shape")
    if weights != KNN_HOST_WEIGHTS_UNIFORM and weights != KNN_HOST_WEIGHTS_DISTANCE:
        raise Error("precomputed neighbors: unsupported weights")
    for i in range(n_queries * k):
        if UInt64(idx[i]) >= UInt64(n_index):
            raise Error("precomputed neighbor index outside reference data")


def knn_classify_neighbors_host(
    dist_address: Int,
    idx_address: Int,
    y_address: Int,
    lp: _I32P,
    pp: _F32P,
    up: _I32P,
    ni: Int,
    nq: Int,
    kk: Int,
    no: Int,
    want_proba: Bool,
    n_classes: List[Int],
    weights: Int,
) raises:
    """Vote on globally merged neighbors: float32 distances and uint32
    global indices, both `nq x kk`."""
    if ni < 1 or nq < 1 or kk < 1 or kk > ni or no < 1:
        raise Error("precomputed neighbors: invalid shape")
    var xp = u32_ptr(idx_address)
    var idx = List[UInt32](length=nq * kk, fill=UInt32(0))
    for i in range(nq * kk):
        idx[i] = xp[i]
    host_validate_neighbors(idx, ni, nq, kk, no, weights)
    if len(n_classes) != no:
        raise Error("precomputed neighbors: class-count shape mismatch")
    for count in n_classes:
        if count < 1:
            raise Error("precomputed neighbors: class counts must be positive")
    var weighted = weights == KNN_HOST_WEIGHTS_DISTANCE
    var w = List[Float32]()
    if weighted:
        w = host_distance_weights(read_f32(dist_address, nq * kk), nq, kk)
    var ys = List[List[Int32]]()
    var uniqs = List[List[Int32]]()
    for i in range(no):
        ys.append(read_i32(y_address + i * ni * 4, ni))
        uniqs.append(host_unique_labels(ys[i], ni))
        if len(uniqs[i]) != n_classes[i]:
            raise Error(
                "knn_classifier_predict: the implemented getUniquelabels found "
                + String(len(uniqs[i]))
                + " classes for output "
                + String(i)
                + " and the caller sized its buffers for "
                + String(n_classes[i])
                + "; one of the two class sets is wrong and nothing is "
                + "written"
            )
    var labels_out = List[Int32](length=nq * no, fill=Int32(0))
    var off = 0
    for i in range(no):
        var n_uniq = len(uniqs[i])
        var mono = host_monotonic(ys[i], ni, uniqs[i])
        var proba: List[Float32]
        if weighted:
            proba = host_weighted_class_probs(idx, mono, w, n_uniq, nq, kk)
        else:
            proba = host_class_probs(idx, mono, n_uniq, nq, kk)
        if want_proba:
            for j in range(nq * n_uniq):
                pp[off + j] = proba[j]
            off += nq * n_uniq
        else:
            host_class_vote(proba, uniqs[i], n_uniq, nq, labels_out, no, i)
    if not want_proba:
        for j in range(nq * no):
            lp[j] = labels_out[j]
    var uoff = 0
    for i in range(no):
        for j in range(len(uniqs[i])):
            up[uoff + j] = uniqs[i][j]
        uoff += len(uniqs[i])


def knn_regress_neighbors_host(
    dist_address: Int,
    idx_address: Int,
    y_address: Int,
    op: _F32P,
    ni: Int,
    nq: Int,
    kk: Int,
    no: Int,
    weights: Int,
) raises:
    """Regress on globally merged neighbors."""
    if ni < 1 or nq < 1 or kk < 1 or kk > ni or no < 1:
        raise Error("precomputed neighbors: invalid shape")
    var xp = u32_ptr(idx_address)
    var idx = List[UInt32](length=nq * kk, fill=UInt32(0))
    for i in range(nq * kk):
        idx[i] = xp[i]
    host_validate_neighbors(idx, ni, nq, kk, no, weights)
    var weighted = weights == KNN_HOST_WEIGHTS_DISTANCE
    var w = List[Float32]()
    if weighted:
        w = host_distance_weights(read_f32(dist_address, nq * kk), nq, kk)
    var out = List[Float32](length=nq * no, fill=Float32(0.0))
    for i in range(no):
        var y = read_f32(y_address + i * ni * 4, ni)
        if weighted:
            host_weighted_regress_avg(idx, y, w, nq, kk, out, no, i)
        else:
            host_regress_avg(idx, y, nq, kk, out, no, i)
    for j in range(nq * no):
        op[j] = out[j]


def radius_check_shapes(
    n_index: Int, n_queries: Int, n_features: Int, radius: Float32, who: String
) raises:
    """`neighbors/estimator.mojo::_radius_check_shapes`, its words."""
    if n_index <= 0:
        raise Error(who + ": n_index must be positive, got " + String(n_index))
    if n_queries <= 0:
        raise Error(
            who + ": n_queries must be positive, got " + String(n_queries)
        )
    if n_features <= 0:
        raise Error(
            who + ": n_features must be positive, got " + String(n_features)
        )
    if not (radius > Float32(0.0)):
        raise Error(
            who
            + ": radius must be positive and finite, got "
            + String(radius)
            + ". A radius of zero returns each query's exact duplicates only,"
            " which the index is not built to answer, and a negative or NaN"
            " radius has no neighbourhood at all."
        )


def radius_neighbors_count_host(
    index_address: Int,
    queries_address: Int,
    ap: _I32P,
    ni: Int,
    nq: Int,
    nf: Int,
    rad: Float32,
    mtr: Int,
    marg: Float32,
) raises -> Int:
    """Pass one of the radius query: `nq + 1` int32 row starts to `ap`;
    returns the edge count."""
    var nnz = 0
    radius_check_shapes(ni, nq, nf, rad, "radius_neighbors_count")
    rbc_validate_metric(mtr, marg)
    var index = read_f32(index_address, ni * nf)
    var queries = read_f32(queries_address, nq * nf)
    var counts = List[Int32](length=nq, fill=Int32(0))
    host_rbc_radius_counts(
        index, ni, queries, nq, nf, rad, mtr, marg, counts
    )
    ap[0] = Int32(0)
    for q in range(nq):
        nnz += Int(counts[q])
        ap[q + 1] = Int32(nnz)
    return nnz


def radius_neighbors_fill_host(
    index_address: Int,
    queries_address: Int,
    ap: _I32P,
    idx_address: Int,
    dist_address: Int,
    ni: Int,
    nq: Int,
    nf: Int,
    rad: Float32,
    cap: Int,
    sq: Bool,
    mtr: Int,
    marg: Float32,
) raises -> Int:
    """Pass two of the radius query: columns ascending within each row and
    the distances, in CSR order. Returns the edge count it found."""
    var nnz = 0
    radius_check_shapes(ni, nq, nf, rad, "radius_neighbors_fill")
    rbc_validate_metric(mtr, marg)
    if cap < 0:
        raise Error(
            "radius_neighbors_fill: nnz_capacity must not be negative, got "
            + String(cap)
        )
    var index = read_f32(index_address, ni * nf)
    var queries = read_f32(queries_address, nq * nf)
    var indptr = List[Int32](length=nq + 1, fill=Int32(0))
    if ap[0] != Int32(0) or Int(ap[nq]) != cap:
        raise Error(
            "radius_neighbors_fill: indptr must start at zero and end at"
            " nnz_capacity; call radius_neighbors_count first"
        )
    for q in range(nq):
        if ap[q + 1] < ap[q]:
            raise Error("radius_neighbors_fill: indptr must be nondecreasing")
        indptr[q] = ap[q]
    indptr[nq] = ap[nq]
    var cols = List[Int32](length=cap, fill=Int32(0))
    var dists = List[Float32](length=cap, fill=Float32(0.0))
    var actual_counts = List[Int32](length=nq, fill=Int32(0))
    host_rbc_radius_fill_rows(
        index, ni, queries, nq, nf, rad, sq, mtr, marg,
        indptr, actual_counts, cols, dists,
    )
    for q in range(nq):
        var expected = Int(indptr[q + 1] - indptr[q])
        var actual = Int(actual_counts[q])
        nnz += actual
        if actual != expected:
            raise Error(
                "radius_neighbors_fill: query row " + String(q)
                + " changed from " + String(expected) + " to "
                + String(actual) + " edges between the count and fill calls."
                " Re-run radius_neighbors_count against the arrays this call"
                " was given rather than returning a stale CSR layout."
            )
    if nnz > 0:
        var xp = i32_ptr(idx_address)
        var dp = f32_ptr(dist_address)
        for p in range(nnz):
            xp[p] = cols[p]
            dp[p] = dists[p]
    for q in range(nq + 1):
        ap[q] = indptr[q]
    return nnz


def rbc_knn_search_host(
    index_address: Int,
    queries_address: Int,
    xp: _I32P,
    dp: _F32P,
    ni: Int,
    nq: Int,
    nf: Int,
    kk: Int,
    mtr: Int,
    marg: Float32,
) raises:
    """EXACT k-NN on the host under the ball cover's metric table and total
    order: `nq * kk` int32 indices and float32 TRUE distances."""
    radius_check_shapes(ni, nq, nf, Float32(1.0), "rbc_knn_search")
    rbc_validate_metric(mtr, marg)
    if kk < 1:
        raise Error("rbc_knn_search: k must be at least 1, got " + String(kk))
    if kk > ni:
        raise Error(
            "rbc_knn_search: k = "
            + String(kk)
            + " exceeds the "
            + String(ni)
            + " points in the index. Refused rather than padded: a padded"
            " answer is indistinguishable from a complete one."
        )
    if kk > RBC_KNN_MAX_K:
        raise Error(
            "rbc_knn_search: k = "
            + String(kk)
            + " exceeds RBC_KNN_MAX_K = "
            + String(RBC_KNN_MAX_K)
            + ". Use knn_search, whose selector is sized per launch."
        )
    var index = read_f32(index_address, ni * nf)
    var queries = read_f32(queries_address, nq * nf)
    var out_idx = List[Int32](length=nq * kk, fill=Int32(-1))
    var out_dist = List[Float32](length=nq * kk, fill=Float32(0.0))
    host_rbc_knn_search(
        index, ni, queries, nq, nf, kk, mtr, marg, out_idx, out_dist,
    )
    for i in range(nq * kk):
        xp[i] = out_idx[i]
        dp[i] = out_dist[i]


def kmeans_fit_host(
    x_address: Int,
    centroids_address: Int,
    lp: _U32P,
    weights_address: Int,
    ns: Int,
    nf: Int,
    nc: Int,
    nw: Int,
    mi: Int,
    tl: Float64,
    sd: UInt64,
    ninit: Int,
    ii: Int,
    mm: Int,
    ovs: Float64,
) raises -> Tuple[Float64, Int, Float64, Float64]:
    """Fit k-means on the host; writes the centroids and labels and returns
    (inertia, n_iter, sum_scale, weight_scale)."""
    host_kmeans_validate(ns, nf, nc, nw)
    var x = read_f32(x_address, ns * nf)
    var centroids: List[Float32]
    if ii == INIT_ARRAY:
        centroids = read_f32(centroids_address, nc * nf)
    else:
        centroids = List[Float32](length=nc * nf, fill=Float32(0.0))
    var weights = List[Float32]()
    if nw != 0:
        weights = read_f32(weights_address, nw)
    var labels = List[UInt32](length=ns, fill=UInt32(0))
    var r = host_kmeans_fit(
        x, ns, nf, nc, centroids, labels, weights, nw, mi, tl, sd,
        ninit, ii, mm, ovs,
    )
    var cp = f32_ptr(centroids_address)
    for i in range(nc * nf):
        cp[i] = centroids[i]
    for i in range(ns):
        lp[i] = labels[i]
    return (r.inertia, r.n_iter, r.sum_scale, r.weight_scale)


def kmeans_predict_host(
    x_address: Int, centroids_address: Int, lp: _U32P, ns: Int, nf: Int, nc: Int, mm: Int
) raises:
    """The nearest centroid of every row on the host (`host_kmeans_predict`)."""
    if ns < 1 or nf < 1 or nc < 1:
        raise Error(
            "kmeans_predict needs n_samples, n_features and n_clusters >= 1: got "
            + String(ns)
            + ", "
            + String(nf)
            + ", "
            + String(nc)
        )
    var x = read_f32(x_address, ns * nf)
    var centroids = read_f32(centroids_address, nc * nf)
    var labels = List[UInt32](length=ns, fill=UInt32(0))
    host_kmeans_predict(x, ns, nf, centroids, nc, mm, labels)
    for i in range(ns):
        lp[i] = labels[i]


def kmeans_transform_host(
    x_address: Int, centroids_address: Int, op: _F32P, ns: Int, nf: Int, nc: Int, mm: Int
) raises:
    """The distance from every row to every centroid on the host
    (`host_kmeans_transform`)."""
    if ns < 1 or nf < 1 or nc < 1:
        raise Error(
            "kmeans_transform needs n_samples, n_features and n_clusters >= 1: got "
            + String(ns)
            + ", "
            + String(nf)
            + ", "
            + String(nc)
        )
    var x = read_f32(x_address, ns * nf)
    var centroids = read_f32(centroids_address, nc * nf)
    var out = List[Float32](length=ns * nc, fill=Float32(0.0))
    host_kmeans_transform(x, ns, nf, centroids, nc, mm, out)
    for i in range(ns * nc):
        op[i] = out[i]
