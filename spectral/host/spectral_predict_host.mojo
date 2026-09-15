# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""`SpectralClustering.predict` on the host, and the pieces the device pass
shares with it (lane/spectral-predict, 2026-09-15; DEVIATION 2860).

NEW CAPABILITY. Neither cuML nor scikit-learn labels a new row under a
fitted SpectralClustering. The rule is the Nystrom out-of-sample extension
of Bengio, Paiement, Vincent, Delalleau, Le Roux and Ouimet, "Out-of-Sample
Extensions for LLE, Isomap, MDS, Eigenmaps, and Spectral Clustering" (NIPS
2003), section 3 (spectral clustering), with Fowlkes, Belongie, Chung and
Malik, "Spectral Grouping Using the Nystrom Method" (TPAMI 2004), followed
by the fit's own k-means assignment. The device spelling is
`spectral/impl/spectral_predict.mojo`; `tools/identity_break.py`'s infer and
batch parts hold the two to the same bytes.

WHAT THE FIT KEEPS (`SpectralPredictionState`, filled by the `_keep` fit
entries only when `prediction_data=True`, COPIES of values the fit already
computes and nothing recomputed):

    eigenvalues   `k` Ritz values `theta_c` of the NEGATED normalized
                  Laplacian, in EMBEDDING column order (the reversed
                  gather's order)
    eigenvectors  `n_train x k` row-major unit Ritz vectors BEFORE the
                  division by the degree scaling, embedding column order
    diag          `n_train` `sqrt(degree)` with zeros replaced by ones, the
                  vector the fit divides by
    centroids     `n_clusters x k` the fit's final k-means centroids

THE RULE, for a query row `q` with affinity slots `(j_s, a_s)`:

 1. Affinity. `affinity='nearest_neighbors'`: the identical k-NN of `q`
    against the training rows (`knn_search` at L2SqrtExpanded, the fit's own
    search) at `k = n_neighbors`; each neighbor gets `a = 0.5`, the value
    the fit's symmetrization `0.5 * (a_ij + a_ji)` gives a one-directional
    edge, because a new row has no reverse edges; the training degrees are
    NOT changed. The slots are the neighbors in ascending training index.
    `affinity='precomputed'`: the caller's `(n_queries, n_train)` affinity,
    slot `j` for every training row `j`, in ascending `j`.
 2. Degree. `d_q = ftz(d_q + a_s)` over the slots in order, seeded `+0.0`
    (the fit's per-row ascending degree fold); `s_q = ftz(identical_sqrt(
    d_q))`, and `1.0` where that is zero (the fit's zero-to-one rule).
 3. Nystrom. For each embedding column `c`, `mu_c = ftz(1.0 + theta_c)` is
    the normalized affinity's eigenvalue. Then
        acc = ftz(identical_mul_add(identical_div(a_s,
                  identical_mul(s_q, diag[j_s])), u_c[j_s], acc))
    over the slots in order, seeded `+0.0`, and
        e_c(q) = identical_div(identical_div(acc, mu_c), s_q)
    which is Bengio et al.'s `(1 / lambda_k) sum_i Ktilde(x, x_i) v_k[i]`
    followed by the fit's own row scaling (the division by the degree
    scaling, `divide_rows_kernel`).
 4. Assignment. `kmeans_predict` at `METRIC_L2_EXPANDED` against the fit's
    centroids: the fit's final assignment pass, ties to the lowest centroid
    index.

THE THRESHOLD. Nothing is dropped: the clustering fit keeps every column
(`drop_first = false`), the trivial one included (its `mu` is near 1). A
column whose `|mu_c|` is below `SPECTRAL_PREDICT_MIN_ABS_EIGENVALUE` (or is
NaN) would amplify the projection by more than `1 / threshold`, so predict
REFUSES BY NAME, naming the column and the value. The test is on the
float32 `mu_c` above, computed on the host by both bindings.

ON THE TRAINING ROWS the rule reproduces `embedding_` only in exact
arithmetic and only for the fit's own graph: a training row asked as a query
sees itself as a neighbor at `0.5` (the fit gave it `1.0` on the diagonal,
which the normalized Laplacian then sets aside), no reverse edges, and the
Lanczos pairs satisfy the eigen-equation only to `eigen_tol`. So
`predict(X_train) == labels_` is MEASURED, not promised
(`bench/results/identity_break/2026-09-15_spectral-predict/`).

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` (the metrics family's
define) or `-D MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE=1` (this pass alone)
negates embedding column 0 of every query, a value and not an order.
"""
from std.sys.compile import is_defined

from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from cluster.host.kmeans_oracle import METRIC_L2_EXPANDED, host_kmeans_predict
from core.knn_host_predict import KNN_HOST_METRIC_FROM_IS_SQRT, host_knn_search


comptime SPECTRAL_PREDICT_HOST_SABOTAGE = (
    is_defined["MOJOLEARN_HOST_SABOTAGE"]()
    or is_defined["MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE"]()
)

#: DEVIATION 2860: a used column with `|1 + theta_c|` below this is refused.
comptime SPECTRAL_PREDICT_MIN_ABS_EIGENVALUE = Float32(1e-3)

#: The weight of a query's edge to one of its k nearest training rows: the
#: fit's `0.5 * (1.0 + 0.0)` for an edge with no reverse edge.
comptime SPECTRAL_PREDICT_ONE_WAY_EDGE = Float32(0.5)

comptime SPECTRAL_AFFINITY_NEAREST_NEIGHBORS = 0
comptime SPECTRAL_AFFINITY_PRECOMPUTED = 1


struct SpectralPredictionState(Movable):
    """What predict needs from the fit; see the module docstring."""

    var eigenvalues: List[Float32]
    var eigenvectors: List[Float32]
    var diag: List[Float32]
    var centroids: List[Float32]

    def __init__(out self):
        self.eigenvalues = List[Float32]()
        self.eigenvectors = List[Float32]()
        self.diag = List[Float32]()
        self.centroids = List[Float32]()


@fieldwise_init
struct SpectralSlots(Movable):
    """`width` slots per query: `cols` a training row index or -1 (empty),
    `vals` the affinity. Row-major `n_queries x width`."""

    var width: Int
    var cols: List[Int32]
    var vals: List[Float32]


@fieldwise_init
struct SpectralPrediction(Movable):
    var labels: List[Int32]
    var embedding: List[Float32]


def spectral_keep_embedding_order(
    ritz: List[Float32],
    ritz_vectors: List[Float32],
    k: Int,
    n: Int,
    mut state: SpectralPredictionState,
):
    """The reversed gather of `compute_eigenpairs` applied to the Ritz values
    and the undivided Ritz vectors: embedding column `c` is Lanczos column
    `k - 1 - c`. A copy in a different order; no arithmetic."""
    state.eigenvalues = List[Float32](capacity=k)
    for c in range(k):
        state.eigenvalues.append(ritz[k - 1 - c])
    state.eigenvectors = List[Float32](capacity=n * k)
    for p in range(n):
        for c in range(k):
            state.eigenvectors.append(ritz_vectors[(k - 1 - c) * n + p])


def spectral_predict_validate(
    n_train: Int,
    n_queries: Int,
    n_features: Int,
    n_components: Int,
    n_clusters: Int,
    n_neighbors: Int,
    affinity: Int,
) raises:
    """The refusals both bindings raise, in one order and one wording."""
    if affinity != SPECTRAL_AFFINITY_NEAREST_NEIGHBORS and affinity != SPECTRAL_AFFINITY_PRECOMPUTED:
        raise Error(
            "spectral_predict: affinity must be 0 (nearest_neighbors) or 1"
            " (precomputed), got " + String(affinity)
        )
    if n_train < 2:
        raise Error("spectral_predict: the fitted model holds " + String(n_train) + " training rows")
    if n_queries < 1:
        raise Error("spectral_predict: X has no rows; refused by name")
    if n_components < 1 or n_components >= n_train:
        raise Error(
            "spectral_predict: n_components=" + String(n_components)
            + " must satisfy 1 <= n_components < n_train=" + String(n_train)
        )
    if n_clusters < 1 or n_clusters > n_train:
        raise Error("spectral_predict: n_clusters=" + String(n_clusters) + " is outside [1, n_train]")
    if affinity == SPECTRAL_AFFINITY_NEAREST_NEIGHBORS:
        if n_features < 1:
            raise Error("spectral_predict: X has no features; refused by name")
        if n_neighbors < 1 or n_neighbors > n_train:
            raise Error(
                "spectral_predict: n_neighbors=" + String(n_neighbors)
                + " must satisfy 1 <= n_neighbors <= n_train=" + String(n_train)
            )


def spectral_predict_check_state(
    state: SpectralPredictionState, n_train: Int, n_components: Int, n_clusters: Int
) raises:
    if (
        len(state.eigenvalues) != n_components
        or len(state.eigenvectors) != n_train * n_components
        or len(state.diag) != n_train
        or len(state.centroids) != n_clusters * n_components
    ):
        raise Error(
            "spectral_predict: the prediction data does not match n_train="
            + String(n_train) + ", n_components=" + String(n_components)
            + ", n_clusters=" + String(n_clusters) + "; refused by name"
        )


def spectral_predict_mu(eigenvalues: List[Float32]) raises -> List[Float32]:
    """`mu_c = ftz(1.0 + theta_c)` per column, refused by name below the
    DEVIATION 2860 threshold (a NaN fails `>=` and is refused too)."""
    var out = List[Float32](capacity=len(eigenvalues))
    for c in range(len(eigenvalues)):
        var mu = ftz(Float32(1.0) + eigenvalues[c])
        if not (abs(mu) >= SPECTRAL_PREDICT_MIN_ABS_EIGENVALUE):
            raise Error(
                "spectral_predict: embedding column " + String(c)
                + " has normalized affinity eigenvalue 1 + theta = " + String(mu)
                + ", |value| below the DEVIATION 2860 threshold "
                + String(SPECTRAL_PREDICT_MIN_ABS_EIGENVALUE)
                + "; the Nystrom extension would divide by it, so predict is refused by name"
            )
        out.append(mu)
    return out^


def spectral_slots_from_knn(idx: List[UInt32], n_queries: Int, k: Int) -> SpectralSlots:
    """Each query's k neighbor indices sorted ascending (an insertion sort
    on distinct integers), every value the one-way edge weight."""
    var cols = List[Int32](capacity=n_queries * k)
    var vals = List[Float32](capacity=n_queries * k)
    for q in range(n_queries):
        var row = List[Int32](capacity=k)
        for s in range(k):
            var v = Int32(idx[q * k + s])
            var pos = len(row)
            row.append(v)
            while pos > 0 and row[pos - 1] > v:
                row[pos] = row[pos - 1]
                pos -= 1
            row[pos] = v
        for s in range(k):
            cols.append(row[s])
            vals.append(SPECTRAL_PREDICT_ONE_WAY_EDGE)
    return SpectralSlots(k, cols^, vals^)


def spectral_slots_from_dense(affinity: List[Float32], n_queries: Int, n_train: Int) raises -> SpectralSlots:
    """Slot `j` of query `q` is training row `j` with `affinity[q, j]`. A
    non-finite or negative affinity is refused by name, as the fit refuses
    one (a negative degree has a NaN square root)."""
    var cols = List[Int32](capacity=n_queries * n_train)
    var vals = List[Float32](capacity=n_queries * n_train)
    for q in range(n_queries):
        for j in range(n_train):
            var v = affinity[q * n_train + j]
            if not (v >= Float32(0.0)) or v == Float32(1.0) / Float32(0.0):
                raise Error(
                    "spectral_predict: the affinity to the training rows has a non-finite or"
                    " negative value at (" + String(q) + ", " + String(j) + "); refused by name"
                )
            cols.append(Int32(j))
            vals.append(v)
    return SpectralSlots(n_train, cols^, vals^)


def host_spectral_nystrom(
    slots: SpectralSlots,
    n_queries: Int,
    state: SpectralPredictionState,
    mu: List[Float32],
    k: Int,
) -> List[Float32]:
    """Steps 2 and 3 of the rule, one query at a time. Row-major
    `n_queries x k`."""
    var w = slots.width
    var out = List[Float32](length=n_queries * k, fill=Float32(0.0))
    for q in range(n_queries):
        var deg = Float32(0.0)
        for s in range(w):
            if slots.cols[q * w + s] >= 0:
                deg = ftz(deg + slots.vals[q * w + s])
        var sd = ftz(identical_sqrt(deg))
        if sd == Float32(0.0):
            sd = Float32(1.0)
        for c in range(k):
            var acc = Float32(0.0)
            for s in range(w):
                var j = Int(slots.cols[q * w + s])
                if j >= 0:
                    var kt = identical_div(
                        slots.vals[q * w + s], identical_mul(sd, state.diag[j])
                    )
                    acc = ftz(identical_mul_add(kt, state.eigenvectors[j * k + c], acc))
            var e = identical_div(identical_div(acc, mu[c]), sd)
            comptime if SPECTRAL_PREDICT_HOST_SABOTAGE:
                # THE NEGATIVE CONTROL: wrong on purpose (module docstring).
                if c == 0:
                    e = -e
            out[q * k + c] = e
    return out^


def host_spectral_predict(
    input: List[Float32],
    train_x: List[Float32],
    n_train: Int,
    n_queries: Int,
    n_features: Int,
    n_components: Int,
    n_clusters: Int,
    n_neighbors: Int,
    affinity: Int,
    state: SpectralPredictionState,
) raises -> SpectralPrediction:
    """The whole rule on the host. `input` is the queries (`n_queries x
    n_features`) or the precomputed affinity (`n_queries x n_train`);
    `train_x` is the training rows for nearest_neighbors and unread
    otherwise."""
    spectral_predict_validate(
        n_train, n_queries, n_features, n_components, n_clusters, n_neighbors, affinity
    )
    spectral_predict_check_state(state, n_train, n_components, n_clusters)
    var mu = spectral_predict_mu(state.eigenvalues)
    var k = n_components
    var slots: SpectralSlots
    if affinity == SPECTRAL_AFFINITY_NEAREST_NEIGHBORS:
        if len(input) < n_queries * n_features or len(train_x) < n_train * n_features:
            raise Error("spectral_predict: a buffer is shorter than its shape; refused by name")
        var dist = List[Float32](length=n_queries * n_neighbors, fill=Float32(0.0))
        var idx = List[UInt32](length=n_queries * n_neighbors, fill=UInt32(0))
        host_knn_search(
            train_x, n_train, input, n_queries, n_features, n_neighbors,
            KNN_HOST_METRIC_FROM_IS_SQRT, True, dist, idx,
        )
        slots = spectral_slots_from_knn(idx, n_queries, n_neighbors)
    else:
        if len(input) < n_queries * n_train:
            raise Error("spectral_predict: a buffer is shorter than its shape; refused by name")
        slots = spectral_slots_from_dense(input, n_queries, n_train)
    var emb = host_spectral_nystrom(slots, n_queries, state, mu, k)
    var u_labels = List[UInt32](length=n_queries, fill=UInt32(0))
    host_kmeans_predict(emb, n_queries, k, state.centroids, n_clusters, METRIC_L2_EXPANDED, u_labels)
    var labels = List[Int32](capacity=n_queries)
    for i in range(n_queries):
        labels.append(Int32(u_labels[i]))
    return SpectralPrediction(labels^, emb^)
