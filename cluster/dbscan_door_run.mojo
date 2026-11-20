# SPDX-License-Identifier: Apache-2.0
"""DBSCAN door RUN: the checks on typed values and the typed fit plan.

`bindings/cfb_dbscan.mojo` is the UNPACK/PACK side: it reads the estimator's
Python fields once, refuses what it cannot convert, and calls these checks at
the point the public method made them, so the refusal order and wording are
unchanged. Nothing here imports Python.
"""
from core.dbscan_native_api import DBSCANFitRequest, DBSCANCoreRequest
from core.dbscan_native_state import dbscan_method_accepts
from core.family_cast_impl import family_min_f32
from core.host_finite_scan import host_all_finite_f32

comptime DBSCAN_METRIC_PRECOMPUTED = 3
comptime DBSCAN_EPS_NN_BRUTE_FORCE = 0
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def dbscan_check_method(method: Int, metric: Int, metric_repr: String, algorithm_repr: String) raises:
    """The ball cover serves 'euclidean' only."""
    if not dbscan_method_accepts(method, metric):
        raise Error("mojolearn DBSCAN: metric=" + metric_repr + " with "
                    + "algorithm=" + algorithm_repr + " is refused. The ball cover "
                    + "computes Euclidean distances for its landmark radii and "
                    + "its three pruning bounds (neighbors/impl/"
                    + "ball_cover/) against the radius itself, so it serves "
                    + "'euclidean' only ('cosine' compares unit rows against "
                    + "2 * eps, a threshold the index does not take). Pass "
                    + "algorithm='brute', which serves this metric. This is a "
                    + "scope boundary and not a property of the algorithm: the "
                    + "ball cover's pruning rests on the triangle inequality, "
                    + "which L1 satisfies, so an L1 index is reachable work in "
                    + "that lane")


def dbscan_check_prediction(metric: Int, predicting: Bool) raises:
    if metric == DBSCAN_METRIC_PRECOMPUTED and predicting:
        raise Error("mojolearn DBSCAN: prediction_data=True with "
                    + "metric='precomputed' is refused by name: predict needs each "
                    + "query's distances to the core samples, which a distance "
                    + "matrix over the training rows does not give")


def dbscan_check_eps(eps: Float64) raises:
    if eps <= 0:
        raise Error("mojolearn DBSCAN eps must be positive")


def dbscan_check_min_samples(min_samples: Int) raises:
    if min_samples < 1:
        raise Error("mojolearn DBSCAN min_samples must be at least 1")


def dbscan_check_max_iterations(cap: Int) raises:
    if cap < 0:
        raise Error("mojolearn DBSCAN max_iterations must be None (the fixed "
                    + "point) or a positive cap")


def dbscan_check_budget(budget: Int) raises:
    if budget < 0:
        raise Error("mojolearn DBSCAN max_mbytes_per_batch cannot be negative")


def dbscan_check_x(x_address: Int, n: Int, d: Int, admitted: Bool, metric: Int) raises:
    """The finite scan (skipped for an X the public door already admitted)
    and the precomputed matrix's shape and sign, on the admitted buffer."""
    if not admitted and not host_all_finite_f32(_F32P(unsafe_from_address=x_address), n * d):
        # Before this check a NaN row reached the kernel and came back
        # labelled noise (-1), silently; scikit-learn refuses it.
        raise Error("mojolearn DBSCAN: X contains a NaN or an infinity; a "
                    + "non-finite row has no distance to any other, so it is "
                    + "refused by name")
    if metric == DBSCAN_METRIC_PRECOMPUTED:
        if n != d:
            raise Error("mojolearn DBSCAN: metric='precomputed' needs a square "
                        + "distance matrix, got shape (" + String(n) + ", " + String(d) + ")")
        if Float64(family_min_f32(_F32P(unsafe_from_address=x_address), n * d)) < 0:
            raise Error("mojolearn DBSCAN: metric='precomputed' refuses a "
                        + "negative distance, as scikit-learn does")


def dbscan_fit_request(x_address: Int, labels_address: Int, weight_address: Int, workspace_address: Int,
                       n: Int, d: Int, min_samples: Int, budget: Int, cap: Int, method: Int,
                       metric: Int, eps: Float64) -> DBSCANFitRequest:
    """ORDER MATCHES bindings/_mojolearn_estimators.mojo::dbscan_fit_binding:
    n_rows, n_features, eps, min_samples, budget_mb, max_iter, eps_nn_method,
    metric. precomputed takes the brute arm's adjacency path whatever
    `algorithm` names (DEVIATION 5114)."""
    var effective = DBSCAN_EPS_NN_BRUTE_FORCE if metric == DBSCAN_METRIC_PRECOMPUTED else method
    return DBSCANFitRequest(x_address, labels_address, weight_address, workspace_address,
        n, d, min_samples, budget, cap, effective, metric, eps)


def dbscan_core_request(x_address: Int, labels_address: Int, workspace_address: Int, n: Int, d: Int) -> DBSCANCoreRequest:
    return DBSCANCoreRequest(x_address, labels_address, workspace_address, n, d)
