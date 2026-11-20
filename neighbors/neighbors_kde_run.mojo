# SPDX-License-Identifier: Apache-2.0
"""KernelDensity door RUN: the checks on typed values, the host copies and the
typed native requests.

`bindings/neighbors_kde_native_door.mojo` is the UNPACK/PACK side: it reads the
estimator's fields once, compares the admitted scalars, formats the
diagnostics it alone can read, allocates and publishes the arrays, releases
the GIL around native work (`bindings/kde_native_boundary.mojo`) and calls
these functions at the point the public method made each step, so refusal
order and wording are unchanged. Nothing here imports the interpreter.
"""
from std.memory import unsafe_memcpy
from core.admitted_input import admitted_cast_f32
from core.cast_in import CAST_KIND_FLOAT
from core.host_finite_scan import host_finite_status
from core.kde_native_api import KDEPrepareRequest, KDEQueryRequest, KDEResidentQuery, KDE_CAP_RESIDENT, KDE_CAP_IMMUTABLE
from core.kde_native_state import KDEFitKey

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
# The tree-traversal knob a brute-force KDE accepts at its default only.
comptime KDE_DEFAULT_LEAF_SIZE = 40


# ------------------------------------------------------------ _validate_params


def kde_refuse_bandwidth_name(name: String) raises:
    raise Error("NotImplementedError: mojolearn KernelDensity: bandwidth='" + name
                + "' is refused; compute "
                + "the rule on the host and pass the float (sklearn's scott is "
                + "n ** (-1/(d+4)), silverman is (n (d+2) / 4) ** (-1/(d+4)))")


def kde_bandwidth_positive(bandwidth: Float64) -> Bool:
    """A positive, non-NaN bandwidth."""
    return bandwidth > 0.0 and bandwidth == bandwidth


def kde_refuse_bandwidth(width_repr: String) raises:
    raise Error("mojolearn KernelDensity: bandwidth must be positive, got " + width_repr)


def kde_refuse_algorithm(algorithm_repr: String) raises:
    raise Error("mojolearn KernelDensity: algorithm=" + algorithm_repr + " is refused; this is "
                + "cuML's brute-force KDE, there is no tree to choose")


def kde_refuse_tree_knobs() raises:
    raise Error("mojolearn KernelDensity: atol/rtol/breadth_first/leaf_size "
                + "are tree-traversal knobs; this is brute force (refused)")


def kde_refuse_names() raises:
    raise Error("mojolearn KernelDensity: kernel and metric are names")


def kde_check_metric_params_count(count: Int) raises:
    """cuML's rule, `kernel_density.py:302-307`: at most ONE entry, and its
    VALUE is taken regardless of its key."""
    if count != 1:
        raise Error("mojolearn KernelDensity: only metrics with a single "
                    + "argument are supported (cuML raises the same, "
                    + "kernel_density.py:304-306)")


def kde_minkowski_supported(p: Float64) -> Bool:
    """The binding serves Minkowski p=2 only (a NaN p is not 2)."""
    return p == 2.0


def kde_refuse_minkowski(params_repr: String, p_repr: String) raises:
    raise Error("NotImplementedError: mojolearn KernelDensity: metric_params=" + params_repr
                + " asks for "
                + "Minkowski p=" + p_repr + ". The KERNEL computes any finite "
                + "positive normal p (kde/impl/distance/distance.mojo, "
                + "DEVIATION 552) and kde/estimator.mojo takes it as "
                + "`metric_arg`; what is missing is a SLOT IN THE "
                + "BINDING -- kde_score_samples_binding length-checks "
                + "its params list at exactly 5 entries "
                + "(bindings/_mojolearn_estimators.mojo:647, and the same "
                + "check in bindings/_mojolearn_estimators_host.mojo:213). "
                + "Pass p=2, "
                + "or apply the two-line binding change in "
                + "kde/README.md's HAND-OFF.")


# ------------------------------------------------------------ fit and scores


def kde_refuse_unfitted() raises:
    raise Error("mojolearn KernelDensity: call fit() first")


def kde_refuse_features(query_features: String, fit_features: String) raises:
    raise Error("mojolearn KernelDensity: X has " + query_features + " features, "
                + "fit saw " + fit_features)


def kde_resident_capable(capabilities: Int) -> Bool:
    return (capabilities & KDE_CAP_RESIDENT) != 0


def kde_immutable_capable(capabilities: Int) -> Bool:
    """The opt-in immutable snapshot: a transport-only copy of caller bytes."""
    return (capabilities & KDE_CAP_IMMUTABLE) != 0


def kde_copy_f32(dst: Int, src: Int, count: Int):
    """`value.copy()` of a C-order float32 buffer."""
    unsafe_memcpy(dest=_F32P(unsafe_from_address=dst), src=_F32P(unsafe_from_address=src), count=count)


def kde_copy_finite_f32(src: Int, dst: Int, rows: Int, cols: Int) -> Bool:
    """The borrowed fit set's copy with the finite test in the same pass:
    one parallel row-task copy (float32 to float32: the same bits). True when
    every word is finite."""
    # parallel: the copy and the finiteness verdict over row tasks across host cores
    return admitted_cast_f32(src, cols * 4, 4, rows, cols, 4, CAST_KIND_FLOAT, _F32P(unsafe_from_address=dst))


def kde_query_finite_status(query: Int, count: Int) -> Int:
    """The host route's parallel query finite scan: 0 when every word is finite."""
    return Int(host_finite_status(query, count))


def kde_fit_key(train: Int, rows: Int, features: Int, weights: Int, kernel: String, metric: String,
                bandwidth: Float64) -> KDEFitKey:
    return KDEFitKey(train, rows, features, weights, kernel.copy(), metric.copy(), bandwidth)


def kde_prepare_request(key: KDEFitKey, has_weights: Bool) -> KDEPrepareRequest:
    """The resident preparation of a fit key. The fit's native finite
    copy/cast already tested the training set: finite_tested is set."""
    return KDEPrepareRequest(key.train, key.weights, key.rows, key.features, key.bandwidth,
                             has_weights, True, key.kernel.copy(), key.metric.copy())


def kde_query_request(train: Int, query: Int, weights: Int, dst: Int, train_rows: Int, query_rows: Int,
                      features: Int, bandwidth: Float64, has_weights: Bool, want_total: Bool,
                      kernel: String, metric: String) -> KDEQueryRequest:
    """The direct (non-resident) score request."""
    return KDEQueryRequest(train, query, weights, dst, train_rows, query_rows, features, bandwidth,
                           has_weights, want_total, kernel.copy(), metric.copy())


def kde_resident_query(handle: Int, query: Int, dst: Int, status: Int, rows: Int, features: Int,
                       bandwidth: Float64, want_total: Bool, kernel: String, metric: String) -> KDEResidentQuery:
    return KDEResidentQuery(handle, query, dst, status, rows, features, bandwidth, want_total,
                            kernel.copy(), metric.copy())
