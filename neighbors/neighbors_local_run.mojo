# SPDX-License-Identifier: Apache-2.0
"""LocalOutlierFactor, NearestCentroid and KNNImputer door RUN: the checks on
typed values, the typed plans and the host buffer steps.

`bindings/neighbors_local_native.mojo` is the UNPACK/PACK side: it reads the
estimator's fields once, refuses what it cannot convert, allocates and
publishes the public arrays, releases the GIL around native work and calls
these functions at the point the public method made each step, so refusal
order and wording are unchanged. Nothing here imports the interpreter.
"""
from std.memory import unsafe_memcpy
from std.os import getenv
from core.lof_plan import LofPlan

comptime LOF_RESULT_SCORES = 0
comptime LOF_RESULT_DECISION = 1
comptime LOF_RESULT_LABELS = 2
# `offset_` of a fit with contamination='auto' (scikit-learn's constant).
comptime LOF_AUTO_OFFSET = -1.5
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


# ------------------------------------------------------------------ shared


def local_check_n_neighbors(n_neighbors: Int) raises:
    """LOF `_check_params` and KNNImputer.transform share this refusal."""
    if n_neighbors < 1:
        raise Error("n_neighbors must be >= 1")


def local_copy_f32(dst: Int, src: Int, count: Int):
    """`X.copy()` of a C-order float32 buffer into a new one of its size."""
    unsafe_memcpy(dest=_F32P(unsafe_from_address=dst), src=_F32P(unsafe_from_address=src), count=count)


def local_write_i32(values: List[Int], dst: Int):
    """`_i32(list)`: a class- or feature-sized int list as int32 words."""
    var p = _I32P(unsafe_from_address=dst)
    for i in range(len(values)):  # driver-loop: over a class- or feature-sized int list of model structure
        p[i] = Int32(values[i])


# ------------------------------------------------------------------ LOF


def lof_refuse_metric(metric_repr: String, p_repr: String) raises:
    raise Error("NotImplementedError: LocalOutlierFactor: metric=" + metric_repr + " (p=" + p_repr
                + ") is not implemented; euclidean (or minkowski with p=2) only")


def lof_refuse_metric_params() raises:
    raise Error("NotImplementedError: LocalOutlierFactor: metric_params is not implemented")


def lof_check_contamination(c: Float64) raises:
    if not (0.0 < c and c <= 0.5):
        raise Error("contamination must be 'auto' or in (0, 0.5]")


def lof_check_fit_rows(n: Int, d: Int) raises:
    if n < 2:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of 2 is required.")


def lof_fit_neighbors(n_neighbors: Int, n: Int) -> Int:
    """The fit's k: `min(n_neighbors, n - 1)` (a row is not its own neighbor)."""
    return min(n_neighbors, n - 1)


def lof_plan(rows: Int, reference_rows: Int, features: Int, k: Int, fitting: Bool,
             unfused: Bool, old_items: Bool) -> LofPlan:
    """The staged plan. The old-items arm applies only to the fused kNN."""
    return LofPlan(rows, reference_rows, features, k, fitting, unfused, old_items if not unfused else False)


def lof_refuse_fit_predict_novelty() raises:
    raise Error("AttributeError: fit_predict is not available when novelty=True; use predict on new data")


def lof_check_novelty(novelty: Bool, what: String) raises:
    if not novelty:
        raise Error("AttributeError: " + what + " is not available when novelty=False; use fit_predict")


# ------------------------------------------------------------ NearestCentroid


def nc_refuse_metric() raises:
    raise Error("NearestCentroid: metric must be 'euclidean' or 'manhattan'")


def nc_check_label_rows(labels: Int, n: Int) raises:
    if labels != n:
        raise Error("X and y have different numbers of rows")


def nc_check_classes(classes: Int) raises:
    if classes < 2:
        raise Error("The number of classes has to be greater than one; got " + String(classes) + " class")


def nc_check_priors_ndim(ndim: Int) raises:
    if ndim != 1:
        raise Error("mojolearn: priors must be 1-D, got " + String(ndim) + "-D")


def nc_check_priors_size(size: Int, classes: Int) raises:
    if size != classes:
        raise Error("priors must have one entry per class")


def nc_split_ops() -> Bool:
    """MOJOLEARN_NC_SPLIT_OPS, read at each fit as the Python fit read it."""
    return getenv("MOJOLEARN_NC_SPLIT_OPS", "") == "1"


def nc_check_variance(constant: Bool) raises:
    if constant:
        raise Error("All features have zero variance. Division by zero.")


def nc_refuse_decision_metric() raises:
    raise Error("AttributeError: decision_function is available for metric='euclidean' only")


def nc_fused(capable: Bool, rows: Int) -> Bool:
    """The fused entry serves a non-empty query only."""
    return capable and rows > 0


def nc_proba_mode(logarithmic: Bool, wide: Bool) -> Int:
    """0 float32 probabilities, 1 their logarithm, 2 float64 probabilities."""
    return 1 if logarithmic else (2 if wide else 0)


# ------------------------------------------------------------- KNNImputer


def imp_refuse_missing_values() raises:
    raise Error("NotImplementedError: KNNImputer: missing_values must be a number or NaN (float input)")


def imp_refuse_metric() raises:
    raise Error("NotImplementedError: KNNImputer: metric must be 'nan_euclidean'")


def imp_refuse_weights() raises:
    raise Error("NotImplementedError: KNNImputer: weights must be 'uniform' or 'distance'")


def imp_round_f32(value: Float64) -> Float64:
    """`_f32_scalar(v)`: `float(v)` rounded once to float32."""
    return Float64(Float32(value))


def imp_cells_length(n: Int, d: Int, lean: Bool) -> Int:
    """The cell list's length; the lean fit entry writes no cell list."""
    return 1 if lean else max(n * d, 1)


def imp_colmiss_length(d: Int) -> Int:
    return max(d, 1)


def imp_read_count(info: Int) -> Int:
    """The missing-cell count `xn_nan_cells` wrote to its info word."""
    return Int(_I32P(unsafe_from_address=info)[0])


@fieldwise_init
struct ImpFitColumns(Copyable, Movable):
    var valid: List[Bool]
    var missing: List[Int]


def imp_fit_columns(colmiss: Int, d: Int, n: Int) -> ImpFitColumns:
    """The fit's column masks from the per-column NaN counts: a column is
    valid when some row is present, missing when some row is NaN."""
    var cm = _I32P(unsafe_from_address=colmiss)
    var valid = List[Bool]()
    var missing = List[Int]()
    for f in range(d):  # driver-loop: over the d feature columns (the fit's column masks)
        valid.append(Int(cm[f]) < n)
    for f in range(d):  # driver-loop: over the d feature columns (the fit's column masks)
        if Int(cm[f]) > 0:
            missing.append(f)
    return ImpFitColumns(valid^, missing^)


def imp_check_features(d: Int, fit_features: Int) raises:
    if d != fit_features:
        raise Error("X has a different number of features than during fit")


def imp_cells_operation(old_items: Bool) -> String:
    """The per-missing-cell imputation entry of the A/B arm."""
    return "knn_impute_cells" if old_items else "knn_impute_tiled"


@fieldwise_init
struct ImpKeepPlan(Copyable, Movable):
    var keep: List[Int]
    var flags: List[Int]
    var all_valid: Bool


def imp_keep_plan(present: List[Bool]) -> ImpKeepPlan:
    """The kept columns (valid at fit) and the per-column empty flags."""
    var keep = List[Int]()
    var flags = List[Int]()
    var all_valid = True
    for f in range(len(present)):  # driver-loop: over the d feature columns (the fit's column masks)
        if present[f]:
            keep.append(f)
            flags.append(0)
        else:
            all_valid = False
            flags.append(1)
    return ImpKeepPlan(keep^, flags^, all_valid)
