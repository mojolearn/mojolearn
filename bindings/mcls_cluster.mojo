# SPDX-License-Identifier: Apache-2.0
# The cluster, divergence, silhouette, trustworthiness and regression
# metrics of python/mojolearn/_metrics_impl.py, one native call each (lane
# mcb2-mcls): the Python loaders' refusals in their order and wording
# (bindings/mcls_common.mojo), then the SAME cuML-backed entry of the
# metrics binding the shell resolved (`_get_binding(numeric_mode)`) with the
# same addresses and parameter lists. The options of the regression errors
# and r2_score run the x_metrics program of bindings/metrics_reg_native.mojo
# itself (`_mean_error`, `_r2_options`), as `regression_error_options` and
# `r2_score_options` did. Same kernels, same operands: the bits do not move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays import classical_address
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_call, family_array, family_as_f32, family_min,
)
from bindings.metrics_reg_native import _mean_error, _r2_options
from bindings.mcls_common import (
    mc_b, mc_len, mc_float, mc_int, mc_is_str, mc_entry, mc_base, mc_shape_of, mc_rank, mc_dim,
    mc_i32_1d, mc_f32_1d, mc_regression_f32_1d, mc_pair_check, mc_weights, mc_encode_store,
)
from core.family_cast_impl import family_all_finite_f32
from core.mcls_impl import mcls_min_i32, mcls_max_i32, mcls_copy_words
from checks.numerics import portable_exp64, portable_log64


def _params(*values: Int) raises -> PythonObject:
    """The scalar list a metrics entry takes (`[n, lower, upper, ...]`)."""
    var out = Python.list()
    for i in range(len(values)):  # small-loop(values: the scalar words of one entry call): bounded by the call site
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _as_float(value: PythonObject) raises -> PythonObject:
    """`float(value)` of an entry's answer."""
    return family_call(mc_b().float, value)


def _pair_i32(a: PythonObject, b: PythonObject, name_a: String, name_b: String,
              context: PythonObject) raises -> PythonObject:
    """`_pair_1d(a, b, name_a, name_b, _as_i32_1d)`."""
    var x = mc_i32_1d(a, name_a, context)
    var y = mc_i32_1d(b, name_b, context)
    mc_pair_check(x, y, name_a, name_b)
    return Python.tuple(x, y)


def _cluster_codes(labels_true: PythonObject, labels_pred: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`_prepare_cluster_labels`: both label arrays mapped onto 0..k-1 over
    the sorted union of their labels, encoded ONCE end to end by the base
    binding's `prepare_targets` (`_native_union_codes`). Returns
    (codes of both, n, upper = k - 1)."""
    var pair = _pair_i32(labels_true, labels_pred, "labels_true", "labels_pred", context)
    var yt = pair[0]
    var yp = pair[1]
    var nt = Int(py=yt.size)
    var npred = Int(py=yp.size)
    var both = family_array(Python.tuple(nt + npred), "<i4", context)
    var base = Int(py=classical_address(both))
    _ = both  # keep the Python object alive through the native call above
    mcls_copy_words(Int(py=classical_address(yt)), base, nt)
    _ = yt  # keep the Python object alive through the native call above
    mcls_copy_words(Int(py=classical_address(yp)), base + 4 * nt, npred)
    _ = yp  # keep the Python object alive through the native call above
    var b = mc_b()
    var total = nt + npred
    var classes = family_call(Python.import_module("array").array, PythonObject("q"),
                              py_call(b.bytes, PythonObject(8 * total)))
    var codes = family_array(Python.tuple(total), "<i4", context)
    var k = Int(py=family_call(mc_base("prepare_targets", context), classical_address(both), PythonObject(2),
                               PythonObject(total), py_call(classes.buffer_info)[0], classical_address(codes),
                               PythonObject(0), PythonObject(0), PythonObject(0)))
    _ = both  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    if k < 0:
        raise Error("RuntimeError: prepare_targets refused integer labels")
    _ = both^
    return Python.tuple(codes, PythonObject(nt), PythonObject(k - 1))


def _cluster_call(binding: PythonObject, entry: String, prepared: PythonObject, extra: PythonObject) raises -> PythonObject:
    """One remapped-label entry: `entry(yt, yp, [n, 0, k - 1, *extra])`."""
    var codes = prepared[0]
    var n = Int(py=prepared[1])
    var base = Int(py=classical_address(codes))
    _ = codes  # keep the Python object alive through the native call above
    var params = _params(n, 0, Int(py=prepared[2]))
    for i in range(mc_len(extra)):  # small-loop(extra: the trailing scalars of one entry call): at most one beta
        _ = py_call(params.append, extra[i])
    return _as_float(family_call(mc_entry(binding, entry), PythonObject(base), PythonObject(base + 4 * n), params))


def _raw_pair_call(binding: PythonObject, entry: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """rand_score / adjusted_rand_score: raw labels, `entry(yt, yp, [n])`."""
    var pair = _pair_i32(args[1], args[2], "labels_true", "labels_pred", context)
    var yt = pair[0]
    var yp = pair[1]
    var result = _as_float(family_call(mc_entry(binding, entry), classical_address(yt), classical_address(yp),
                                       _params(Int(py=yt.size))))
    _ = yp  # keep the Python object alive through the native call above
    _ = yt  # keep the Python object alive through the native call above
    _ = yt^
    _ = yp^
    return result


def _entropy(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (binding, clustering, base)."""
    var lab = mc_i32_1d(args[1], "clustering", context)
    var n = Int(py=lab.size)
    var words = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(lab)))
    _ = lab  # keep the Python object alive through the native call above
    var lower = mcls_min_i32(words, n)
    _ = lab  # keep the Python object alive through the native call above
    var upper = mcls_max_i32(words, n)
    _ = lab  # keep the Python object alive through the native call above
    var value = Float64(py=_as_float(family_call(mc_entry(args[0], "entropy"), classical_address(lab),
                                                 _params(n, lower, upper))))
    _ = lab  # keep the Python object alive through the native call above
    _ = lab^
    if args[2] is None:
        return PythonObject(value)
    # `_portable_math.log(_portable_math.exp(value), base)`: the pinned
    # binary64 exp and log (checks/numerics.mojo, the polynomials of
    # portable_math.c), with that module's refusals.
    var e = portable_exp64(value)
    if e == e and e - e != 0.0 and value - value == 0.0:
        raise Error("OverflowError: math range error")
    if e <= 0.0:
        raise Error("math domain error")
    var top = portable_log64(e)
    var base = mc_float(args[2])
    if base <= 0.0:
        raise Error("math domain error")
    var bottom = portable_log64(base)
    if bottom == 0.0:
        # `result / log(base)` with log(base) == 0: Python's own
        # ZeroDivisionError, restored unchanged at the door
        return family_call(Python.import_module("operator").truediv, PythonObject(top), PythonObject(bottom))
    return PythonObject(top / bottom)


def _silhouette(args: PythonObject, caller: String, context: PythonObject) raises -> PythonObject:
    """`_silhouette`; args = (binding, X, labels, metric, chunksize):
    (mean, per-sample scores)."""
    var b = mc_b()
    var metric = args[3]
    var ok = mc_is_str(metric)
    if ok:
        var lowered = String(family_call(metric.lower))
        ok = lowered == "euclidean" or lowered == "l2"
    if not ok:
        raise Error("NotImplementedError: mojolearn " + caller + ": metric=" + family_repr(metric) + " is refused; only "
                    + "'euclidean' and 'l2' are implemented (they are the same "
                    + "DistanceType::L2SqrtUnexpanded, which is the arm cuML's "
                    + "silhouette_score.pyx dispatches by default). 'cityblock', "
                    + "'cosine', 'l1', 'manhattan' and 'sqeuclidean' are other "
                    + "distance kernels and are not in metrics/ (NOT_IMPLEMENTED.tsv)")
    var x = family_as_f32(args[1], 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), rows * cols):
        raise Error("mojolearn " + caller + ": X contains NaN or infinity; refused here "
                    + "(metrics/README.md HAND-OFF ask 3), because a NaN distance "
                    + "reaches sil_op as +0.0 (DEVIATION 656) rather than as an error")
    _ = x  # keep the Python object alive through the native calls above
    var lab = mc_i32_1d(args[2], "labels", context)
    var n_lab = Int(py=lab.size)
    if n_lab != rows:
        raise Error("mojolearn " + caller + ": labels has " + String(n_lab) + " entries and X has "
                    + String(rows) + " rows")
    # `_labels.encode_labels(lab)`: prepare_targets over the int32 labels.
    var store = family_call(Python.import_module("array").array, PythonObject("q"),
                            py_call(b.bytes, PythonObject(8 * n_lab)))
    var mapped = family_array(Python.tuple(n_lab), "<i4", context)
    var k = Int(py=family_call(mc_base("prepare_targets", context), classical_address(lab), PythonObject(2),
                               PythonObject(n_lab), py_call(store.buffer_info)[0], classical_address(mapped),
                               PythonObject(0), PythonObject(0), PythonObject(0)))
    _ = lab  # keep the Python object alive through the native call above
    _ = mapped  # keep the Python object alive through the native call above
    if k < 0:
        raise Error("RuntimeError: prepare_targets refused integer labels")
    var chunk = 40000
    if args[4] is not None:
        chunk = mc_int(args[4])
    if chunk < 1:
        raise Error("mojolearn " + caller + ": chunksize must be at least 1, got " + String(chunk))
    var scores = family_array(Python.tuple(rows), "<f4", context)
    var mean = _as_float(family_call(mc_entry(args[0], "silhouette"), classical_address(x), classical_address(mapped),
                                     classical_address(scores), _params(rows, cols, k, chunk)))
    _ = mapped  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = scores  # keep the Python object alive through the native call above
    _ = x^
    _ = lab^
    _ = mapped^
    return Python.tuple(mean, scores)


def _trustworthiness(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (binding, X, X_embedded, n_neighbors, metric, batch_size)."""
    var metric = args[4]
    var ok = mc_is_str(metric)
    if ok:
        ok = String(family_call(metric.lower)) == "euclidean"
    if not ok:
        raise Error("NotImplementedError: mojolearn trustworthiness: metric=" + family_repr(metric)
                    + " is refused; only 'euclidean' is implemented, and cuML's trustworthiness.pyx:86 "
                    + "refuses every other name too")
    var x = family_as_f32(args[1], 2, "X", context)
    var emb = family_as_f32(args[2], 2, "X_embedded", context)
    var rows = Int(py=x.shape[0])
    if rows != Int(py=emb.shape[0]):
        raise Error("mojolearn trustworthiness: X has " + String(rows) + " rows and X_embedded has "
                    + String(Int(py=emb.shape[0])))
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), Int(py=x.size)):
        raise Error("mojolearn trustworthiness: X contains NaN or infinity; refused here (metrics/README.md HAND-OFF ask 3)")
    _ = x  # keep the Python object alive through the native calls above
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(emb))), Int(py=emb.size)):
        raise Error("mojolearn trustworthiness: X_embedded contains NaN or infinity; refused here "
                    + "(metrics/README.md HAND-OFF ask 3)")
    _ = emb  # keep the Python object alive through the native calls above
    var batch = mc_int(args[5])
    if batch < 1:
        raise Error("mojolearn trustworthiness: batch_size must be at least 1, got " + family_repr(args[5]))
    var result = _as_float(family_call(mc_entry(args[0], "trustworthiness"), classical_address(x), classical_address(emb),
                                       _params(rows, Int(py=x.shape[1]), Int(py=emb.shape[1]), mc_int(args[3]), batch)))
    _ = emb  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = x^
    _ = emb^
    return result


def _kl_divergence(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (binding, P, Q)."""
    var p = mc_f32_1d(args[1], "P", True, context)
    var q = mc_f32_1d(args[2], "Q", True, context)
    mc_pair_check(p, q, "P", "Q")
    if family_min(p) < 0:
        raise Error("mojolearn kl_divergence: P has a negative entry; refused here (metrics/README.md "
                    + "HAND-OFF ask 3) rather than letting log of a negative reach the fold")
    if family_min(q) < 0:
        raise Error("mojolearn kl_divergence: Q has a negative entry; refused here (metrics/README.md "
                    + "HAND-OFF ask 3) rather than letting log of a negative reach the fold")
    var result = _as_float(family_call(mc_entry(args[0], "kl_divergence"), classical_address(p), classical_address(q),
                                       _params(Int(py=p.size))))
    _ = q  # keep the Python object alive through the native call above
    _ = p  # keep the Python object alive through the native call above
    _ = p^
    _ = q^
    return result


def _regression_error(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_regression_error`; args = (binding, name, y_true, y_pred,
    sample_weight, multioutput, numeric_mode)."""
    var name = String(args[1])
    var multioutput = args[5]
    var shape = mc_shape_of(args[2])
    if (args[4] is not None or not mc_is_str(multioutput) or String(multioutput) != "uniform_average"
            or (mc_rank(shape) == 2 and mc_dim(shape, 1) != 1)):
        # `regression_error_options`: the x_metrics program
        var rest = Python.tuple(args[2], args[3], args[4], args[5], args[6])
        if name == "mean_squared_error":
            return _mean_error(rest, "sq", name, False, 0.0, "", context)
        if name == "mean_absolute_error":
            return _mean_error(rest, "abs", name, False, 0.0, "", context)
        return _mean_error(rest, "sq", name, True, 0.0, "", context)
    var yt = mc_regression_f32_1d(args[2], "y_true", context)
    var yp = mc_regression_f32_1d(args[3], "y_pred", context)
    mc_pair_check(yt, yp, "y_true", "y_pred")
    var result = _as_float(family_call(mc_entry(args[0], name), classical_address(yt), classical_address(yp),
                                       _params(Int(py=yt.size))))
    _ = yp  # keep the Python object alive through the native call above
    _ = yt  # keep the Python object alive through the native call above
    _ = yt^
    _ = yp^
    return result


def _r2(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """r2_score; args = (binding, y_true, y_pred, sample_weight, multioutput,
    force_finite, numeric_mode)."""
    var multioutput = args[4]
    var shape = mc_shape_of(args[1])
    if (not mc_is_str(multioutput) or String(multioutput) != "uniform_average"
            or not Bool(py=family_call(mc_b().bool, args[5]))
            or (mc_rank(shape) == 2 and mc_dim(shape, 1) != 1)):
        # `r2_score_options`: the x_metrics program
        return _r2_options(Python.tuple(args[1], args[2], args[3], args[4], args[5], args[6]), context)
    var y = mc_f32_1d(args[1], "y_true", True, context)
    var yh = mc_f32_1d(args[2], "y_pred", True, context)
    mc_pair_check(y, yh, "y_true", "y_pred")
    var n = Int(py=y.size)
    var result: PythonObject
    if args[3] is not None:
        var w = mc_weights(args[3], n, "r2_score", context)
        result = _as_float(family_call(mc_entry(args[0], "r2_score_weighted"), classical_address(y),
                                       classical_address(yh), classical_address(w), _params(n)))
        _ = y  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = yh  # keep the Python object alive through the native call above
        _ = w^
    else:
        result = _as_float(family_call(mc_entry(args[0], "r2_score"), classical_address(y), classical_address(yh),
                                       _params(n)))
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
    _ = y^
    _ = yh^
    return result


def mcls_cluster_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The label-pair, divergence, silhouette, trustworthiness and
    regression metrics; args[0] is the metrics binding."""
    var b = mc_b()
    if operation == "family_mcls_rand":
        return _raw_pair_call(args[0], "rand_score", args, context)
    if operation == "family_mcls_adjusted_rand":
        return _raw_pair_call(args[0], "adjusted_rand_score", args, context)
    if operation == "family_mcls_entropy":
        return _entropy(args, context)
    if operation == "family_mcls_mutual_info":
        # args = (binding, labels_true, labels_pred, contingency)
        if args[3] is not None:
            raise Error("NotImplementedError: mojolearn mutual_info_score: contingency= is refused; the "
                        + "implemented entry builds the contingency matrix on the device "
                        + "(metrics/impl/stats/detail/contingency_matrix.mojo) and has "
                        + "no arm that consumes a precomputed one")
        return _cluster_call(args[0], "mutual_info_score", _cluster_codes(args[1], args[2], context), Python.list())
    if operation == "family_mcls_fowlkes_mallows":
        # args = (binding, labels_true, labels_pred, sparse)
        if not (mc_is_str(args[3]) and String(args[3]) == "deprecated"):
            if Bool(py=family_call(Python.import_module("operator").ne, args[3], PythonObject("deprecated"))):
                raise Error("NotImplementedError: mojolearn fowlkes_mallows_score: sparse= is refused; scikit-learn "
                            + "deprecated and ignores it, and this surface has no sparse arm")
        var empty = False
        try:
            empty = (Int(py=py_call(b.len, args[1])) == 0 and Int(py=py_call(b.len, args[2])) == 0)
        except:
            empty = False
        if empty:
            return PythonObject(0.0)
        return _cluster_call(args[0], "fowlkes_mallows_score", _cluster_codes(args[1], args[2], context), Python.list())
    if operation == "family_mcls_homogeneity":
        return _cluster_call(args[0], "homogeneity_score", _cluster_codes(args[1], args[2], context), Python.list())
    if operation == "family_mcls_completeness":
        return _cluster_call(args[0], "completeness_score", _cluster_codes(args[1], args[2], context), Python.list())
    if operation == "family_mcls_v_measure":
        # args = (binding, labels_true, labels_pred, beta)
        return _cluster_call(args[0], "v_measure_score", _cluster_codes(args[1], args[2], context),
                             Python.list(family_call(b.float, args[3])))
    if operation == "family_mcls_hcv":
        # args = (binding, labels_true, labels_pred, beta): the labels are
        # checked and remapped once; the three entries are the three calls
        # homogeneity_score, completeness_score and v_measure_score made
        var prepared = _cluster_codes(args[1], args[2], context)
        var h = _cluster_call(args[0], "homogeneity_score", prepared, Python.list())
        var c = _cluster_call(args[0], "completeness_score", prepared, Python.list())
        var v = _cluster_call(args[0], "v_measure_score", prepared, Python.list(family_call(b.float, args[3])))
        return Python.tuple(h, c, v)
    if operation == "family_mcls_kl":
        return _kl_divergence(args, context)
    if operation == "family_mcls_silhouette_score":
        # args = (binding, X, labels, metric, sample_size, random_state, chunksize)
        if args[4] is not None or args[5] is not None:
            raise Error("NotImplementedError: mojolearn silhouette_score: sample_size / random_state are "
                        + "refused; subsample X yourself so the rows that were scored are "
                        + "the rows you chose (cuML's surface has no subsampling either)")
        return _silhouette(Python.tuple(args[0], args[1], args[2], args[3], args[6]), "silhouette_score", context)[0]
    if operation == "family_mcls_silhouette_samples":
        # args = (binding, X, labels, metric, chunksize)
        return _silhouette(args, "silhouette_samples", context)[1]
    if operation == "family_mcls_trustworthiness":
        return _trustworthiness(args, context)
    if operation == "family_mcls_regression_error":
        return _regression_error(args, context)
    if operation == "family_mcls_r2":
        return _r2(args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
