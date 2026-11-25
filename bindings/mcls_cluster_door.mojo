# SPDX-License-Identifier: Apache-2.0
# The cluster, divergence, silhouette, trustworthiness and regression
# metrics of python/mojolearn/_metrics_impl.py, one native call each (lane
# mcb2-mcls): the Python loaders' refusals in their order and wording
# (bindings/mcls_common.mojo), then the SAME cuML-backed entry of the
# metrics binding the shell resolved (`_get_binding(numeric_mode)`) with the
# same addresses and parameter lists. The options of the regression errors
# and r2_score run the x_metrics program of bindings/metrics_reg_door.mojo
# itself (`_mean_error`, `_r2_options`), as `regression_error_options` and
# `r2_score_options` did. Same kernels, same operands: the bits do not move.
# The label-pair metrics (rand, adjusted rand, mutual information,
# homogeneity, completeness, v-measure, Fowlkes-Mallows and the fused
# homogeneity_completeness_v_measure; lane mf-cluster5) are UNPACK here and
# THE TYPED RUN of core/metrics_cluster_table.mojo, the one body the search
# scorers' RUN calls: the same image entry bodies, the same union codes.
from bindings.operation_names_mcls import (
    MCLS_OPN_FAMILY_MCLS_ADJUSTED_RAND,
    MCLS_OPN_FAMILY_MCLS_COMPLETENESS,
    MCLS_OPN_FAMILY_MCLS_ENTROPY,
    MCLS_OPN_FAMILY_MCLS_FOWLKES_MALLOWS,
    MCLS_OPN_FAMILY_MCLS_HCV,
    MCLS_OPN_FAMILY_MCLS_HOMOGENEITY,
    MCLS_OPN_FAMILY_MCLS_KL,
    MCLS_OPN_FAMILY_MCLS_MUTUAL_INFO,
    MCLS_OPN_FAMILY_MCLS_R2,
    MCLS_OPN_FAMILY_MCLS_RAND,
    MCLS_OPN_FAMILY_MCLS_REGRESSION_ERROR,
    MCLS_OPN_FAMILY_MCLS_SILHOUETTE_SAMPLES,
    MCLS_OPN_FAMILY_MCLS_SILHOUETTE_SCORE,
    MCLS_OPN_FAMILY_MCLS_TRUSTWORTHINESS,
    MCLS_OPN_FAMILY_MCLS_V_MEASURE,
)
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.entry_table import EntryRef
from core.metrics_cluster_table import (
    CLUSTER_OP_RAND, CLUSTER_OP_ADJUSTED_RAND, CLUSTER_OP_MUTUAL_INFO, CLUSTER_OP_HOMOGENEITY,
    CLUSTER_OP_COMPLETENESS, CLUSTER_OP_V_MEASURE, CLUSTER_OP_FOWLKES_MALLOWS, CLUSTER_OP_HCV,
    cluster_ops_entry, cluster_pair_run, cluster_hcv_run,
)
from bindings.admitted_family_door import admit_family
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays_door import classical_address
from bindings.family_common_unpack import (
    FAMILY_F32P,
    FAMILY_I32P,
    family_repr,
    family_min,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
)
from bindings.metrics_reg_door import _mean_error, _r2_options
from bindings.metrics_native_inputs_door import metric_shape
from bindings.mcls_common_unpack import (
    mc_b, mc_len, mc_float, mc_int, mc_is_str, mc_entry, mc_shape_of, mc_rank, mc_dim, mc_i32_1d, mc_f32_1d,
    mc_regression_f32_1d, mc_pair_check, mc_weights, mc_encode_store, mc_encode_signed,
)
from core.host_finite_scan import host_all_finite_f32
from bindings.metrics_slot_door import mc_slots_open, mc_slots_call, mc_slot_inputs
from bindings.metric_resident_pair_door import mc_resident_caller_pair
from bindings.arena_prog_run import metric_source_word
from metrics.slot_ops import (
    MS_RAND, MS_ARI, MS_ENTROPY, MS_R2, MS_R2_W, MS_MSE, MS_MAE, MS_RMSE, MS_KL, MS_SILHOUETTE, MS_TRUST,
    MS_MI, MS_FM, MS_HOMOG, MS_COMPL, MS_V_MEASURE, MS_HCV,
)
from core.mcls_impl import mcls_min_i32, mcls_max_i32
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


def _cluster_entry(binding: PythonObject) raises -> EntryRef:
    """UNPACK: the metrics image's cluster label-pair entry
    (core/metrics_cluster_table.mojo), resolved from the key the image
    published (`metrics_cluster_key = (mode, vendor)`), once per call."""
    var key = binding.metrics_cluster_key
    return cluster_ops_entry(Int(py=key[0]), Int(py=key[1]))


def _cluster_ms(operation: Int) -> Int:
    """The `metric_slots` operation of a union-code cluster operation."""
    if operation == CLUSTER_OP_MUTUAL_INFO:
        return MS_MI
    if operation == CLUSTER_OP_FOWLKES_MALLOWS:
        return MS_FM
    if operation == CLUSTER_OP_HOMOGENEITY:
        return MS_HOMOG
    if operation == CLUSTER_OP_COMPLETENESS:
        return MS_COMPL
    if operation == CLUSTER_OP_V_MEASURE:
        return MS_V_MEASURE
    return MS_HCV


def _cluster_union_device(binding: PythonObject, operation: Int, args: PythonObject, n: Int,
                          beta: Float64, context: PythonObject) raises -> Optional[PythonObject]:
    """The union-code metrics (mutual information, Fowlkes-Mallows,
    homogeneity, completeness, v-measure, and the three at once) over
    admitted int32 labels of which at least one is resident in the active
    `resident()` scope (lane vdf-xm-wire): the device union step
    (`mc_resident_pair`, MS_ENCODE_PAIR: the codes `cluster_union_codes`
    makes, kept on the device) then the entry over [0, k-1] reading the two
    code slots in place: two entry calls, k and the answer come back, no
    label or code word does. None: the host route (`cluster_pair_run` /
    `cluster_hcv_run`), unchanged. Same codes, same kernels: same bits.
    The pair is the CALLER'S two label vectors (`args[1]`, `args[2]`, the
    door's admitted arguments; bindings/metric_resident_pair_door.mojo), not
    their int32 copies, which had a new address per call and so never hit
    the scope (lane gd-resident-functions: `stats {'x_prep': (4|8, 0)}`)."""
    var state = mc_slots_open(context)
    if not state.resident.active():
        return Optional[PythonObject]()  # outside resident(): one scope read, the host route
    var pair = mc_resident_caller_pair(binding, args[1], args[2], False, state, context)
    if not pair.live():
        return Optional[PythonObject]()
    var addrs: List[Int] = [0, 0]
    var slots: List[Int] = [pair.codes_t, pair.codes_p]
    var ints: List[Int] = [n, 0, pair.k - 1]
    var floats = List[Float64]()
    if operation == CLUSTER_OP_V_MEASURE or operation == CLUSTER_OP_HCV:
        floats.append(beta)
    var result = mc_slots_call(binding, _cluster_ms(operation), addrs, slots, ints, floats, List[Int](), state)
    if operation == CLUSTER_OP_HCV:
        return Optional[PythonObject](result)  # the entry's 3-tuple of floats
    return Optional[PythonObject](_as_float(result))


def _cluster_pair(binding: PythonObject, operation: Int, args: PythonObject, beta_in: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """UNPACK / RUN / PACK of one label-pair metric (lane mf-cluster5): the
    labels admitted as `_pair_1d(a, b, ..., _as_i32_1d)` admits them and
    their addresses read with the GIL held, then THE TYPED RUN
    (core/metrics_cluster_table.mojo `cluster_pair_run`: raw labels for the
    Rand indices, the union codes `_prepare_cluster_labels` made for the
    others, the image's entry) with the GIL released, the search scorers'
    one body. `beta_in` (v_measure's; unread otherwise) is read as `float(beta)`
    after the labels, in the door's order."""
    var pair = _pair_i32(args[1], args[2], "labels_true", "labels_pred", context)
    var yt = pair[0]
    var yp = pair[1]
    var beta = 1.0 if operation != CLUSTER_OP_V_MEASURE else Float64(py=family_call(mc_b().float, beta_in))
    var n = Int(py=yt.size)
    if operation == CLUSTER_OP_RAND or operation == CLUSTER_OP_ADJUSTED_RAND:
        # the Rand indices read RAW labels: resident ones in place (lane vdf-xm-kernels)
        var pair_values: List[PythonObject] = [yt, yp]
        var ints: List[Int] = [n]
        var device = mc_slot_inputs(binding, MS_RAND if operation == CLUSTER_OP_RAND else MS_ARI, pair_values, ints,
                                    List[Float64](), List[Int](), args, UInt64(0),
                                    metric_source_word(0, 1) | metric_source_word(1, 2), context)
        if device:
            return _as_float(device.value())
    else:
        # the union-code metrics: resident labels encoded on the device (lane vdf-xm-wire)
        var device = _cluster_union_device(binding, operation, args, n, beta, context)
        if device:
            return device.value()
    var t = Int(py=classical_address(yt))
    var p = Int(py=classical_address(yp))
    var entry = _cluster_entry(binding)
    var value = Float64(0.0)
    with GILReleased(Python()):
        value = cluster_pair_run(entry, operation, t, p, n, beta)
    _ = yt  # keep the label arrays alive through the RUN above
    _ = yp
    return PythonObject(value)


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
    var lab_values: List[PythonObject] = [lab]
    var lab_ints: List[Int] = [n, Int(lower), Int(upper)]
    var device = mc_slot_inputs(args[0], MS_ENTROPY, lab_values, lab_ints, List[Float64](), List[Int](), args, UInt64(0),
                                 metric_source_word(0, 1), context)
    var value: Float64
    if device:
        value = Float64(py=_as_float(device.value()))
    else:
        value = Float64(py=_as_float(family_call(mc_entry(args[0], "entropy"), classical_address(lab),
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
    var base = mc_float(args[2], context)
    if base <= 0.0:
        raise Error("math domain error")
    var bottom = portable_log64(base)
    if bottom == 0.0:
        # `result / log(base)` with log(base) == 0: Python's own
        # ZeroDivisionError, restored unchanged at the door
        return family_call(native_loaded_module("operator").truediv, PythonObject(top), PythonObject(bottom))
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
    # Device inputs (lane vdf-xm-kernels): a DeviceRows X is gathered on the
    # device just before the entry and refused there if not finite; a host X
    # resident in the active resident() scope is read in place.
    var state = mc_slots_open(context)
    var shape = state.rows_shape(args[0], args[1])
    var on_rows = shape[0] >= 0
    var x = PythonObject(None)
    var rows = shape[0]
    var cols = shape[1]
    var x_slot = -1
    if not on_rows:
        x = admit_family(args[1], 2, "X", context).array
        rows = Int(py=x.shape[0])
        cols = Int(py=x.shape[1])
        if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), rows * cols):
            raise Error("mojolearn " + caller + ": X contains NaN or infinity; refused here "
                        + "(metrics/README.md HAND-OFF ask 3), because a NaN distance "
                        + "reaches sil_op as +0.0 (DEVIATION 656) rather than as an error")
        _ = x  # keep the Python object alive through the native calls above
        x_slot = state.slot_from(args[0], x, args, 1, False)  # a converted copy of X is never kept (lane vdf-xm-wire2)
    var lab = mc_i32_1d(args[2], "labels", context)
    var n_lab = Int(py=lab.size)
    if n_lab != rows:
        raise Error("mojolearn " + caller + ": labels has " + String(n_lab) + " entries and X has "
                    + String(rows) + " rows")
    # `_labels.encode_labels(lab)`: prepare_targets over the int32 labels.
    # p3-labels-metrics: the class table is a native scratch list.
    var store = List[Int64](unsafe_uninit_length=max(n_lab, 1))  # fb-metrics: only the first k words are written (was zero-filled)
    var mapped = family_array(Python.tuple(n_lab), "<i4", context)
    var k = mc_encode_signed(Int(py=classical_address(lab)), 4, n_lab, Int(store.unsafe_ptr()),
                             Int(py=classical_address(mapped)))
    # The class table is written through its integer address by prepare_targets.
    # Retain its owner until that write has completed, even though only k is returned.
    _ = store^
    _ = lab  # keep the Python object alive through the native call above
    _ = mapped  # keep the Python object alive through the native call above
    if k < 0:
        raise Error("RuntimeError: prepare_targets refused integer labels")
    _ = mapped  # keep the Python object alive through the native calls above
    _ = lab  # keep the Python object alive through the native calls above
    var chunk = 40000
    if args[4] is not None:
        chunk = mc_int(args[4])
    if chunk < 1:
        raise Error("mojolearn " + caller + ": chunksize must be at least 1, got " + String(chunk))
    var scores = family_array(Python.tuple(rows), "<f4", context)
    var mean: PythonObject
    if on_rows or x_slot >= 0:
        if on_rows:
            x_slot = state.rows(args[0], args[1])[0]
        var addrs: List[Int] = [0 if on_rows else Int(py=classical_address(x)), Int(py=classical_address(mapped))]
        var slots: List[Int] = [x_slot, -1]
        var ints: List[Int] = [rows, cols, k, chunk, (1 if caller == "silhouette_score" else 2) if on_rows else 0]
        var outs: List[Int] = [Int(py=classical_address(scores))]
        mean = _as_float(mc_slots_call(args[0], MS_SILHOUETTE, addrs, slots, ints, List[Float64](), outs, state))
    else:
        mean = _as_float(family_call(mc_entry(args[0], "silhouette"), classical_address(x), classical_address(mapped),
                                     classical_address(scores), _params(rows, cols, k, chunk)))
    _ = x  # keep the Python object alive through the native call above
    _ = scores  # keep the Python object alive through the native call above
    _ = mapped  # keep the Python object alive through the native call above
    _ = lab  # keep the Python object alive through the native call above
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
    # Device inputs (lane vdf-xm-kernels): X as in `_silhouette`; X_embedded
    # stays a host input (the embedded k-NN's boundary is host pointers).
    var state = mc_slots_open(context)
    var shape = state.rows_shape(args[0], args[1])
    var on_rows = shape[0] >= 0
    var x = PythonObject(None)
    var rows = shape[0]
    var m = shape[1]
    var x_slot = -1
    if not on_rows:
        x = admit_family(args[1], 2, "X", context).array
        rows = Int(py=x.shape[0])
        m = Int(py=x.shape[1])
    var emb = admit_family(args[2], 2, "X_embedded", context).array
    if rows != Int(py=emb.shape[0]):
        raise Error("mojolearn trustworthiness: X has " + String(rows) + " rows and X_embedded has "
                    + String(Int(py=emb.shape[0])))
    if not on_rows:
        if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), Int(py=x.size)):
            raise Error("mojolearn trustworthiness: X contains NaN or infinity; refused here (metrics/README.md HAND-OFF ask 3)")
        _ = x  # keep the Python object alive through the native calls above
        x_slot = state.slot_from(args[0], x, args, 1, False)  # a converted copy of X is never kept (lane vdf-xm-wire2)
    # X_embedded resident in the active resident() scope (lane vdf-xm-wire): the embedded k-NN reads
    # its slot in place and the entry refuses a non-finite one on the device with these words
    var emb_slot = state.slot_from(args[0], emb, args, 2, False)
    if emb_slot < 0:
        if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(emb))), Int(py=emb.size)):
            raise Error("mojolearn trustworthiness: X_embedded contains NaN or infinity; refused here "
                        + "(metrics/README.md HAND-OFF ask 3)")
    _ = emb  # keep the Python object alive through the native calls above
    var batch = mc_int(args[5])
    if batch < 1:
        raise Error("mojolearn trustworthiness: batch_size must be at least 1, got " + family_repr(args[5]))
    var result: PythonObject
    if on_rows or x_slot >= 0 or emb_slot >= 0:
        if on_rows:
            x_slot = state.rows(args[0], args[1])[0]
        var addrs: List[Int] = [0 if on_rows else Int(py=classical_address(x)), Int(py=classical_address(emb))]
        var slots: List[Int] = [x_slot, emb_slot]
        var ints: List[Int] = [rows, m, Int(py=emb.shape[1]), mc_int(args[3]), batch, 1 if on_rows else 0]
        result = _as_float(mc_slots_call(args[0], MS_TRUST, addrs, slots, ints, List[Float64](), List[Int](), state))
    else:
        result = _as_float(family_call(mc_entry(args[0], "trustworthiness"), classical_address(x), classical_address(emb),
                                       _params(rows, m, Int(py=emb.shape[1]), mc_int(args[3]), batch)))
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
    var kl_values: List[PythonObject] = [p, q]
    var kl_ints: List[Int] = [Int(py=p.size)]
    var device = mc_slot_inputs(args[0], MS_KL, kl_values, kl_ints, List[Float64](), List[Int](), args, UInt64(0),
                                 metric_source_word(0, 1) | metric_source_word(1, 2), context)
    var result: PythonObject
    if device:
        result = _as_float(device.value())
    else:
        result = _as_float(family_call(mc_entry(args[0], "kl_divergence"), classical_address(p), classical_address(q),
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
    var shape = mc_shape_of(args[2], context)
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
    var reg_values: List[PythonObject] = [yt, yp]
    var reg_ints: List[Int] = [Int(py=yt.size)]
    var reg_op = MS_MSE if name == "mean_squared_error" else (MS_MAE if name == "mean_absolute_error" else MS_RMSE)
    var device = mc_slot_inputs(args[0], reg_op, reg_values, reg_ints, List[Float64](), List[Int](), args, UInt64(0),
                                 metric_source_word(0, 2) | metric_source_word(1, 3), context)
    var result: PythonObject
    if device:
        result = _as_float(device.value())
    else:
        result = _as_float(family_call(mc_entry(args[0], name), classical_address(yt), classical_address(yp),
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
    var shape = mc_shape_of(args[1], context)
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
        var w_values: List[PythonObject] = [y, yh, w]
        var w_ints: List[Int] = [n]
        var device = mc_slot_inputs(args[0], MS_R2_W, w_values, w_ints, List[Float64](), List[Int](), args, UInt64(0),
                                     metric_source_word(0, 1) | metric_source_word(1, 2) | metric_source_word(2, 3),
                                     context)
        if device:
            result = _as_float(device.value())
        else:
            result = _as_float(family_call(mc_entry(args[0], "r2_score_weighted"), classical_address(y),
                                           classical_address(yh), classical_address(w), _params(n)))
        _ = y  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = yh  # keep the Python object alive through the native call above
        _ = w^
    else:
        var r2_values: List[PythonObject] = [y, yh]
        var r2_ints: List[Int] = [n]
        var device = mc_slot_inputs(args[0], MS_R2, r2_values, r2_ints, List[Float64](), List[Int](), args, UInt64(0),
                                     metric_source_word(0, 1) | metric_source_word(1, 2), context)
        if device:
            result = _as_float(device.value())
        else:
            result = _as_float(family_call(mc_entry(args[0], "r2_score"), classical_address(y), classical_address(yh),
                                           _params(n)))
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
    _ = y^
    _ = yh^
    return result


def mcls_cluster_operation(operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The label-pair, divergence, silhouette, trustworthiness and
    regression metrics; args[0] is the metrics binding."""
    var b = mc_b()
    if operation == MCLS_OPN_FAMILY_MCLS_RAND:
        return _cluster_pair(args[0], CLUSTER_OP_RAND, args, PythonObject(None), context)
    if operation == MCLS_OPN_FAMILY_MCLS_ADJUSTED_RAND:
        return _cluster_pair(args[0], CLUSTER_OP_ADJUSTED_RAND, args, PythonObject(None), context)
    if operation == MCLS_OPN_FAMILY_MCLS_ENTROPY:
        return _entropy(args, context)
    if operation == MCLS_OPN_FAMILY_MCLS_MUTUAL_INFO:
        # args = (binding, labels_true, labels_pred, contingency)
        if args[3] is not None:
            raise Error("NotImplementedError: mojolearn mutual_info_score: contingency= is refused; the "
                        + "implemented entry builds the contingency matrix on the device "
                        + "(metrics/impl/stats/detail/contingency_matrix.mojo) and has "
                        + "no arm that consumes a precomputed one")
        return _cluster_pair(args[0], CLUSTER_OP_MUTUAL_INFO, args, PythonObject(None), context)
    if operation == MCLS_OPN_FAMILY_MCLS_FOWLKES_MALLOWS:
        # args = (binding, labels_true, labels_pred, sparse)
        if not (mc_is_str(args[3]) and String(args[3]) == "deprecated"):
            if Bool(py=family_call(native_loaded_module("operator").ne, args[3], PythonObject("deprecated"))):
                raise Error("NotImplementedError: mojolearn fowlkes_mallows_score: sparse= is refused; scikit-learn "
                            + "deprecated and ignores it, and this surface has no sparse arm")
        var empty = False
        try:
            # len(Array) invokes its Python __len__. Capture the admitted
            # native shape once for each target before the empty-input branch.
            var true_shape = metric_shape(args[1], context)
            var pred_shape = metric_shape(args[2], context)
            empty = (len(true_shape) > 0 and len(pred_shape) > 0
                     and Int(py=true_shape[0]) == 0 and Int(py=pred_shape[0]) == 0)
        except:
            empty = False
        if empty:
            return PythonObject(0.0)
        return _cluster_pair(args[0], CLUSTER_OP_FOWLKES_MALLOWS, args, PythonObject(None), context)
    if operation == MCLS_OPN_FAMILY_MCLS_HOMOGENEITY:
        return _cluster_pair(args[0], CLUSTER_OP_HOMOGENEITY, args, PythonObject(None), context)
    if operation == MCLS_OPN_FAMILY_MCLS_COMPLETENESS:
        return _cluster_pair(args[0], CLUSTER_OP_COMPLETENESS, args, PythonObject(None), context)
    if operation == MCLS_OPN_FAMILY_MCLS_V_MEASURE:
        # args = (binding, labels_true, labels_pred, beta)
        return _cluster_pair(args[0], CLUSTER_OP_V_MEASURE, args, args[3], context)
    if operation == MCLS_OPN_FAMILY_MCLS_HCV:
        # args = (binding, labels_true, labels_pred, beta): one union encode and ONE entry call (the image's fused body,
        # N356: the same three values the three single entries return),
        # through the typed RUN with the GIL released (lane mf-cluster5).
        var pair = _pair_i32(args[1], args[2], "labels_true", "labels_pred", context)
        var yt = pair[0]
        var yp = pair[1]
        var beta = Float64(py=family_call(b.float, args[3]))
        var n = Int(py=yt.size)
        var device = _cluster_union_device(args[0], CLUSTER_OP_HCV, args, n, beta, context)
        if device:
            return device.value()
        var t = Int(py=classical_address(yt))
        var p = Int(py=classical_address(yp))
        var entry = _cluster_entry(args[0])
        var hcv = (Float64(0.0), Float64(0.0), Float64(0.0))
        with GILReleased(Python()):
            hcv = cluster_hcv_run(entry, t, p, n, beta)
        _ = yt  # keep the label arrays alive through the RUN above
        _ = yp
        return Python.tuple(PythonObject(hcv[0]), PythonObject(hcv[1]), PythonObject(hcv[2]))
    if operation == MCLS_OPN_FAMILY_MCLS_KL:
        return _kl_divergence(args, context)
    if operation == MCLS_OPN_FAMILY_MCLS_SILHOUETTE_SCORE:
        # args = (binding, X, labels, metric, sample_size, random_state, chunksize)
        if args[4] is not None or args[5] is not None:
            raise Error("NotImplementedError: mojolearn silhouette_score: sample_size / random_state are "
                        + "refused; subsample X yourself so the rows that were scored are "
                        + "the rows you chose (cuML's surface has no subsampling either)")
        return _silhouette(Python.tuple(args[0], args[1], args[2], args[3], args[6]), "silhouette_score", context)[0]
    if operation == MCLS_OPN_FAMILY_MCLS_SILHOUETTE_SAMPLES:
        # args = (binding, X, labels, metric, chunksize)
        return _silhouette(args, "silhouette_samples", context)[1]
    if operation == MCLS_OPN_FAMILY_MCLS_TRUSTWORTHINESS:
        return _trustworthiness(args, context)
    if operation == MCLS_OPN_FAMILY_MCLS_REGRESSION_ERROR:
        return _regression_error(args, context)
    if operation == MCLS_OPN_FAMILY_MCLS_R2:
        return _r2(args, context)
    raise Error("RuntimeError: unknown family operation " + String(operation))


def mcls_cluster_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The named form, for the one caller that still passes the name (lane
    bw2-metrics): bindings/neighbors_knn_native_door.mojo `_reg_score`, a
    file an open A/B arm reads, passes "family_mcls_r2". Every other caller
    passes the leaf code. Delete this form when that caller passes
    MCLS_OPN_FAMILY_MCLS_R2."""
    if operation == "family_mcls_r2":
        return mcls_cluster_operation(MCLS_OPN_FAMILY_MCLS_R2, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
