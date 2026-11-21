# SPDX-License-Identifier: Apache-2.0
# SVC `fit`, `decision_function`, `predict`, `predict_proba`,
# `predict_log_proba`, `score` and SVR `fit`, `predict`, `score`
# (python/mojolearn/_svm_impl.py), one native call each.
#
# This owner is the Python it replaces: `_validate_params` with its refusals
# in the same wording and order, `_as_labels` / `_as_targets`, the exact
# gamma (`_resolve_gamma`: the binding's `scale_gamma_limbs`, then
# core/scale_gamma_exact.mojo's once-rounded rational), the per-row bounds
# of `_c_rows`, and the solver driver: the binary machine, the one-vs-one
# machines over the class pairs (`# driver-loop`), libsvm's Platt
# cross-validation per pair (five folds), then the decision, vote and
# probability epilogues. Every number comes from the same binding entries
# (`svc_fit`, `svc_predict`, `svc_pair_epilogue`, `svc_splitmix_perm`,
# `svc_platt_train`, `svc_ovo_layout`, `svr_fit`, `svr_predict`) called with
# the same parameter lists on the same operands in the same order, so the
# bits do not move. An intermediate one entry writes and the next reads
# (a pair's rows, a fold's held rows, the fold decisions) stays the buffer
# it was written into; the Python's slice copies of their prefixes are not
# made (the entries read only those prefixes).
#
# The binding keeps no device state between calls (DEVIATION 873): `predict`
# hands the fitted support vectors over at every call, as before. There is
# no resident entry for them.
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family, admit_family_copied, admitted_array
from core.admitted_input import admitted_cast_f32
from core.cast_in import CAST_KIND_SIGNED
from core.admitted_convert import admitted_convert_vector
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.label_admission_native_unpack import labels_primitive_kind
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields, py_item
from bindings.classical_arrays_door import classical_address, classical_typed
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.gp_svm_records_door import gpsvm_no_fit_callbacks
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.classical_labels_door import labels_cast, classical_encode_labels
from bindings.native_array_view_pack import native_array_ndarray_view
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from bindings.family_common_unpack import FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, family_repr
from bindings.family_common_pack import family_call, family_array
from std.python._cpython import GILReleased
from core.svc_native_api import SvcDrivers
from svm.svc_kernel_native_run import svc_query_run, svc_coef_run, svc_gather_columns_run
from bindings.svc_kernel_provider_unpack import SvcConnection, svc_connection, svc_connection_publish
from bindings.svc_kernel_query_unpack import svc_query_unpack, svc_query_refused, svc_coef_unpack, SVC_QUERY_DECISION, SVC_QUERY_CODES, SVC_QUERY_PROBA, SVC_QUERY_LOG_PROBA
from bindings.svc_kernel_query_pack import svc_query_result, svc_query_pack, svc_coef_output
from bindings.svc_kernel_svr_unpack import svr_fit_unpack, svr_query_unpack, svr_query_refused, svr_score_unpack
from bindings.svc_kernel_svr_pack import svr_fit_pack, svr_query_output, svr_score_pack
from bindings.svc_common_unpack import svc_builtins, svc_field, svc_float, svc_int, svc_truthy, svc_read_f64, svc_shape_of
from bindings.svc_common_pack import svc_prefix, svc_scalar_array, svc_decode, svc_threshold_codes
from bindings.svc_common_door import svc_accuracy_codes
from bindings.svc_common_run import svc_round_f32
from bindings.device_rows_door_run import DoorRowsStore
from providers.python.svc_descriptor_pyapi_door import svc_epilogue_execute, svc_platt_execute, svc_splitmix_execute, svc_descriptor_execute, svc_ovo_execute
from providers.python.compiled_binding_snapshot_pyapi_door import CompiledBindingSnapshot
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from providers.python.classical_native_dispatch import CLASSICAL_NATIVE_DISPATCH
from svm.operation_metadata import SvcOperationSettings, SVC_ABI_NAME
from bindings.operation_names_svc import SVC_OPN_FAMILY_SVC_SVC_COEF, SVC_OPN_FAMILY_SVC_SVC_DECISION, SVC_OPN_FAMILY_SVC_SVC_FIT, SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA, SVC_OPN_FAMILY_SVC_SVC_PREDICT, SVC_OPN_FAMILY_SVC_SVC_PROBA, SVC_OPN_FAMILY_SVC_SVC_SCORE, SVC_OPN_FAMILY_SVC_SVC_VALIDATE, SVC_OPN_FAMILY_SVC_SVR_FIT, SVC_OPN_FAMILY_SVC_SVR_PREDICT, SVC_OPN_FAMILY_SVC_SVR_SCORE, SVC_OPN_FAMILY_SVC_SVR_VALIDATE
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry, neural_descriptor_connect, neural_descriptor_execute, neural_descriptor_info
from bindings.svm_keys_unpack import svm_keys
from bindings.svc_kernel_rows_unpack import _GLUE_FOLD, _GLUE_FOLD_FINISH, _addr, _entry, _gather, _gather_i32, _gather_support_columns, _pair_select, _svm_fit_admit, _svm_fit_entry, _svm_fit_host_rows, _svm_fit_refuse, _svm_query_refuse, _svm_x_scan
from bindings.svc_kernel_params_unpack import _c_rows, _resolve_gamma, _svm_validate, svm_settings_unpack

comptime _DEV636_Y = "mojolearn SVC: y contains a non-finite value (DEVIATION 636: a NaN or inf cannot be fitted; a computed NaN carries a vendor-specific payload and cannot sit in a hashed stage)"


def _svc_provider(binding: PythonObject) raises -> PythonObject:
    # One common provider admission per public operation, never a pair/fold
    # cache lookup. Retain its checked module/table until the operation ends.
    # The route is the build's (providers/python/classical_native_dispatch.mojo):
    # default None = by-name entries with the device-resident fitted state;
    # the environment read this replaced ran once per public SVC/SVR operation.
    if not CLASSICAL_NATIVE_DISPATCH:
        return PythonObject(None)
    var factory = neural_descriptor_entry(binding, "svm_binary_api_v1", False)
    if factory is None:
        raise Error("SVC native dispatch candidate is unavailable for this binding")
    var native = neural_descriptor_connect(factory, String(SVC_ABI_NAME), 12)
    if neural_descriptor_execute(native, 5, [], [], []) != 1:
        raise Error("SVC native dispatch candidate requires identical mode")
    var flags = neural_descriptor_execute(native, 8, [], [], [])
    return PythonObject(alloc=CompiledBindingSnapshot(native, binding, String(SVC_ABI_NAME), 1, [flags]))


def _svc_publish_dispatch(fields: PythonObject, native: PythonObject, context: PythonObject) raises:
    if native is not None:
        # Plain diagnostic values only: never persist module/capsule owners or
        # make the native table part of a model's pickle/checkpoint state.
        var nk = svm_keys()
        if py_contains(fields, nk[].k__native_svc_dispatch):
            native_release_admit(py_item(fields, nk[].k__native_svc_dispatch), context)
        py_setitem(fields, nk[].k__native_svc_dispatch, neural_descriptor_info(native.downcast_value_ptr[CompiledBindingSnapshot]()[].provider))


# ---------------------------------------------------------------- the machines


def _solve(fields: PythonObject, binding: PythonObject, x_addr: Int, n_rows: Int, n_cols: Int, labels_addr: Int,
                gamma: Float64, c_rows_addr: Int, context: PythonObject,
                settings: SvcOperationSettings, native: PythonObject, fold: DoorRowsStore, scale_gamma: Bool,
                device_form: Bool) raises -> PythonObject:
    """`SVC._solve`: ONE binary C-SVC solve; (n_support, dual, support, sv,
    info) with the worst-case buffers uncut. `c_rows_addr` 0 = unweighted.
    `device_form` (the GPU binding, lane a8a) calls `svc_fit`'s
    15-value form: X is the host words at `x_addr` or the fold `fold`
    gathered on the device, gamma='scale' (`scale_gamma`) is resolved from
    the fit's device copy, and info's slots 5-7 return X's finite and range
    words and the gamma solved with. A flagged X is refused here in
    admit_x's wording ('Input X contains NaN or infinity (SVC).')."""
    var dual = family_array(Python.tuple(PythonObject(n_rows)), "<f4", context)
    var support = family_array(Python.tuple(PythonObject(n_rows)), "<i4", context)
    var sv = family_array(Python.tuple(PythonObject(n_rows * n_cols)), "<f4", context)
    var info = family_array(Python.tuple(PythonObject(8)), "<f8", context)
    # ORDER MATCHES bindings/_mojolearn_svm.mojo::svc_fit_binding: n_rows,
    # n_features, kernel, gamma, C, tol, max_iter, nochange_steps, degree,
    # coef0[, the per-row bounds' address[, the fold's three store ids,
    # scale_gamma]]
    var n_support = PythonObject(None)
    if native is not None:
        n_support = PythonObject(svc_descriptor_execute(native, 0,
            [x_addr, labels_addr, _addr(dual), _addr(support), _addr(sv), _addr(info), c_rows_addr],
            settings.fit_ints(n_rows, n_cols), settings.fit_floats()))
    else:
        var params = Python.list(n_rows, n_cols, settings.kernel, settings.gamma, settings.c, settings.tol,
            settings.max_iter, settings.nochange_steps, settings.degree, settings.coef0)
        if device_form:
            _ = py_call(params.append, PythonObject(c_rows_addr))
            _ = py_call(params.append, PythonObject(fold.base_id))
            _ = py_call(params.append, PythonObject(fold.base_rows))
            _ = py_call(params.append, PythonObject(fold.index_id))
            _ = py_call(params.append, PythonObject(1 if scale_gamma else 0))
        elif c_rows_addr != 0:
            _ = py_call(params.append, PythonObject(c_rows_addr))
        n_support = family_call(_entry(binding, "svc_fit", context), PythonObject(x_addr), PythonObject(labels_addr),
            classical_address(dual), classical_address(support), classical_address(sv), classical_address(info), params)
    _ = sv  # keep the Python object alive through the native call above
    _ = support  # keep the Python object alive through the native call above
    _ = dual  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    if device_form:
        _svm_fit_refuse(Int(svc_read_f64(info, 5)), Int(svc_read_f64(info, 6)), fold, "SVC")
    return Python.tuple(PythonObject(svc_int(n_support)), dual, support, sv, info)


def _machine(fields: PythonObject, binding: PythonObject, q: PythonObject, dual: PythonObject, sv: PythonObject,
             n_support: Int, bias: Float64, label0: Float64, label1: Float64, predict_class: Bool,
             support: PythonObject, out_addr: Int, context: PythonObject,
             settings: SvcOperationSettings, native: PythonObject, n_rows: Int, query_cols: Int, query_address: Int,
             predict_entry: PythonObject, lease: Int, slot: Int, resident_entry: PythonObject,
             fold: DoorRowsStore = DoorRowsStore.none()) raises -> PythonObject:
    """`SVC._machine`: one machine's decisions (or labels) on q, written at
    `out_addr` (n float32). `predict_entry` is the binding's resolved
    `svc_predict` (None: resolved here), so a pair loop looks it up once.
    `resident_entry` is the binding's `svc_predict_resident` (None: the
    binding has none, or the descriptor route runs): the machine's support
    rows and duals are then read from slot `slot` of `lease` (0: uploaded for
    this call only), the query and the output in place (lane ne1-a-gp-svm).
    Returns the resident entry's [finite, range] words of the query (lane
    a7a; the caller refuses through `_svm_query_refuse`), None for the
    other routes (they test the query themselves). `fold` (active only with
    the resident entry, a computed kernel and support rows): the query is a
    cross-validation fold the entry gathers on the device."""
    var n_features = settings.features
    var query = q
    var query_addr = query_address
    var vectors = sv
    var words = PythonObject(None)
    if settings.kernel == 4 and n_support > 0:
        # X is the cross-kernel against the TRAINING rows; the decision
        # reads its columns at the support vectors and no support rows.
        query = _gather_support_columns(binding, query_address, n_rows, query_cols, support, n_support, context, native)
        query_addr = _addr(query)
        n_features = n_support
        vectors = dual
    # ORDER MATCHES bindings/_mojolearn_svm.mojo::svc_predict_binding:
    # n_rows, n_features, n_support, b, classes[0], classes[1], kernel,
    # gamma, predict_class, cache_size_mib, degree, coef0
    if native is not None:
        _ = svc_descriptor_execute(native, 1,
            [query_addr, _addr(dual) if n_support > 0 else 0,
             _addr(vectors) if n_support > 0 else 0, out_addr],
            settings.predict_ints(n_rows, n_features, n_support, predict_class),
            settings.predict_floats(bias, label0, label1))
    else:
        var params = Python.list(n_rows, n_features, n_support, bias, label0, label1, settings.kernel,
            settings.gamma, 1 if predict_class else 0, settings.cache_mib, settings.degree, settings.coef0)
        var entry = PythonObject(None)
        if resident_entry is not None:
            # ORDER MATCHES bindings/_mojolearn_svm.mojo::svc_predict_resident_binding:
            # svc_predict's 12, then lease, slot
            _ = py_call(params.append, PythonObject(lease))
            _ = py_call(params.append, PythonObject(slot))
            if fold.active():
                # then the fold's store ids (no query address)
                _ = py_call(params.append, PythonObject(fold.base_id))
                _ = py_call(params.append, PythonObject(fold.base_rows))
                _ = py_call(params.append, PythonObject(fold.index_id))
            entry = resident_entry
        else:
            entry = predict_entry if predict_entry is not None else _entry(binding, "svc_predict", context)
        var result = family_call(entry, PythonObject(query_addr),
            classical_address(dual) if n_support > 0 else PythonObject(0),
            classical_address(vectors) if n_support > 0 else PythonObject(0), PythonObject(out_addr), params)
        if resident_entry is not None:
            words = result
    _ = query  # keep the Python object alive through the native call above
    _ = vectors  # keep the Python object alive through the native call above
    _ = dual
    _ = support
    _ = native
    _ = query^
    _ = vectors^
    return words


# ---------------------------------------------------------------- SVC fit


def _as_labels(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_svm_impl._as_labels`: (float32 labels or None, classes, the label
    pair or the int32 codes when more than two classes)."""
    var b = svc_builtins()
    # The one label encoder (classical_encode_labels; lane a2-admit-family
    # deleted the SVM's svc_encode_1d), after the 1-D test it made first.
    var shape = PythonObject(None)
    var classes = PythonObject(None)
    var codes = PythonObject(None)
    try:
        shape = svc_shape_of(y, context)
        if len(shape) == 1:
            var encoded = classical_encode_labels(y, context)
            classes = encoded.classes
            codes = encoded.codes
    except error:
        if String(error).find("NaN label") < 0:
            raise error
        raise Error(_DEV636_Y)
    if len(shape) != 1:
        raise Error("mojolearn SVC: y must be 1-D, got " + String(len(shape)) + "-D shape " + family_repr(shape))
    if len(classes) < 2:
        raise Error("mojolearn SVC: y has " + String(len(classes)) + " class; at least two are needed")
    if len(classes) > 2:
        return Python.tuple(PythonObject(None), classes, codes)
    var real = True
    for i in range(len(classes)):  # small-loop(classes: the two labels)
        if labels_primitive_kind(py_type(classes[i]), context) < 2:
            real = False
    if not real:
        # non-numeric labels: the solver sees each row's code, 0.0 or 1.0
        var n = Int(py=codes.size)
        # The int32 codes (0 or 1) as float32, one cast in row tasks into an
        # uninitialised Array (lane a2-admit-family: was a call into the base
        # binding's cast_elements over a zero-filled Array; codes 0 and 1
        # convert exactly either way).
        var f = admitted_array(Python.tuple(PythonObject(n)), "<f4", context)
        if n > 0:
            _ = admitted_cast_f32(Int(py=classical_address(codes)), 4, 0, n, 1, 4, CAST_KIND_SIGNED,
                                  FAMILY_F32P(unsafe_from_address=Int(py=classical_address(f))))
            _ = codes  # keep the Python object alive through the native call above
            _ = f  # keep the Python object alive through the native call above
        _ = codes^
        return Python.tuple(f, classes, Python.tuple(PythonObject(0.0), PythonObject(1.0)))
    # The finiteness test rides in the admission (lane a2-admit-family: was a
    # second pass through the base binding's all_finite_f32).
    var f = admit_family(y, 1, "y", context, False, finite=True, refusal=_DEV636_Y).array
    return Python.tuple(f, classes, Python.tuple(classes[0], classes[1]))


def _set_binary(fields: PythonObject, solved: PythonObject, n_cols: Int, context: PythonObject) raises:
    """The binary machine's public attributes from its uncut buffers."""
    var n_support = Int(py=solved[0])
    var info = solved[4]
    var nk = svm_keys()
    py_setitem(fields, nk[].k_n_support_, PythonObject(n_support))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(svc_read_f64(info, 2))))
    py_setitem(fields, nk[].k_intercept_, svc_scalar_array(svc_round_f32(svc_read_f64(info, 0)), "<f4", context))
    # compact C-order Arrays detached from the worst-case buffers
    py_setitem(fields, nk[].k_support_,
               svc_prefix(solved[2], n_support, Python.tuple(PythonObject(n_support)), "<i4", context))
    py_setitem(fields, nk[].k_support_vectors_,
               svc_prefix(solved[3], n_support * n_cols, Python.tuple(PythonObject(n_support), PythonObject(n_cols)), "<f4", context))
    py_setitem(fields, nk[].k_dual_coef_,
               svc_prefix(solved[1], n_support, Python.tuple(PythonObject(1), PythonObject(n_support)), "<f4", context))


def _platt(fields: PythonObject, binding: PythonObject, x_meta: ArrayMetadata, codes: PythonObject, gamma: Float64,
           c_rows: PythonObject, k: Int, context: PythonObject,
           settings: SvcOperationSettings, native: PythonObject) raises:
    """`SVC._fit_probability`: libsvm's `svm_binary_svc_probability` per class
    pair, five shuffled folds, then `sigmoid_train`; sets probA_ / probB_."""
    var b = svc_builtins()
    var operator = native_loaded_module("operator")
    var nk = svm_keys()
    if py_contains(fields, nk[].k__prob_ab):
        _ = family_call(fields.pop, PythonObject("_prob_ab"), PythonObject(None))
    if not svc_truthy(svc_field(fields, nk[].k__hp_probability, "_hp_probability")):
        return
    var rs = svc_field(fields, nk[].k__hp_random_state, "_hp_random_state")
    var seed = PythonObject(0) if rs is None else family_call(b.int, rs)
    var mask = family_call(operator.sub, family_call(operator.lshift, PythonObject(1), PythonObject(64)), PythonObject(1))
    var precomputed = settings.kernel == 4
    var n_all = Int(py=codes.shape[0])
    var x_rows = x_meta.shape[0]
    var x_cols = x_meta.shape[1]
    var ab = Python.list()
    var a_values = family_array(Python.tuple(PythonObject(k * (k - 1) // 2)), "<f8", context)
    var b_values = family_array(Python.tuple(PythonObject(k * (k - 1) // 2)), "<f8", context)
    var pair_no = 0
    # SplitMix64's golden-ratio increment (above the Int64 range, so built
    # as a Python int)
    var golden = family_call(b.int, PythonObject("9E3779B97F4A7C15"), PythonObject(16))
    for ci in range(k):  # driver-loop: over the class pairs (one probability fit each)
        for cj in range(ci + 1, k):  # driver-loop: over the class pairs (one probability fit each)
            # the pair's rows in row order; libsvm's +1 is class ci
            var sel = _pair_select(binding, codes, ci, cj, PythonObject(None), context, native)
            var idx_a = sel[0]
            var lab_a = sel[1]
            var n = Int(py=sel[3])
            var key = family_call(operator.xor, seed,
                                  family_call(operator.and_, family_call(operator.mul, PythonObject(pair_no),
                                                                         golden), mask))
            key = family_call(operator.and_, key, mask)
            var perm = family_array(Python.tuple(PythonObject(max(n, 1))), "<i4", context)
            if n > 0:
                _ = svc_splitmix_execute(binding, native, classical_address(perm), PythonObject(n),
                                family_call(operator.and_, key, PythonObject(0xFFFFFFFF)),
                                family_call(operator.rshift, key, PythonObject(32)))
                _ = perm  # keep the Python object alive through the native call above
            var dec_perm = family_array(Python.tuple(PythonObject(max(n, 1))), "<f4", context)
            var consts = family_array(Python.tuple(PythonObject(10)), "<f4", context)
            var cst = FAMILY_F32P(unsafe_from_address=_addr(consts))
            _ = consts  # keep the Python object alive through the native call above
            for f in range(5):  # small-loop(Platt folds: five per pair)
                var begin = f * n // 5
                var end = (f + 1) * n // 5
                var h = end - begin
                var nt = n - h
                var rows_held = family_array(Python.tuple(PythonObject(max(n, 1))), "<i4", context)
                var lab = family_array(Python.tuple(PythonObject(max(nt, 1))), "<f4", context)
                var n_pos = 0
                if n > 0:
                    n_pos = svc_int(svc_epilogue_execute(binding, native, classical_address(idx_a),
                                                classical_address(perm), classical_address(lab_a),
                                                classical_address(rows_held),
                                                Python.list(PythonObject(_GLUE_FOLD), PythonObject(n), PythonObject(begin),
                                                            PythonObject(end), classical_address(lab))))
                    _ = idx_a  # keep the Python object alive through the native call above
                    _ = perm  # keep the Python object alive through the native call above
                    _ = lab_a  # keep the Python object alive through the native call above
                    _ = rows_held  # keep the Python object alive through the native call above
                    _ = lab  # keep the Python object alive through the native call above
                var n_neg = nt - n_pos
                if n_pos == 0 or n_neg == 0:
                    cst[2 * f] = Float32(1.0)
                    _ = consts  # keep the Python object alive through the native call above
                    cst[2 * f + 1] = Float32(0.0) if nt == 0 else (Float32(1.0) if n_pos > 0 else Float32(-1.0))
                    _ = consts  # keep the Python object alive through the native call above
                    continue
                if h == 0:
                    continue
                var rows_addr = _addr(rows_held)
                _ = rows_held  # keep the Python object alive through the native call above
                var hrows_addr = rows_addr + 4 * nt
                _ = rows_held  # keep the Python object alive through the native call above
                var sub: PythonObject
                if precomputed:
                    sub = _gather(binding, x_meta.source.addr, x_rows, x_cols, rows_addr, nt, rows_addr, nt, context, native)
                    _ = rows_held  # keep the Python object alive through the native call above
                else:
                    sub = _gather(binding, x_meta.source.addr, x_rows, x_cols, rows_addr, nt, 0, x_cols, context, native)
                    _ = rows_held  # keep the Python object alive through the native call above
                # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
                var sub_c = Optional[PythonObject]()
                if c_rows is not None:
                    sub_c = _gather(binding, _addr(c_rows), n_all, 1, rows_addr, nt, 0, 1, context, native)
                    _ = rows_held  # keep the Python object alive through the native call above
                var cols = Int(py=sub.shape[1])
                var solved = _solve(fields, binding, _addr(sub), nt, cols, _addr(lab), gamma,
                                    0 if not sub_c else _addr(sub_c.value()), context, settings, native,
                                    DoorRowsStore.none(), False, False)
                _ = lab  # keep the Python object alive through the native call above
                _ = sub_c  # keep the Python object alive through the native call above
                var q: PythonObject
                if precomputed:
                    q = _gather(binding, x_meta.source.addr, x_rows, x_cols, hrows_addr, h, rows_addr, nt, context, native)
                    _ = rows_held  # keep the Python object alive through the native call above
                else:
                    q = _gather(binding, x_meta.source.addr, x_rows, x_cols, hrows_addr, h, 0, x_cols, context, native)
                var n_sv = Int(py=solved[0])
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                var dual = svc_prefix(solved[1], n_sv, Python.tuple(PythonObject(1), PythonObject(n_sv)), "<f4", context)
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                var sv = svc_prefix(solved[3], n_sv * cols, Python.tuple(PythonObject(n_sv), PythonObject(cols)), "<f4", context)
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                var support = svc_prefix(solved[2], n_sv, Python.tuple(PythonObject(n_sv)), "<i4", context)
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                _ = _machine(fields, binding, q, dual, sv, n_sv, svc_round_f32(svc_read_f64(solved[4], 0)), 0.0, 1.0, False,
                         support, _addr(dec_perm) + 4 * begin, context, settings, native, h, nt if precomputed else x_cols, _addr(q),
                         PythonObject(None), 0, 0, PythonObject(None))
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                _ = dec_perm  # keep the Python object alive through the native call above
                _ = sub^
                _ = sub_c^
                _ = q^
                _ = rows_held^
                _ = lab^
            # libsvm's orientation (positive toward class i) and +1/-1
            # labels, back in pair row order, float64
            var dec = family_array(Python.tuple(PythonObject(max(n, 1))), "<f8", context)
            var labels = family_array(Python.tuple(PythonObject(max(n, 1))), "<f8", context)
            if n > 0:
                _ = svc_epilogue_execute(binding, native, classical_address(dec_perm),
                                classical_address(perm), classical_address(lab_a), classical_address(dec),
                                Python.list(PythonObject(_GLUE_FOLD_FINISH), PythonObject(n), classical_address(consts),
                                            classical_address(labels), PythonObject(0)))
                _ = dec_perm  # keep the Python object alive through the native call above
                _ = dec  # keep the Python object alive through the native call above
                _ = perm  # keep the Python object alive through the native call above
                _ = lab_a  # keep the Python object alive through the native call above
                _ = consts  # keep the Python object alive through the native call above
                _ = labels  # keep the Python object alive through the native call above
            var ab_out = family_array(Python.tuple(PythonObject(2)), "<f8", context)
            _ = svc_platt_execute(binding, native,
                            classical_address(dec) if n > 0 else PythonObject(0),
                            classical_address(labels) if n > 0 else PythonObject(0),
                            classical_address(ab_out), PythonObject(n))
            _ = dec  # keep the Python object alive through the native call above
            _ = labels  # keep the Python object alive through the native call above
            _ = ab_out  # keep the Python object alive through the native call above
            var av = svc_read_f64(ab_out, 0)
            var bv = svc_read_f64(ab_out, 1)
            _ = py_call(ab.append, Python.tuple(PythonObject(av), PythonObject(bv)))
            FAMILY_F64P(unsafe_from_address=_addr(a_values))[pair_no] = av
            _ = a_values  # keep the Python object alive through the native call above
            FAMILY_F64P(unsafe_from_address=_addr(b_values))[pair_no] = bv
            _ = b_values  # keep the Python object alive through the native call above
            pair_no += 1
            _ = idx_a^
            _ = lab_a^
            _ = perm^
            _ = dec_perm^
            _ = consts^
    py_setitem(fields, nk[].k__prob_ab, ab)
    py_setitem(fields, nk[].k_probA_, a_values)
    py_setitem(fields, nk[].k_probB_, b_values)


def _fit_ovo(fields: PythonObject, binding: PythonObject, x_meta: ArrayMetadata, classes: PythonObject, codes: PythonObject,
             gamma: Float64, c_rows: PythonObject, context: PythonObject,
             settings: SvcOperationSettings, native: PythonObject) raises:
    """`SVC._fit_ovo` and `_set_ovo_native`: one binary machine per class
    pair (i < j, class j the solver's +1), then scikit-learn's multiclass
    layout from `svc_ovo_layout`."""
    var b = svc_builtins()
    var n_rows = x_meta.shape[0]
    var n_cols = x_meta.shape[1]
    var k = len(classes)
    var nk = svm_keys()
    var precomputed = settings.kernel == 4
    var pairs = Python.list()
    var meta = Python.list(PythonObject(k), PythonObject(n_rows), PythonObject(k * (k - 1) // 2))
    var sup_addrs = Python.list()
    var dual_addrs = Python.list()
    var total = 0
    var intercept = family_array(Python.tuple(PythonObject(k * (k - 1) // 2)), "<f4", context)
    var n_iter = family_array(Python.tuple(PythonObject(k * (k - 1) // 2)), "<i4", context)
    var q_no = 0
    for i in range(k):  # driver-loop: over the class pairs (one binary machine each)
        for j in range(i + 1, k):  # driver-loop: over the class pairs (one binary machine each)
            var sel = _pair_select(binding, codes, i, j, c_rows, context, native)
            var idx_a = sel[0]
            var m = Int(py=sel[3])
            var sub: PythonObject
            if precomputed:
                # the pair's kernel matrix: its rows AND its columns
                sub = _gather(binding, x_meta.source.addr, n_rows, n_cols, _addr(idx_a), m, _addr(idx_a), m, context, native)
                _ = idx_a  # keep the Python object alive through the native call above
            else:
                sub = _gather(binding, x_meta.source.addr, n_rows, n_cols, _addr(idx_a), m, 0, n_cols, context, native)
                _ = idx_a  # keep the Python object alive through the native call above
            var sub_c = sel[2]
            var solved = _solve(fields, binding, _addr(sub), m, Int(py=sub.shape[1]), _addr(sel[1]), gamma,
                                0 if sub_c is None else _addr(sub_c), context, settings, native,
                                DoorRowsStore.none(), False, False)
            _ = sub_c  # keep the Python object alive through the native call above
            var info = solved[4]
            _ = sub_c  # keep the Python object alive through the native call above
            if svc_round_f32(svc_read_f64(info, 3)) != 0.0 or svc_round_f32(svc_read_f64(info, 4)) != 1.0:
                raise Error("RuntimeError: mojolearn SVC: the solver's label pair for classes ("
                            + family_repr(classes[i]) + ", " + family_repr(classes[j]) + ") is not (0.0, 1.0); "
                            + "the class mapping cannot be trusted")
            var n_sv = Int(py=solved[0])
            _ = sub_c  # keep the Python object alive through the native call above
            var sub_cols = Int(py=sub.shape[1])
            var sv_rows = svc_prefix(solved[3], n_sv * sub_cols, Python.tuple(PythonObject(n_sv), PythonObject(sub_cols)),
                                     "<f4", context)
            _ = sub_c  # keep the Python object alive through the native call above
            # the pair's support in X's row numbers, gathered in the binding
            var pair_support = _gather_i32(binding, _addr(idx_a), m, _addr(solved[2]), n_sv, context, native)
            _ = sub_c  # keep the Python object alive through the native call above
            _ = idx_a  # keep the Python object alive through the native call above
            if precomputed:
                # the support rows of the WHOLE kernel matrix
                sv_rows = _gather(binding, x_meta.source.addr, n_rows, n_cols, _addr(pair_support), n_sv, 0, n_cols, context, native)
                _ = pair_support  # keep the Python object alive through the native call above
                _ = idx_a  # keep the Python object alive through the native call above
            var dual = svc_prefix(solved[1], n_sv, Python.tuple(PythonObject(1), PythonObject(n_sv)), "<f4", context)
            _ = sub_c  # keep the Python object alive through the native call above
            var bias = svc_round_f32(svc_read_f64(info, 0))
            var iters = Int(svc_read_f64(info, 2))
            var pair = Python.dict()
            py_setitem(pair, nk[].k_i, PythonObject(i))
            py_setitem(pair, nk[].k_j, PythonObject(j))
            py_setitem(pair, nk[].k_dual, dual)
            py_setitem(pair, nk[].k_sv, sv_rows)
            py_setitem(pair, nk[].k_support, pair_support)
            _ = idx_a  # keep the Python object alive through the native call above
            py_setitem(pair, nk[].k_b, PythonObject(bias))
            py_setitem(pair, nk[].k_n_iter, PythonObject(iters))
            _ = py_call(pairs.append, pair)
            _ = py_call(meta.append, PythonObject(i))
            _ = py_call(meta.append, PythonObject(j))
            _ = py_call(meta.append, PythonObject(n_sv))
            _ = py_call(sup_addrs.append, classical_address(pair_support) if n_sv > 0 else PythonObject(0))
            _ = pair_support  # keep the Python object alive through the native call above
            _ = idx_a  # keep the Python object alive through the native call above
            _ = py_call(dual_addrs.append, classical_address(dual) if n_sv > 0 else PythonObject(0))
            _ = dual  # keep the Python object alive through the native call above
            total += n_sv
            FAMILY_F32P(unsafe_from_address=_addr(intercept))[q_no] = Float32(-bias)
            _ = intercept  # keep the Python object alive through the native call above
            FAMILY_I32P(unsafe_from_address=_addr(n_iter))[q_no] = Int32(iters)
            _ = n_iter  # keep the Python object alive through the native call above
            q_no += 1
            _ = sub^
            _ = sel^
            _ = solved^
            _ = sub_c  # keep the Python object alive through the native call above
    # `_set_ovo_native`: the support_ order, dual_coef_ cells and n_support_
    var cap = min(n_rows, total)
    var support = family_array(Python.tuple(PythonObject(max(1, cap))), "<i4", context)
    var n_support = family_array(Python.tuple(PythonObject(k)), "<i4", context)
    var dual_all = family_array(Python.tuple(PythonObject(max(1, (k - 1) * cap))), "<f4", context)
    var n_sv = svc_int(svc_ovo_execute(binding, native, sup_addrs, dual_addrs, meta,
                                   Python.list(classical_address(support), classical_address(n_support),
                                               classical_address(dual_all), PythonObject(cap))))
    _ = n_support  # keep the Python object alive through the native call above
    _ = support  # keep the Python object alive through the native call above
    _ = dual_all  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_classes_, classes)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(n_cols))
    py_setitem(fields, nk[].k__pairs, pairs)
    if n_sv > 0:
        var sup = svc_prefix(support, n_sv, Python.tuple(PythonObject(n_sv)), "<i4", context)
        _ = dual_all  # keep the Python object alive through the native call above
        _ = n_support  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_support_, sup)
        py_setitem(fields, nk[].k_support_vectors_,
                   _gather(binding, x_meta.source.addr, n_rows, n_cols, _addr(sup), n_sv, 0, n_cols, context, native))
        _ = sup  # keep the Python object alive through the native call above
        _ = dual_all  # keep the Python object alive through the native call above
        _ = n_support  # keep the Python object alive through the native call above
        _ = support  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_dual_coef_,
                   svc_prefix(dual_all, (k - 1) * n_sv, Python.tuple(PythonObject(k - 1), PythonObject(n_sv)), "<f4", context))
        _ = n_support  # keep the Python object alive through the native call above
        _ = support  # keep the Python object alive through the native call above
    else:
        py_setitem(fields, nk[].k_support_, family_array(Python.tuple(PythonObject(0)), "<i4", context))
        py_setitem(fields, nk[].k_support_vectors_,
                   family_array(Python.tuple(PythonObject(0), PythonObject(n_cols)), "<f4", context))
        py_setitem(fields, nk[].k_dual_coef_,
                   family_array(Python.tuple(PythonObject(k - 1), PythonObject(0)), "<f4", context))
    _ = dual_all  # keep the Python object alive through the native calls above
    _ = n_support  # keep the Python object alive through the native calls above
    _ = support  # keep the Python object alive through the native calls above
    py_setitem(fields, nk[].k_intercept_, intercept)
    py_setitem(fields, nk[].k_n_support_, n_support)
    py_setitem(fields, nk[].k_n_iter_, n_iter)
    py_setitem(fields, nk[].k__gamma, PythonObject(gamma))
    py_setitem(fields, nk[].k__label0, PythonObject(0.0))
    py_setitem(fields, nk[].k__label1, PythonObject(1.0))
    _ = support^
    _ = dual_all^
    _platt(fields, binding, x_meta, codes, gamma, c_rows, k, context, settings, native)


def _svc_fit(model: PythonObject, args: PythonObject, context: PythonObject, native: PythonObject) raises:
    """`SVC.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var binding = args[0]
    var fields = py_fields(model)
    var hp = svm_settings_unpack(model, fields, True, context)
    # Lane a8a: the GPU binding's binary fit takes X on the device (a fold
    # gathered there, host words copied once) and tests it there; every
    # other route tests the host words here (`_svm_x_scan`). One wording.
    var entry = _svm_fit_entry(binding, native)
    var fold = DoorRowsStore.none()
    var x_pin = ArrayPin()
    var x_meta = _svm_fit_admit(args[1], "SVC", entry, context, fold, x_pin)
    var nk = svm_keys()
    py_setitem(fields, nk[].k_input_copied_,
               PythonObject(True if fold.active() else admit_family_copied(x_meta, args[1])))
    var x = x_pin.keep
    var y = args[2]
    var parsed = _as_labels(y, context)
    var labels = parsed[0]
    var classes = parsed[1]
    var pair = parsed[2]
    var n_rows = x_meta.shape[0]
    var n_cols = x_meta.shape[1]
    var n_y = Int(py=pair.shape[0]) if labels is None else Int(py=labels.shape[0])
    if n_y != n_rows:
        raise Error("mojolearn SVC: y has " + String(n_y) + " entries, X has " + String(n_rows) + " rows")
    var precomputed = hp.precomputed
    if precomputed and n_cols != n_rows:
        raise Error("mojolearn SVC: kernel='precomputed' needs the square n x n "
                    + "kernel matrix as X, got " + String(n_rows) + " x " + String(n_cols))
    # The device-X route: one binary machine on a computed kernel. The
    # one-vs-one pairs and Platt's folds gather host rows of X, and a
    # precomputed X is gathered by rows and columns: those read host words.
    var device_x = (entry and labels is not None and not precomputed
                    and not hp.probability)
    if fold.active() and not device_x:
        x_meta = _svm_fit_host_rows(args[1], context, x_pin)
        x = x_pin.keep
        fold = DoorRowsStore.none()
    var gamma = 1.0
    var scale_gamma = False
    if device_x:
        var resolved = hp.device_gamma(n_cols)
        gamma = resolved[0]
        scale_gamma = resolved[1]
    else:
        _svm_x_scan(x_meta, "SVC")
        gamma = _resolve_gamma(hp, binding, x_meta, context, native)
    var c_rows = _c_rows(binding, hp.c, n_rows, args[3],
                         py_get(fields, nk[].k_class_weight, PythonObject(None)), y, "SVC", context, native)
    var settings = hp.fit_operation(gamma, n_cols)
    if labels is None:
        _fit_ovo(fields, binding, x_meta, classes, pair, gamma, c_rows, context, settings, native)
        _svc_publish_dispatch(fields, native, context)
        _ = x^
        _ = x_meta  # retain captured storage owners through the last native use
        _ = x_pin
        _ = c_rows^
        return
    if py_contains(fields, nk[].k__pairs):
        _ = family_call(fields.pop, PythonObject("_pairs"), PythonObject(None))
    var solved = _solve(fields, binding, x_meta.source.addr, n_rows, n_cols, _addr(labels), gamma,
                        0 if c_rows is None else _addr(c_rows), context, settings, native, fold, scale_gamma,
                        device_x)
    _ = labels  # keep the Python object alive through the native call above
    var info = solved[4]
    if device_x:
        # the gamma the entry solved with (gamma='scale' from its device copy)
        gamma = svc_read_f64(info, 7)
    _ = labels  # keep the Python object alive through the native call above
    var label0 = svc_round_f32(svc_read_f64(info, 3))
    var label1 = svc_round_f32(svc_read_f64(info, 4))
    # THE LABEL PAIR IS CHECKED, NOT ASSUMED (the solver's own host sort).
    var p0 = svc_float(pair[0])
    var p1 = svc_float(pair[1])
    if label0 != svc_round_f32(p0) or label1 != svc_round_f32(p1):
        raise Error("RuntimeError: mojolearn SVC: the solver's label pair "
                    + "(" + family_repr(PythonObject(label0)) + ", " + family_repr(PythonObject(label1)) + ") does not match the "
                    + "host's sorted unique (" + family_repr(PythonObject(p0)) + ", "
                    + family_repr(PythonObject(p1)) + "); the class mapping cannot be trusted")
    py_setitem(fields, nk[].k_classes_, classes)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(n_cols))
    _set_binary(fields, solved, n_cols, context)
    _ = labels  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k__gamma, PythonObject(gamma))
    py_setitem(fields, nk[].k__label0, PythonObject(label0))
    py_setitem(fields, nk[].k__label1, PythonObject(label1))
    if hp.probability:
        var wide = svc_threshold_codes(labels, label0, True, 0, 1, context)
        var n = Int(py=wide.size)
        var codes = family_array(Python.tuple(PythonObject(n)), "<i4", context)
        if n > 0:
            # The 0 / 1 codes narrowed int64 -> int32 by the typed parallel
            # conversion (core/admitted_convert.mojo), not a py_call into the
            # base binding.
            _ = admitted_convert_vector(Int(py=classical_address(wide)), "<i8", Int(py=classical_address(codes)),
                                        "<i4", n)
            _ = wide  # keep the Python object alive through the native call above
            _ = codes  # keep the Python object alive through the native call above
        _ = wide^
        _platt(fields, binding, x_meta, codes, gamma, c_rows, 2, context, settings, native)
    elif py_contains(fields, nk[].k__prob_ab):
        _ = family_call(fields.pop, PythonObject("_prob_ab"), PythonObject(None))
    _svc_publish_dispatch(fields, native, context)
    _ = x^
    _ = x_meta  # retain captured storage owners through the last native use
    _ = x_pin
    _ = labels^
    _ = c_rows^


# ---------------------------------------------------------------- SVC inference


def _svc_query(model: PythonObject, fields: PythonObject, binding: PythonObject, source: PythonObject, kind: Int,
               context: PythonObject, drivers: SvcDrivers) raises -> PythonObject:
    """One inference operation (`kind`: SVC_QUERY_*): UNPACK, the public
    array, every machine and the epilogue in ONE release of the GIL, the
    refusal of a query the device entry flagged, PACK."""
    var unpacked = svc_query_unpack(model, fields, binding, source, kind, context, drivers)
    var result = svc_query_result(unpacked, context)
    var complete = False
    with GILReleased(Python()):
        complete = svc_query_run(unpacked.drivers, unpacked.plan)
    if not complete:
        svc_query_refused(unpacked)
        raise Error("RuntimeError: mojolearn SVC: the device entry flagged the query")
    var packed = svc_query_pack(unpacked, result, context)
    # The owners of X, the fitted arrays, the scratch and the lease outlive RUN and PACK.
    _ = unpacked
    return packed


# ---------------------------------------------------------------- SVR


def _svr_fit(model: PythonObject, args: PythonObject, context: PythonObject, native: PythonObject) raises:
    """`SVR.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight):
    UNPACK, one typed solve with the GIL released, PACK."""
    var unpacked = svr_fit_unpack(model, args, context, native)
    var count = 0
    with GILReleased(Python()):
        count = unpacked.connection.driver.fit(unpacked.request)
    unpacked.plan.count = count
    svr_fit_pack(unpacked, context)
    # The owners of X, the targets, the bounds and the provider outlive RUN and PACK.
    _ = unpacked


def _svr_predict(model: PythonObject, fields: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject, connection: SvcConnection) raises -> PythonObject:
    """`SVR.predict`: UNPACK, the public vector, then in one release of
    the GIL the cross-kernel's support columns (kernel='precomputed') and
    one typed query; then the refusal of a query the device entry flagged."""
    var unpacked = svr_query_unpack(model, fields, binding, source, context, connection.candidate)
    var result = svr_query_output(unpacked.request.rows, context)
    unpacked.request.dst = _addr(result)
    with GILReleased(Python()):
        if unpacked.gather_columns != 0:
            svc_gather_columns_run(connection.drivers, unpacked.q.address, unpacked.q.rows, unpacked.q.cols,
                                   unpacked.gather_columns, unpacked.gather_count, unpacked.request.x)
        _ = unpacked.connection.driver.query(unpacked.request, unpacked.resident)
    svr_query_refused(unpacked)
    # The owners of X, the fitted arrays, the lease and the provider outlive RUN.
    _ = unpacked
    return result


def _svr_score(model: PythonObject, fields: PythonObject, binding: PythonObject, source: PythonObject, y: PythonObject,
               context: PythonObject, connection: SvcConnection) raises -> PythonObject:
    """`SVR.score`: R^2 of the prediction against y."""
    var pred = _svr_predict(model, fields, binding, source, context, connection)
    svr_score_unpack(Int(py=pred.shape[0]), y, context)
    return svr_score_pack(pred, y, context)


def _linear_coef(fields: PythonObject, classifier: Bool, context: PythonObject, drivers: SvcDrivers) raises -> PythonObject:
    """Public `coef_` and composed RFE: UNPACK (every fitted buffer captured
    once), one GEMV per fitted machine with the GIL released, the matrix."""
    var unpacked = svc_coef_unpack(fields, classifier, context)
    var result = svc_coef_output(len(unpacked.rows), unpacked.features, context)
    var address = _addr(result)
    with GILReleased(Python()):
        svc_coef_run(drivers, unpacked.rows, unpacked.features, unpacked.sign, address)
    _ = unpacked
    return result


# ---------------------------------------------------------------- dispatch


def _svm_lifecycle_admit(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    """Fit bookkeeping read by the public lifecycle: exact native values,
    and no Python completion callback a native fit could be asked to run."""
    var b = native_loaded_module("builtins")
    var nk = svm_keys()
    if py_type(py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0))) is not b.int:
        raise Error("TypeError: SVM fitting depth requires an exact integer")
    var fitted = py_get(fields, nk[].k__mojolearn_fitted, PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: SVM fitted state requires a native boolean or integer")
    # The ancestry scan runs once per class state (bindings/gp_svm_records.mojo).
    gpsvm_no_fit_callbacks(model, fields, "TypeError: SVM fitting cannot invoke Python completion callbacks")


def _svc_method(operation: Int) -> String:
    """The public method a kernel-machine operation serves, for the
    not-fitted text (a static literal; the old door built it per call from
    five suffix tests)."""
    if operation == SVC_OPN_FAMILY_SVC_SVC_DECISION:
        return "decision_function"
    if operation == SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA:
        return "predict_log_proba"
    if operation == SVC_OPN_FAMILY_SVC_SVC_PROBA:
        return "predict_proba"
    if operation == SVC_OPN_FAMILY_SVC_SVC_PREDICT or operation == SVC_OPN_FAMILY_SVC_SVR_PREDICT:
        return "predict"
    if operation == SVC_OPN_FAMILY_SVC_SVC_SCORE or operation == SVC_OPN_FAMILY_SVC_SVR_SCORE:
        return "score"
    return "fit"


def svc_kernel_operation(model: PythonObject, operation: Int, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """The kernel machines' whole public operation: argument admission, the
    fitted and feature-name checks, fit bookkeeping (depth, fitted flag,
    feature names, public class labels), the algorithm through one admitted
    provider, and NumPy results for callers outside the package.
    `operation` is the leaf code of bindings/operation_names_svc.mojo."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: SVM public arguments require an exact tuple")
    var fields = py_fields(model)
    var classifier = operation >= SVC_OPN_FAMILY_SVC_SVC_COEF and operation <= SVC_OPN_FAMILY_SVC_SVC_VALIDATE
    if operation == SVC_OPN_FAMILY_SVC_SVC_VALIDATE or operation == SVC_OPN_FAMILY_SVC_SVR_VALIDATE:
        _svm_validate(model, fields, classifier, context)
        return PythonObject(None)
    var fitting = operation == SVC_OPN_FAMILY_SVC_SVC_FIT or operation == SVC_OPN_FAMILY_SVC_SVR_FIT
    var public = (operation == SVC_OPN_FAMILY_SVC_SVC_DECISION or operation == SVC_OPN_FAMILY_SVC_SVC_PREDICT
                  or operation == SVC_OPN_FAMILY_SVC_SVC_PROBA or operation == SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA
                  or operation == SVC_OPN_FAMILY_SVC_SVC_SCORE or operation == SVC_OPN_FAMILY_SVC_SVR_PREDICT
                  or operation == SVC_OPN_FAMILY_SVC_SVR_SCORE)
    if not fitting and not public:
        # coef_ (the native property schema) and composed callers: algorithm only.
        var connected = svc_connection(args[0])
        var plain = _linear_coef(fields, operation == SVC_OPN_FAMILY_SVC_SVC_COEF, context, connected.drivers)
        svc_connection_publish(fields, connected, context)
        return plain
    _svm_lifecycle_admit(model, fields, context)
    var nk = svm_keys()
    var depth = Int(py=py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0)))
    if public:
        classical_require_fitted(model, False, _svc_method(operation), context)
        if depth == 0:
            estimator_feature_check(model, fields, args[1], context)
    if fitting:
        # The fits still take the by-name / descriptor route of `_svc_provider` (group D).
        var native = _svc_provider(args[0])
        var initial_depth = estimator_fit_begin(fields)
        try:
            if operation == SVC_OPN_FAMILY_SVC_SVC_FIT:
                _svc_fit(model, args, context, native)
            else:
                _svr_fit(model, args, context, native)
        finally:
            estimator_fit_end(model, initial_depth)
        # A nested set_params may have replaced the dictionary's contents.
        fields = py_fields(model)
        _svc_publish_dispatch(fields, native, context)
        estimator_fit_complete(model, args[1], classifier, context)
        return model
    # One provider admission per public operation; RUN takes its typed drivers.
    var connection = svc_connection(args[0])
    var result = _svc_kernel_operation(model, operation, args, context, fields, connection)
    # A nested set_params may have replaced the dictionary's contents.
    fields = py_fields(model)
    svc_connection_publish(fields, connection, context)
    if (operation == SVC_OPN_FAMILY_SVC_SVC_SCORE or operation == SVC_OPN_FAMILY_SVC_SVR_SCORE
            or not estimator_outside() or py_get(context, nk[].k_ndarray) is None
            or py_type(result) is not py_item(context, nk[].k_array_class)):
        return result
    if operation == SVC_OPN_FAMILY_SVC_SVR_PREDICT:
        # scikit-learn's regression predictions are float64: widened exactly.
        var wide = classical_typed(result.shape, "<f8", context)
        labels_cast(Int(py=classical_address(result)), "<f4", Int(py=classical_address(wide)), "<f8",
                    Int(py=result.size), context)
        _ = result^
        return wide
    return native_array_ndarray_view(result, context)


def _svc_kernel_operation(model: PythonObject, operation: Int, args: PythonObject,
                          context: PythonObject, fields: PythonObject,
                          connection: SvcConnection) raises -> PythonObject:
    if operation == SVC_OPN_FAMILY_SVC_SVC_DECISION:
        return _svc_query(model, fields, args[0], args[1], SVC_QUERY_DECISION, context, connection.drivers)
    if operation == SVC_OPN_FAMILY_SVC_SVC_PREDICT:
        # the predicted codes decoded to the fitted classes
        var nk = svm_keys()
        return svc_decode(model, svc_field(fields, nk[].k_classes_, "classes_"),
                          _svc_query(model, fields, args[0], args[1], SVC_QUERY_CODES, context, connection.drivers), context)
    if operation == SVC_OPN_FAMILY_SVC_SVC_PROBA:
        return _svc_query(model, fields, args[0], args[1], SVC_QUERY_PROBA, context, connection.drivers)
    if operation == SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA:
        return _svc_query(model, fields, args[0], args[1], SVC_QUERY_LOG_PROBA, context, connection.drivers)
    if operation == SVC_OPN_FAMILY_SVC_SVC_SCORE:
        # N382: accuracy on codes. y is encoded once against classes_ and
        # compared with the predicted codes in the x_metrics COUNT program
        # (classical_score_from_codes, bindings/classical_labels_door.mojo); the
        # prediction is never decoded to label objects and re-encoded. The
        # same matched count over the same row count: no bit change.
        var codes = _svc_query(model, fields, args[0], args[1], SVC_QUERY_CODES, context, connection.drivers)
        var nk = svm_keys()
        return PythonObject(svc_accuracy_codes(args[2], svc_field(fields, nk[].k_classes_, "classes_"), codes,
                                               Int(py=codes.shape[0]), context))
    if operation == SVC_OPN_FAMILY_SVC_SVR_PREDICT:
        return _svr_predict(model, fields, args[0], args[1], context, connection)
    if operation == SVC_OPN_FAMILY_SVC_SVR_SCORE:
        return _svr_score(model, fields, args[0], args[1], args[2], context, connection)
    raise Error("RuntimeError: unknown kernel SVM operation " + String(operation))
