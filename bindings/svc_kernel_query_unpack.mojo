# SPDX-License-Identifier: Apache-2.0
# UNPACK of the kernel SVC's inference (`decision_function`, `predict`,
# `predict_proba`, `predict_log_proba`, the codes `score` compares) and of
# `coef_` (SVC and SVR): the fitted test, the admitted query X, the fitted
# machines with their owners, the resident lease, the scratch buffers, and
# the typed plan RUN reads (svm/svc_kernel_native_run.mojo). Every field of
# the fitted state is read here, once; RUN sees addresses and plain words.
# The door is bindings/svc_kernel_native_door.mojo.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_item
from bindings.boundary_read_door import buffer_view
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_pandas_door import classical_admit_value
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.classical_arrays_door import classical_is_array
from bindings.family_common_unpack import FAMILY_F64P, FAMILY_I32P
from bindings.family_common_pack import family_array
from bindings.fitted_resident_door import fitted_resident_lease
from bindings.svc_common_unpack import svc_builtins, svc_field, svc_has, svc_float, svc_int, svc_truthy, svc_read_f32
from bindings.svm_keys_unpack import svm_keys
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_shape_once, array_metadata_once
from core.svc_native_api import SvcDrivers
from svm.operation_metadata import SvcOperationSettings
from svm.svc_kernel_native_run import SvcMachine, SvcQueryPlan, SvcCoefRow, SVC_KERNEL_PRECOMPUTED
from bindings.svc_kernel_rows_unpack import _EPI_BINARY, _EPI_LOG_PROBA, _EPI_OVO, _EPI_OVR, _EPI_PROBA, _EPI_VOTES, _SVC_RESIDENT_SLOT, _SvmQuery, _addr, _i32_array, _native_exporter, _query, _svm_query_refuse, svm_device_ok
from bindings.svc_kernel_params_unpack import _kernel_code

# What a query operation returns.
comptime SVC_QUERY_DECISION = 0
comptime SVC_QUERY_CODES = 1
comptime SVC_QUERY_PROBA = 2
comptime SVC_QUERY_LOG_PROBA = 3
# How PACK finishes it (bindings/svc_kernel_query_pack.mojo).
comptime SVC_RESULT_F32_ROWS = 0        # (n,) float32: the binary decisions
comptime SVC_RESULT_F32_PAIRS = 1       # (n, pairs) float32: 'ovo'
comptime SVC_RESULT_F64_CLASSES = 2     # (n, classes) float64: 'ovr', probabilities
comptime SVC_RESULT_CODES = 3           # (n,) int64 codes
comptime SVC_RESULT_ARGMAX = 4          # (n, classes) float64 'ovr' scores, then their row argmax (break_ties)


def svc_predict_settings(fields: PythonObject, context: PythonObject) raises -> SvcOperationSettings:
    """The fitted machine's operation words: the kernel, degree and coef0
    the fit validated, the gamma it resolved, the prediction buffer."""
    var nk = svm_keys()
    return SvcOperationSettings(_kernel_code(fields, context), svc_int(svc_field(fields, nk[].k__hp_degree, "_hp_degree")),
        0, 0, svc_int(svc_field(fields, nk[].k_n_features_in_, "n_features_in_")), svc_float(svc_field(fields, nk[].k__gamma, "_gamma")),
        0.0, 0.0, svc_float(svc_field(fields, nk[].k__hp_coef0, "_hp_coef0")), svc_float(svc_field(fields, nk[].k__hp_cache_size, "_hp_cache_size")))


struct SvcQueryUnpacked(Movable):
    """A query: the plan, the drivers, the admitted X (`q`), the two status
    words of the device entry, and `owners`, a Python list that keeps every
    buffer the plan addresses alive until PACK: the fitted duals, support
    rows and support indices, the decision scratch, the pair codes, the
    sigmoid parameters. `result` says how PACK shapes and finishes the
    public array; `pair_count` and `classes` its dimensions."""
    var q: _SvmQuery
    var owners: PythonObject
    var plan: SvcQueryPlan
    var drivers: SvcDrivers
    var status: List[Int32]
    var result: Int
    var binary: Bool

    def __init__(out self, var q: _SvmQuery, owners: PythonObject, var plan: SvcQueryPlan, drivers: SvcDrivers,
                 var status: List[Int32], result: Int, binary: Bool):
        self.q = q^
        self.owners = owners
        self.plan = plan^
        self.drivers = drivers.copy()
        self.status = status^
        self.result = result
        self.binary = binary


def _keep(owners: PythonObject, value: PythonObject) raises -> Int:
    """Hold `value` for the operation and return its first word's address."""
    _ = py_call(owners.append, value)
    return _addr(value)


def svc_support_columns(support: PythonObject, count: Int, owners: PythonObject,
                        context: PythonObject) raises -> Tuple[Int, Int]:
    """kernel='precomputed': (address, count) of a machine's int32 support
    indices; `count` < 0 takes the index count from `support` itself. A
    C-order int32 exporter (a fitted ndarray or array('i')) is read in
    place: a memoryview held in `owners` keeps its export (so its storage
    cannot move) until the operation ends; no per-element read and no
    packed copy. An int32 Array is used as is; any other sequence is packed
    once (`_i32_array`), as before."""
    if not classical_is_array(support, context) and _native_exporter(support):
        var view = buffer_view(support)
        if view.ndim == 1 and view.matches("<i4"):
            _ = py_call(owners.append, py_call(svc_builtins().memoryview, support))
            var address = view.address()
            var n = count if count >= 0 else view.shape[0]
            view.release()
            return (address, n)
        view.release()
    var held = _i32_array(support, context)
    var n = count if count >= 0 else Int(py=held.shape[0])
    return (_keep(owners, held), n)


def svc_query_unpack(model: PythonObject, fields: PythonObject, binding: PythonObject, source: PythonObject,
                     kind: Int, context: PythonObject, drivers: SvcDrivers) raises -> SvcQueryUnpacked:
    """One inference operation of a fitted SVC (`kind`: SVC_QUERY_*)."""
    var nk = svm_keys()
    var ab = PythonObject(None)
    if kind == SVC_QUERY_PROBA or kind == SVC_QUERY_LOG_PROBA:
        ab = py_get(fields, nk[].k__prob_ab, PythonObject(None))
        if ab is None:
            raise Error("AttributeError: mojolearn SVC: predict_proba is not available when "
                        + "probability=False (scikit-learn's rule); fit with probability=True")
    var pairs = py_get(fields, nk[].k__pairs, PythonObject(None))
    var binary = pairs is None
    var settings = svc_predict_settings(fields, context)
    var precomputed = settings.kernel == SVC_KERNEL_PRECOMPUTED
    var n_machines = 1 if binary else len(pairs)
    # The device entry (a GPU binding's device table) runs every machine.
    var resident = drivers.has_device and n_machines > 0
    var supports = List[Int]()
    if binary:
        supports.append(svc_int(svc_field(fields, nk[].k_n_support_, "n_support_")) if svc_has(fields, nk[].k_dual_coef_) else 0)
    else:
        for pno in range(n_machines):  # driver-loop: over the class pairs (one support count each)
            supports.append(Int(py=py_item(pairs[pno], nk[].k_dual).shape[1]))
    var q = _query(fields, source, context, svm_device_ok(resident, settings.kernel, supports))
    var n_rows = q.rows
    var owners = Python.list()
    var plan = SvcQueryPlan(settings)
    plan.x = q.address
    plan.rows = n_rows
    plan.cols = q.cols
    plan.resident = resident
    plan.base_id = q.store.base_id
    plan.base_rows = q.store.base_rows
    plan.index_id = q.store.index_id
    var status = List[Int32](length=2, fill=Int32(0))
    plan.status = Int(status.unsafe_ptr())
    # The fitted machines. The support rows and duals cross to the device
    # once per fitted state (bindings/fitted_resident_door.mojo; one lease
    # for every machine, slot = its index; precomputed reads none).
    var mirrors = Python.list()
    if binary:
        var dual = svc_field(fields, nk[].k_dual_coef_, "dual_coef_")
        var sv = svc_field(fields, nk[].k_support_vectors_, "support_vectors_")
        _ = py_call(mirrors.append, dual)
        _ = py_call(mirrors.append, sv)
        var bias = svc_read_f32(svc_field(fields, nk[].k_intercept_, "intercept_"), 0)
        plan.label0 = svc_float(svc_field(fields, nk[].k__label0, "_label0"))
        plan.label1 = svc_float(svc_field(fields, nk[].k__label1, "_label1"))
        var support = svc_field(fields, nk[].k_support_, "support_")
        var n = supports[0]
        var columns = 0
        var gathered = 0
        if precomputed and n > 0:
            columns = svc_support_columns(support, n, owners, context)[0]
            gathered = _keep(owners, family_array(Python.tuple(PythonObject(n_rows), PythonObject(n)), "<f4", context))
        plan.machines.append(SvcMachine(_keep(owners, dual) if n > 0 else 0, _keep(owners, sv) if n > 0 else 0, n,
                                        bias, columns, gathered))
    else:
        for pno in range(n_machines):  # driver-loop: over the class pairs (one fitted machine each)
            var p = pairs[pno]
            var dual = py_item(p, nk[].k_dual)
            var sv = py_item(p, nk[].k_sv)
            _ = py_call(mirrors.append, dual)
            _ = py_call(mirrors.append, sv)
            var n = supports[pno]
            var bias = svc_float(py_item(p, nk[].k_b))
            var columns = 0
            var gathered = 0
            if precomputed and n > 0:
                columns = svc_support_columns(py_item(p, nk[].k_support), n, owners, context)[0]
                gathered = _keep(owners, family_array(Python.tuple(PythonObject(n_rows), PythonObject(n)), "<f4", context))
            plan.machines.append(SvcMachine(_keep(owners, dual) if n > 0 else 0, _keep(owners, sv) if n > 0 else 0, n,
                                            bias, columns, gathered))
    if resident and not precomputed:
        plan.lease = fitted_resident_lease(model, _SVC_RESIDENT_SLOT, binding, "svm_resident_lease", mirrors)
    # The epilogue of the operation and its operands.
    var result = SVC_RESULT_F32_ROWS
    if binary:
        if kind == SVC_QUERY_CODES:
            # the machine's labels, then the codes into classes_
            plan.predict_class = True
            plan.epilogue = _EPI_BINARY
            plan.epilogue_label1 = plan.label1
            result = SVC_RESULT_CODES
        elif kind != SVC_QUERY_DECISION:
            plan.epilogue = _EPI_PROBA if kind == SVC_QUERY_PROBA else _EPI_LOG_PROBA
            result = SVC_RESULT_F64_CLASSES
        if kind != SVC_QUERY_DECISION:
            plan.pairs = 1
            var codes = family_array(Python.tuple(PythonObject(2)), "<i4", context)
            FAMILY_I32P(unsafe_from_address=_addr(codes))[1] = 1
            plan.pair_codes = _keep(owners, codes)
            plan.decisions = _keep(owners, family_array(Python.tuple(PythonObject(n_rows)), "<f4", context))
    else:
        plan.pairs = n_machines
        plan.decisions = _keep(owners, family_array(Python.tuple(PythonObject(n_machines), PythonObject(n_rows)), "<f4", context))
        var codes = family_array(Python.tuple(PythonObject(max(1, 2 * n_machines))), "<i4", context)
        var op = FAMILY_I32P(unsafe_from_address=_addr(codes))
        for i in range(n_machines):  # driver-loop: over the class pairs (their two class codes)
            op[2 * i] = Int32(Int(py=py_item(pairs[i], nk[].k_i)))
            op[2 * i + 1] = Int32(Int(py=py_item(pairs[i], nk[].k_j)))
        plan.pair_codes = _keep(owners, codes)
        if kind == SVC_QUERY_DECISION:
            if String(svc_field(fields, nk[].k_decision_function_shape, "decision_function_shape")) == "ovo":
                plan.epilogue = _EPI_OVO
                result = SVC_RESULT_F32_PAIRS
            else:
                plan.epilogue = _EPI_OVR
                result = SVC_RESULT_F64_CLASSES
        elif kind == SVC_QUERY_CODES:
            if svc_truthy(svc_field(fields, nk[].k__hp_break_ties, "_hp_break_ties")):
                plan.epilogue = _EPI_OVR
                result = SVC_RESULT_ARGMAX
            else:
                plan.epilogue = _EPI_VOTES
                result = SVC_RESULT_CODES
        else:
            plan.epilogue = _EPI_PROBA if kind == SVC_QUERY_PROBA else _EPI_LOG_PROBA
            result = SVC_RESULT_F64_CLASSES
    if plan.epilogue >= 0:
        plan.classes = len(svc_field(fields, nk[].k_classes_, "classes_"))
    if ab is not None and len(ab) > 0 and n_rows > 0:
        var abuf = family_array(Python.tuple(PythonObject(2 * len(ab))), "<f8", context)
        var ap = FAMILY_F64P(unsafe_from_address=_addr(abuf))
        for i in range(len(ab)):  # driver-loop: over the class pairs (their sigmoid parameters)
            ap[2 * i] = svc_float(ab[i][0])
            ap[2 * i + 1] = svc_float(ab[i][1])
        plan.ab = _keep(owners, abuf)
    _ = mirrors
    return SvcQueryUnpacked(q^, owners, plan^, drivers, status^, result, binary)


def svc_query_refused(unpacked: SvcQueryUnpacked) raises:
    """After a RUN the device entry stopped: the refusal of the flagged
    query in the one wording 'Input X contains NaN or infinity (SVC).' (a
    fold: an index out of range or a NaN / infinity)."""
    _svm_query_refuse(Int(unpacked.status[0]), Int(unpacked.status[1]), unpacked.q, "SVC")


# ---------------------------------------------------------------- coef_


struct SvcCoefUnpacked(Movable):
    """`coef_`: one GEMV row per fitted machine, the captured buffers'
    pins, the feature count and the sign convention of the pair layout."""
    var rows: List[SvcCoefRow]
    var pins: List[ArrayPin]
    var features: Int
    var sign: Int

    def __init__(out self, var rows: List[SvcCoefRow], var pins: List[ArrayPin], features: Int, sign: Int):
        self.rows = rows^
        self.pins = pins^
        self.features = features
        self.sign = sign


def _coef_snapshot(value: PythonObject, name: String, context: PythonObject, mut pin: ArrayPin) raises -> ArrayMetadata:
    var nk = svm_keys()
    if py_type(value) is not py_item(context, nk[].k_array_class) and py_type(value) is not py_get(context, nk[].k_ndarray):
        raise Error("TypeError: " + name + " requires an admitted native array")
    classical_admit_value(value, context)
    var shape = array_shape_once(value, context)
    var captured = array_metadata_once(value, shape^, name, context, pin)
    if len(captured.shape) != 2 or not captured.f32_c() or not captured.native_byte_order:
        raise Error("TypeError: " + name + " requires a contiguous native float32 matrix")
    return captured^


def svc_coef_unpack(fields: PythonObject, classifier: Bool, context: PythonObject) raises -> SvcCoefUnpacked:
    """Public coef_ and composed RFE: the fitted buffers of every machine,
    each captured exactly once before RUN, with the checks in their order."""
    var b = native_loaded_module("builtins")
    var name = String("SVC" if classifier else "SVR")
    var nk = svm_keys()
    if not svc_has(fields, nk[].k_dual_coef_):
        raise Error("AttributeError: mojolearn " + name + ": call fit() first")
    if py_type(py_item(fields, nk[].k__hp_kernel)) is not b.str:
        raise Error("TypeError: fitted SVM kernel requires an exact native string")
    if String(py_item(fields, nk[].k__hp_kernel)) != "linear":
        raise Error("AttributeError: mojolearn " + name + ": coef_ is only available for kernel='linear'")
    host_inference_scalar(py_item(fields, nk[].k_n_features_in_), context)
    var d = Int(py=py_item(fields, nk[].k_n_features_in_))
    if d < 0:
        raise Error("ValueError: SVM feature count cannot be negative")
    var pairs = py_get(fields, nk[].k__pairs) if classifier else PythonObject(None)
    if pairs is not None and py_type(pairs) is not b.list and py_type(pairs) is not b.tuple:
        raise Error("TypeError: SVM pair state requires an exact list or tuple")
    var count = len(pairs) if pairs is not None else 1
    var rows = List[SvcCoefRow]()
    var pins = List[ArrayPin]()  # the door's pins of every captured buffer, held through the GEMVs
    for row in range(count):  # driver-loop: capture each fitted buffer exactly once before execution
        var pair = pairs[row] if pairs is not None else fields
        if py_type(pair) is not b.dict:
            raise Error("TypeError: SVM pair state requires an exact dictionary")
        var keys = py_call(b.list, pair)
        for i in range(len(keys)):  # driver-loop: reject key equality/hash hooks before field lookups
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: SVM fitted state requires exact string keys")
        var dual_pin = ArrayPin()
        var sv_pin = ArrayPin()
        var dual = _coef_snapshot(pair["dual" if pairs is not None else "dual_coef_"], "dual_coef_", context, dual_pin)
        var sv = _coef_snapshot(pair["sv" if pairs is not None else "support_vectors_"], "support_vectors_", context, sv_pin)
        if dual.shape[0] != 1 or dual.shape[1] != sv.shape[0] or sv.shape[1] != d:
            raise Error("ValueError: SVM coefficient and support-vector shapes disagree")
        rows.append(SvcCoefRow(dual.source.addr, sv.source.addr, sv.shape[0]))
        pins.append(dual_pin^)
        pins.append(sv_pin^)
    return SvcCoefUnpacked(rows^, pins^, d, 1 if pairs is not None else 0)
