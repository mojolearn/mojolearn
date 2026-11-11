# SPDX-License-Identifier: Apache-2.0
# OneClassSVM `fit`, `score_samples`, `decision_function`, `predict` and
# `fit_predict` (python/mojolearn/_expansion_neighbors.py), one native call
# each. The owner runs the x_neighbors primitives the Python `_XNeighbors`
# methods ran (`xn_pos_compact`, `xn_take_rows`, `xn_take_cols`,
# `xn_variance`, `xn_kernel`, `xn_ocsvm`, `x_neighbors_ocsvm_alpha_init`, the
# resident solve `x_neighbors_ocsvm_resident[_idn]`, `xn_kernel_matmul`,
# `xn_matmul`, `xn_unary`, `xn_p2m_sign_label`) with the same buffers and the
# same integer/float words and launch order through native descriptors.
# Private scratch is Mojo-owned; public arrays remain at the boundary.
# Numerical implementations are unchanged; execution is owed to integration.
from providers.python.x_neighbors_descriptor_pyapi import xn_admit, xn_module, xn_is_module, xn_capabilities
from providers.python.estimator_binding_pyapi import estimator_binding
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.memory import memcpy
from bindings.ocsvm_provider import OcConnection
from bindings.labels_descriptor_boundary import labels_gather
from core.ocsvm_plan import (OcMatrix, OcWeights, OcFitState, oc_f32_address, oc_i32_address,
    oc_take_rows, oc_take_rows_into, oc_take_cols, oc_kernel, oc_variance, OcQueryState,
    oc_kernel_product, oc_fused_labels, oc_product, oc_decision, oc_sign)
from bindings.ocsvm_output_boundary import OcOutput
from bindings.ocsvm_scalar_boundary import oc_dictionary, oc_name, oc_kernel_admit, oc_parameter_repr, oc_mode_value
from bindings.admitted_family import admit_family
from bindings.native_loaded_module import _loaded_fields
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.py_balanced import py_type, py_get, py_setitem, py_fields
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    family_call, family_array, family_min,
)
from core.admitted_convert import admitted_convert_vector
from bindings.gp_svm_records import ocsvm_coef_reused, ocsvm_coef_remember
from bindings.gp_svm_gp_state import gp_x_known
from bindings.svc_common import (
    svc_builtins, svc_field, svc_float, svc_int, svc_truthy, svc_prefix, svc_scalar_array,
)

def _a(value: PythonObject) raises -> PythonObject:
    return classical_address(value)


def _f32_scalar(value: Float64) -> Float64:
    """A Python double rounded once to float32."""
    return Float64(Float32(value))


def _fast_tier_native(mut connection: OcConnection, binding: PythonObject, context: PythonObject) raises -> Bool:
    if not xn_is_module(binding):
        return False
    var module = xn_module(binding)
    var full_value = py_get(_loaded_fields(module), "__name__")
    if py_type(full_value) is not svc_builtins().str:
        return False
    var parts = String(full_value).split(".")
    var name = String(parts[len(parts) - 1])
    var prefix = String("mojolearn") if name == "_mojolearn" else String(name.removeprefix("_mojolearn_"))
    var getter = neural_descriptor_entry(module, prefix + "_numeric_mode", False)
    if getter is not None:
        var mode = oc_mode_value(context, connection.resident_driver().metadata(0))
        return mode is not None and oc_name(mode, "fast", context)
    if len(parts) < 2:
        return False
    return String(parts[len(parts) - 2]) == "fast"


def _resolve_gamma_native(mut connection: OcConnection, gamma: PythonObject,
    address: Int, rows: Int, columns: Int, context: PythonObject) raises -> Float64:
    if gamma is None or oc_name(gamma, "auto", context):
        return 1.0 / Float64(columns)
    if oc_name(gamma, "scale", context):
        var driver = connection.driver()
        var value = Float64(0)
        with GILReleased(Python()):
            value = oc_variance(driver, address, rows * columns)
        return 1.0 / (Float64(columns) * value) if value != 0 else 1.0
    return svc_float(gamma)


def _ocsvm_fit_native(model: PythonObject, binding: PythonObject, source: PythonObject,
    sample_weight: PythonObject, context: PythonObject, mut connection: OcConnection) raises:
    var b = svc_builtins()
    var fields = py_fields(model)
    var kernel = svc_field(fields, "kernel")
    if Bool(py=family_call(b.callable, kernel)):
        raise Error("NotImplementedError: OneClassSVM callable kernels require a native kernel name")
    if not oc_kernel_admit(kernel, context):
        raise Error("NotImplementedError: OneClassSVM: kernel=" + oc_parameter_repr(kernel, context) + " is not implemented")
    var nu = svc_float(svc_field(fields, "nu"))
    if not (0.0 < nu and nu <= 1.0):
        raise Error("nu must be in (0, 1]")
    var x = source if gp_x_known(source, context) else admit_family(source, 2, "X", context, True).array
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var x_address = Int(py=_a(x))
    var precomputed = oc_name(kernel, "precomputed", context)
    if precomputed and n != d:
        raise Error("Precomputed matrix must be a square matrix.")
    var weights = OcWeights()
    if sample_weight is None:
        weights.unit(n)
    else:
        var wv = admit_family(sample_weight, 1, "sample_weight", context, False).array
        if Int(py=wv.shape[0]) != n:
            raise Error("sample_weight and X have different numbers of samples")
        if n > 0 and family_min(wv) < 0:
            raise Error("negative sample_weight is not supported")
        var weight_address = Int(py=_a(wv))
        var driver = connection.driver()
        with GILReleased(Python()):
            weights.compact(driver, weight_address, n)
        _ = wv
    var m = weights.count
    var q = OcMatrix(0, 0)
    var xw = OcMatrix(n, d, x_address)
    var has_gram = False
    var resident_available = False
    var resident_fast = False
    var degree = 0
    var coef0 = Float64(0)
    if precomputed:
        py_setitem(fields, PythonObject("_gamma"), PythonObject(0.0))
        if not weights.selected:
            q = OcMatrix(n, d, x_address)
        else:
            var driver = connection.driver()
            with GILReleased(Python()):
                var selected_rows = oc_take_rows(driver, x_address, n, d, oc_i32_address(weights.rows), m)
                q = oc_take_cols(driver, selected_rows.address(), m, d, oc_i32_address(weights.rows), m)
                _ = selected_rows
        has_gram = True
    else:
        degree = svc_int(svc_field(fields, "degree"))
        coef0 = svc_float(svc_field(fields, "coef0"))
        var gamma = _f32_scalar(_resolve_gamma_native(connection, svc_field(fields, "gamma"), x_address, n, d, context))
        py_setitem(fields, PythonObject("_gamma"), PythonObject(gamma))
        if weights.selected:
            var driver = connection.driver()
            with GILReleased(Python()):
                xw = oc_take_rows(driver, x_address, n, d, oc_i32_address(weights.rows), m)
        if xn_is_module(binding):
            resident_fast = _fast_tier_native(connection, binding, context)
            var entry = "x_neighbors_ocsvm_resident" if resident_fast else "x_neighbors_ocsvm_resident_idn"
            resident_available = neural_descriptor_entry(xn_module(binding), entry, False) is not None
        else:
            resident_available = (xn_capabilities(binding) & 1) != 0
        if not resident_available:
            var code = Int(py=context["family_svc_xn_kernels"][kernel])
            var driver = connection.driver()
            with GILReleased(Python()):
                q = oc_kernel(driver, xw.address(), xw.address(), m, m, d, code, degree, gamma, coef0)
            has_gram = True
    var state = OcFitState(m)
    var driver = connection.driver()
    with GILReleased(Python()):
        state.initialize(driver, oc_f32_address(weights.values), nu)
    state.prepare_metadata()
    var max_iter = svc_int(svc_field(fields, "max_iter"))
    var cap = 10000000 if max_iter < 0 else max_iter
    var tol = _f32_scalar(svc_float(svc_field(fields, "tol")))
    if not has_gram:
        var code = Int(py=context["family_svc_xn_kernels"][kernel])
        var gamma = svc_float(svc_field(fields, "_gamma"))
        if xn_is_module(binding):
            var resident = connection.resident_driver()
            with GILReleased(Python()):
                state.solve_resident(resident, xw.address(), oc_f32_address(weights.values),
                    d, code, degree, cap, resident_fast, gamma, coef0, tol)
        else:
            with GILReleased(Python()):
                state.solve_descriptor(driver, xw.address(), oc_f32_address(weights.values),
                    d, code, degree, cap, gamma, coef0, tol)
    else:
        with GILReleased(Python()):
            state.solve_gram(driver, q.address(), oc_f32_address(weights.values), cap, tol)
    var rho = Float64(state.info[0])
    with GILReleased(Python()):
        state.compact_support(driver)
    var n_sv = state.supports
    var support = PythonObject(None)
    if weights.selected and n_sv > 0:
        # The former first prefix Array was private. Preserve its copy in a
        # native owner before allocating the final mapped public support.
        var prefix = List[Int32](length=n_sv, fill=0)
        memcpy(dest=prefix.unsafe_ptr(), src=state.support_rows.unsafe_ptr(), count=n_sv)
        support = family_array(Python.tuple(PythonObject(n_sv)), "<i4", context)
        labels_gather(context, 0, oc_i32_address(weights.rows), m, oc_i32_address(prefix), n_sv, Int(py=_a(support)))
        _ = prefix
    else:
        support = svc_prefix(oc_i32_address(state.support_rows), n_sv, Python.tuple(PythonObject(n_sv)), "<i4", context)
    py_setitem(fields, PythonObject("support_"), support)
    if precomputed:
        py_setitem(fields, PythonObject("support_vectors_"),
            family_array(Python.tuple(PythonObject(n_sv), PythonObject(0)), "<f4", context))
    else:
        var vectors = family_array(Python.tuple(PythonObject(n_sv), PythonObject(d)), "<f4", context)
        var support_address = Int(py=_a(support))
        var vectors_address = Int(py=_a(vectors))
        with GILReleased(Python()):
            oc_take_rows_into(driver, x_address, n, d, support_address, n_sv, vectors_address)
        _ = support
        _ = vectors
        py_setitem(fields, PythonObject("support_vectors_"), vectors)
    py_setitem(fields, PythonObject("dual_coef_"),
        svc_prefix(oc_f32_address(state.support_values), n_sv,
            Python.tuple(PythonObject(1), PythonObject(n_sv)), "<f4", context))
    py_setitem(fields, PythonObject("intercept_"), svc_scalar_array(-rho, "<f4", context))
    py_setitem(fields, PythonObject("offset_"), PythonObject(rho))
    py_setitem(fields, PythonObject("n_iter_"), PythonObject(Int(state.iterations[0])))
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(d))
    py_setitem(fields, PythonObject("fit_status_"), PythonObject(0))
    _ = x
    _ = xw
    _ = q
    _ = weights
    _ = state
    _ = connection


@fieldwise_init
struct OcIndexPin(Movable):
    var owner: PythonObject
    var words: List[Int32]
    var address: Int
    var count: Int


def _oc_indices(value: PythonObject) raises -> OcIndexPin:
    var typestr = String(value.dtype)
    var shape = value.shape
    if typestr == "<i4":
        # Original exact-int32 storage bypassed conversion and used shape[0].
        var count = Int(py=shape[0])
        return OcIndexPin(value, List[Int32](), Int(py=_a(value)), count)
    if len(shape) == 0:
        var kind = String("float") if typestr.startswith("<f") else String("int")
        raise Error("TypeError: '" + kind + "' object is not iterable")
    if len(shape) != 1:
        raise Error("TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'")
    var count = Int(py=shape[0])
    var words = List[Int32](length=count, fill=0)
    var address = oc_i32_address(words)
    _ = admitted_convert_vector(Int(py=_a(value)), typestr, address, "<i4", count)
    _ = value
    return OcIndexPin(value, words^, address, count)


def _ocsvm_labels_available(binding: PythonObject) raises -> Bool:
    if xn_is_module(binding):
        return neural_descriptor_entry(xn_module(binding), "x_neighbors_ocsvm_labels", False) is not None
    return (xn_capabilities(binding) & 16) != 0


def _ocsvm_run(model: PythonObject, binding: PythonObject, source: PythonObject,
    labels: Bool, public_output: Bool, context: PythonObject, mut connection: OcConnection) raises -> OcOutput:
    var fields = py_fields(model)
    var q = source if gp_x_known(source, context) else admit_family(source, 2, "X", context, True).array
    var dual = svc_field(fields, "dual_coef_")
    var n_sv = Int(py=dual.shape[1])
    var coef: PythonObject
    if ocsvm_coef_reused(model, dual):
        coef = dual
    else:
        coef = admit_family(dual, 2, "dual_coef_", context, False).array
        ocsvm_coef_remember(model, dual, coef)
    var kernel = svc_field(fields, "kernel")
    var precomputed = oc_name(kernel, "precomputed", context)
    var nq = Int(py=q.shape[0])
    var sv = svc_field(fields, "support_vectors_")
    if not precomputed and n_sv > 0 and nq > 0 and not svc_truthy(context["family_svc_xn_old_items"]):
        var degree = svc_int(svc_field(fields, "degree"))
        var gamma = svc_float(svc_field(fields, "_gamma"))
        var coef0 = svc_float(svc_field(fields, "coef0"))
        var d = Int(py=q.shape[1])
        if labels and _ocsvm_labels_available(binding):
            var offset = svc_float(svc_field(fields, "offset_"))
            var result = OcOutput(nq, True, False, context)
            var query_address = Int(py=_a(q))
            var reference_address = Int(py=_a(sv))
            var coefficient_address = Int(py=_a(coef))
            var code = Int(py=context["family_svc_xn_kernels"][kernel])
            var driver = connection.driver()
            with GILReleased(Python()):
                oc_fused_labels(driver, query_address, reference_address, coefficient_address,
                    result.state.address(), nq, n_sv, d, code, degree, gamma, coef0, offset)
            _ = q
            _ = sv
            _ = coef
            return result^
        var result = OcOutput(nq, False, public_output, context)
        var query_address = Int(py=_a(q))
        var reference_address = Int(py=_a(sv))
        var coefficient_address = Int(py=_a(coef))
        var code = Int(py=context["family_svc_xn_kernels"][kernel])
        var driver = connection.driver()
        with GILReleased(Python()):
            oc_kernel_product(driver, query_address, reference_address, coefficient_address,
                result.state.address(), nq, n_sv, d, code, degree, gamma, coef0)
        _ = q
        _ = sv
        _ = coef
        return result^
    var gram = OcMatrix(0, 0)
    if precomputed:
        var support = _oc_indices(svc_field(fields, "support_"))
        var query_address = Int(py=_a(q))
        var columns = Int(py=q.shape[1])
        var driver = connection.driver()
        with GILReleased(Python()):
            gram = oc_take_cols(driver, query_address, nq, columns, support.address, support.count)
        _ = support
    else:
        var code = Int(py=context["family_svc_xn_kernels"][kernel])
        var gamma = svc_float(svc_field(fields, "_gamma"))
        var coef0 = svc_float(svc_field(fields, "coef0"))
        var degree = svc_int(svc_field(fields, "degree"))
        var features = Int(py=q.shape[1])
        var references = Int(py=sv.shape[0])
        var query_address = Int(py=_a(q))
        var reference_address = Int(py=_a(sv))
        var driver = connection.driver()
        with GILReleased(Python()):
            gram = oc_kernel(driver, query_address, reference_address, nq, references, features,
                code, degree, gamma, coef0)
    var result = OcOutput(nq, False, public_output, context)
    var coefficient_address = Int(py=_a(coef))
    var driver = connection.driver()
    with GILReleased(Python()):
        oc_product(driver, gram.address(), coefficient_address, result.state.address(), gram.rows, gram.columns)
    _ = q
    _ = sv
    _ = gram
    _ = coef
    return result^


def _ocsvm_scores(model: PythonObject, binding: PythonObject, source: PythonObject,
    context: PythonObject, mut connection: OcConnection) raises -> PythonObject:
    var result = _ocsvm_run(model, binding, source, False, True, context, connection)
    var public_value = result.public_value
    _ = result
    return public_value


def _ocsvm_decision(model: PythonObject, binding: PythonObject, source: PythonObject,
    context: PythonObject, mut connection: OcConnection) raises -> PythonObject:
    var scores = _ocsvm_run(model, binding, source, False, False, context, connection)
    var result = family_array(Python.tuple(PythonObject(scores.state.count)), "<f4", context)
    var offset = svc_float(svc_field(py_fields(model), "offset_"))
    var address = Int(py=_a(result))
    var driver = connection.driver()
    with GILReleased(Python()):
        oc_decision(driver, scores.state.address(), address, scores.state.count, offset)
    _ = scores
    _ = result
    return result


def _ocsvm_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
    context: PythonObject, mut connection: OcConnection) raises -> PythonObject:
    var first = _ocsvm_run(model, binding, source, True, False, context, connection)
    var n = first.state.count
    var decision = OcQueryState(0, False, False)
    var labels = OcQueryState(0, True, False)
    var label_address = first.state.address()
    if not first.state.labels:
        decision = OcQueryState(n, False, True)
        var offset = svc_float(svc_field(py_fields(model), "offset_"))
        var driver = connection.driver()
        with GILReleased(Python()):
            oc_decision(driver, first.state.address(), decision.address(), n, offset)
        labels = OcQueryState(n, True, True)
        if n > 0:
            with GILReleased(Python()):
                oc_sign(driver, decision.address(), labels.address(), n)
        label_address = labels.address()
    var result = family_array(Python.tuple(PythonObject(n)), "<i8", context)
    if n > 0:
        _ = admitted_convert_vector(label_address, "<i4", Int(py=_a(result)), "<i8", n)
    _ = first
    _ = decision
    _ = labels
    _ = result
    return result


def _oc_binding(source: PythonObject, context: PythonObject) raises -> PythonObject:
    oc_dictionary(context)
    var backend = context["backend_fields"]
    oc_dictionary(backend)
    var binding = source
    if py_type(source) is backend["_HostBinding"] or py_type(source) is backend["_NoGpuBinding"]:
        binding = estimator_binding(PythonObject(None), "_mojolearn_x_neighbors", context)
    return xn_admit(binding)


def svc_ocsvm_operation(model: PythonObject, operation: String, args: PythonObject,
    context: PythonObject) raises -> PythonObject:
    var binding = _oc_binding(args[0], context)
    classical_connect_numpy(context)
    var connection = OcConnection(binding)
    try:
        if operation == "family_svc_ocsvm_fit":
            _ocsvm_fit_native(model, binding, args[1], args[2], context, connection)
            return model
        if operation == "family_svc_ocsvm_score_samples":
            return _ocsvm_scores(model, binding, args[1], context, connection)
        if operation == "family_svc_ocsvm_decision":
            return _ocsvm_decision(model, binding, args[1], context, connection)
        if operation == "family_svc_ocsvm_predict":
            return _ocsvm_predict(model, binding, args[1], context, connection)
        if operation == "family_svc_ocsvm_fit_predict":
            _ocsvm_fit_native(model, binding, args[1], PythonObject(None), context, connection)
            return _ocsvm_predict(model, binding, args[1], context, connection)
        raise Error("RuntimeError: unknown OneClassSVM operation " + operation)
    finally:
        _ = connection
        _ = args
        _ = binding
