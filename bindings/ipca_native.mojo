# SPDX-License-Identifier: Apache-2.0
"""IncrementalPCA admission/publication around typed native requests and owners."""
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_typed_native
from bindings.admitted_input import AdmittedX, admit_f32
from bindings.decomp_inputs import decomp_admit
from bindings.decomp_scalar_admission import decomp_scalar_admit
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.decomp_public_lifecycle import decomp_public_method, decomp_public_begin, decomp_public_finish
from bindings.public_lifecycle_owner import lifecycle_end
from core.ipca_native_api import IPCAFitRequest, IPCAPartialRequest, IPCAPrevious, IPCAQueryRequest, IPCAFitResult, IPCAStateIds
from core.ipca_native_state import IPCAOutputRoom, ipca_output_shape
from bindings.ipca_descriptor_boundary import ipca_connection, ipca_resident, ipca_run_fit, ipca_run_partial, ipca_run_query, ipca_run_free
from bindings.ipca_owner_boundary import ipca_owner_drop, ipca_owner_adopt, ipca_owner_cached, ipca_owner_operand, ipca_owner_handle

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


def _admitted(model: PythonObject, context: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        return False
    var target = context["ipca_class"]
    if py_type(target) is not b.type:
        return False
    return cpython_table()[].PyType_IsSubtype(cls._obj_ptr, target._obj_ptr) != 0


def _int(value: PythonObject, context: PythonObject) raises -> Int:
    decomp_scalar_admit(value, context)
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def _truth(value: PythonObject, context: PythonObject) raises -> Bool:
    # Truth uses length only for exact builtin containers; no element is
    # inspected. Keep those previously accepted values without a protocol.
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if (cls is b.list or cls is b.tuple or cls is b.dict or cls is b.set
            or cls is b.frozenset or cls is b.range or cls is b.complex):
        return Bool(py=value)
    try:
        host_inference_scalar(value, context, True)
    except failure:
        var ndarray = py_get(context, "ndarray")
        if ndarray is not None:
            var dtype = native_numpy_dtype(context)
            if cls is ndarray and Int(py=value.size) == 1:
                # Native text truth also accepts one item at any rank.
                for code in ["U", "S"]:  # driver-loop: native text dtype classes
                    if py_type(value.dtype) is py_type(py_call(dtype, code)):
                        return Bool(py=value)
            else:
                for code in ["F", "D", "G"]:  # driver-loop: exact native complex scalar types
                    if cls is py_call(dtype, code).type:
                        return Bool(py=value)
        decomp_scalar_admit(value, context)
    return Bool(py=value)


def _attr(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    if not py_contains(fields, name):
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    return fields[name]


def _set_mode(fields: PythonObject, context: PythonObject) raises:
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    py_setitem(fields, "numeric_mode_", mode)


def _f32(rows: Int, cols: Int, context: PythonObject, vector: Bool = False) raises -> PythonObject:
    return classical_typed_native(ipca_output_shape(rows, cols, vector), "<f4", context["output_context"])


def _requested(fields: PythonObject, context: PythonObject) raises -> Int:
    var value = fields["n_components"]
    return -1 if value is None else _int(value, context)


def _copy_output(dst: PythonObject, values: List[Float32], count: Int) raises:
    if count:
        memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(dst))),
               src=F32P(unsafe_from_address=Int(values.unsafe_ptr())), count=count)
    _ = values
    _ = dst


def _publish(fields: PythonObject, room: IPCAOutputRoom, count: Int, features: Int,
             noise: Float64, context: PythonObject) raises:
    # Keep all six allocations, prefix copies and public writes in their
    # original order. Only the shape/size envelopes are native now.
    var components = _f32(count, features, context)
    var singular = _f32(count, 0, context, True)
    var mean = _f32(features, 0, context, True)
    var variance = _f32(features, 0, context, True)
    var explained = _f32(count, 0, context, True)
    var ratio = _f32(count, 0, context, True)
    _copy_output(components, room.values[0], count * features)
    _copy_output(singular, room.values[1], count)
    _copy_output(mean, room.values[2], features)
    _copy_output(variance, room.values[3], features)
    _copy_output(explained, room.values[4], count)
    _copy_output(ratio, room.values[5], count)
    py_setitem(fields, "components_", components)
    py_setitem(fields, "singular_values_", singular)
    py_setitem(fields, "mean_", mean)
    py_setitem(fields, "var_", variance)
    py_setitem(fields, "explained_variance_", explained)
    py_setitem(fields, "explained_variance_ratio_", ratio)
    py_setitem(fields, "noise_variance_", PythonObject(noise))
    _ = room


def _state_name(index: Int) -> String:
    if index == 0:
        return "components_"
    if index == 1:
        return "singular_values_"
    if index == 2:
        return "mean_"
    if index == 3:
        return "var_"
    return "explained_variance_"


def _free_kept(connection: Int, mut result: IPCAFitResult) raises:
    if not result.kept:
        return
    for k in range(5):  # driver-loop: five returned resident handles
        var handle = result.state.values[k]
        result.state.values[k] = -1
        if handle >= 0:
            ipca_run_free(connection, handle)
    result.kept = False


def _adopt_state(model: PythonObject, fields: PythonObject, connection: Int,
                 mut result: IPCAFitResult, features: Int) raises:
    if not result.kept:
        return
    for k in range(5):  # driver-loop: five returned resident buffers
        var rows = result.count if k == 0 else 1
        var cols = features if (k == 0 or k == 2 or k == 3) else result.count
        var handle = result.state.values[k]
        result.state.values[k] = -1
        try:
            # The callee takes this handle even if packing the owner fails.
            ipca_owner_adopt(model, connection, fields[PythonObject(_state_name(k))], k, rows, cols, handle)
        except error:
            _free_kept(connection, result)
            raise error^
    result.kept = False


def _finish_fit(model: PythonObject, fields: PythonObject, connection: Int,
                room: IPCAOutputRoom, mut result: IPCAFitResult, seen: Int,
                features: Int, context: PythonObject) raises:
    try:
        py_setitem(fields, "n_samples_seen_", PythonObject(seen))
        py_setitem(fields, "n_components_", PythonObject(result.count))
        py_setitem(fields, "n_features_in_", PythonObject(features))
        _publish(fields, room, result.count, features, result.noise, context)
    except error:
        _free_kept(connection, result)
        raise error^
    _adopt_state(model, fields, connection, result, features)


def ipca_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> AdmittedX:
    var fields = py_fields(model)
    ipca_owner_drop(model)
    _set_mode(fields, context)
    if py_contains(fields, "components_"):
        _ = py_call(fields.pop, "components_")
    if py_contains(fields, "n_samples_seen_"):
        _ = py_call(fields.pop, "n_samples_seen_")
    var x = decomp_admit(source, "IncrementalPCA", context)
    var batch = fields["batch_size"]
    var batch_size = 5 * x.cols if batch is None else _int(batch, context)
    py_setitem(fields, "batch_size_", PythonObject(batch_size))
    var requested = _requested(fields, context)
    var connection = ipca_connection(model, fields, context)
    var room = IPCAOutputRoom(x.cols)
    var request = IPCAFitRequest(x.address, x.rows, x.cols, requested, batch_size,
        room.outputs(), ipca_resident(connection))
    try:
        var result = ipca_run_fit(connection, request)
        _finish_fit(model, fields, connection, room, result, x.rows, x.cols, context)
    finally:
        _ = x
        _ = room
    return x^


def ipca_partial_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "numeric_mode_"):
        _set_mode(fields, context)
    var x = decomp_admit(source, "IncrementalPCA", context)
    var seen_value = py_get(fields, "n_samples_seen_")
    var seen = 0 if seen_value is None else _int(seen_value, context)
    var connection = ipca_connection(model, fields, context)
    var room = IPCAOutputRoom(x.cols)
    var requested = _requested(fields, context)
    var state = List[AdmittedX]()
    var held = List[PythonObject]()
    var previous_count = 0
    var previous_features = x.cols
    var resident = ipca_resident(connection)
    var ids = IPCAStateIds()
    var dummy = room.outputs().components
    var addresses: InlineArray[Int, 4] = [dummy, dummy, dummy, dummy]
    if seen > 0:
        var components = admit_f32(_attr(model, fields, "components_"), 2, "components_", context)
        previous_count = components.rows
        previous_features = components.cols
        state.append(components^)
        state.append(admit_f32(_attr(model, fields, "singular_values_"), 1, "singular_values_", context))
        state.append(admit_f32(_attr(model, fields, "mean_"), 1, "mean_", context))
        state.append(admit_f32(_attr(model, fields, "var_"), 1, "var_", context))
        for k in range(4):  # driver-loop: four admitted prior-state buffers
            addresses[k] = state[k].address
            if resident and state[k].array is fields[PythonObject(_state_name(k))]:
                var owner = ipca_owner_cached(model, connection, state[k].array, k, False)
                if owner is not None:
                    ids.values[k] = ipca_owner_handle(owner)
                    held.append(owner)
    var request = IPCAPartialRequest(x.address, x.rows, x.cols, requested, seen,
        previous_count, previous_features, IPCAPrevious(addresses[0], addresses[1], addresses[2], addresses[3]),
        room.outputs(), resident, ids^, resident)
    try:
        var result = ipca_run_partial(connection, request)
        _finish_fit(model, fields, connection, room, result, seen + x.rows, x.cols, context)
    finally:
        # Older residents stay alive through adoption of every new owner.
        _ = held
        _ = state
        _ = room
        _ = x
    return model


def _check(model: PythonObject, fields: PythonObject) raises:
    if not py_contains(fields, "components_"):
        raise Error("RuntimeError: " + String(py_type(model).__name__) + " is not fitted; call fit first")


def _project(model: PythonObject, fields: PythonObject, x: AdmittedX, inverse: Bool,
             context: PythonObject) raises -> PythonObject:
    var connection = ipca_connection(model, fields, context)
    var components = admit_f32(fields["components_"], 2, "components_", context)
    var count = components.rows
    var features = components.cols
    if not inverse:
        var fitted = _int(_attr(model, fields, "n_features_in_"), context)
        if x.cols != fitted:
            raise Error("X has " + String(x.cols) + " features, but IncrementalPCA is expecting " + String(fitted))
    elif x.cols != count:
        raise Error("x_decomp: gemm inner dimensions " + String(x.cols) + " and " + String(count) + " differ")
    var mean = admit_f32(_attr(model, fields, "mean_"), 1, "mean_", context)
    var variance = admit_f32(_attr(model, fields, "explained_variance_"), 1, "explained_variance_", context)
    var dst = _f32(x.rows, features if inverse else count, context)
    var whiten = _truth(_attr(model, fields, "whiten"), context)
    var held = List[PythonObject]()
    var ids = IPCAStateIds()
    var resident = ipca_resident(connection)
    if resident:
        # Only these three stored arrays are read by a query; borrowed and
        # converted storage keeps the original per-call upload path.
        if components.array is fields["components_"]:
            var owner = ipca_owner_operand(model, connection, components.array, 0, count, features, components.address)
            if owner is not None:
                ids.values[0] = ipca_owner_handle(owner)
                held.append(owner)
        if mean.array is fields["mean_"]:
            var owner = ipca_owner_operand(model, connection, mean.array, 2, 1, features, mean.address)
            if owner is not None:
                ids.values[2] = ipca_owner_handle(owner)
                held.append(owner)
        if whiten and variance.array is fields["explained_variance_"]:
            var owner = ipca_owner_operand(model, connection, variance.array, 4, 1, count, variance.address)
            if owner is not None:
                ids.values[4] = ipca_owner_handle(owner)
                held.append(owner)
    var request = IPCAQueryRequest(x.address, x.rows, features, count, mean.address, components.address,
        variance.address, Int(py=classical_address(dst)), inverse, whiten, resident, ids^)
    try:
        ipca_run_query(connection, request)
    finally:
        _ = held
        _ = x
        _ = components
        _ = mean
        _ = variance
    return dst


def _ipca_dispatch(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = PythonObject(None)
    if operation == "fit":
        _ = ipca_fit(model, args[0], context)
        result = model
    elif operation == "fit_transform":
        var x = ipca_fit(model, args[0], context)
        result = _project(model, py_fields(model), x, False, context)
        _ = x
    elif operation == "partial_fit":
        result = ipca_partial_fit(model, args[0], context)
    elif operation == "transform" or operation == "inverse_transform":
        var fields = py_fields(model)
        _check(model, fields)
        var x = decomp_admit(args[0], "IncrementalPCA", context)
        result = _project(model, fields, x, operation == "inverse_transform", context)
        _ = x
    else:
        raise Error("unknown private IncrementalPCA operation")
    return result


def ipca_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: ipca_native_call requires model, operation, arguments and connected context")
        var model = call[0]
        var b = native_loaded_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple:
            raise Error("TypeError: native IncrementalPCA operation and argument tuple are required")
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        if py_type(context) is not b.dict:
            raise Error("TypeError: native IncrementalPCA requires connected context")
        if not _admitted(model, context):
            raise Error("TypeError: the native IncrementalPCA owner requires an IncrementalPCA instance")
        classical_connect_numpy(context)
        var method = decomp_public_method(operation)
        var depth = decomp_public_begin(model, method, args, True, True, context)
        var result = PythonObject(None)
        try:
            result = _ipca_dispatch(model, operation, args, context)
        finally:
            lifecycle_end(model, method, depth)
        result = decomp_public_finish(model, method, args, result, "", context)
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
