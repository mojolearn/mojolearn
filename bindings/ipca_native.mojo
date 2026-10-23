# SPDX-License-Identifier: Apache-2.0
# The native IncrementalPCA owner: every public IncrementalPCA operation
# (fit, partial_fit, fit_transform, transform, inverse_transform) is one call
# into ipca_c. It converts X once, selects the numeric-mode decomp binding
# (`_Kit(numeric_mode_)`), runs one compiled driver (x_decomp/ipca_owner.mojo:
# the `_partial` / `_gram_svd` / transform steps the Python shell sequenced
# through the kit, the same entries, operands and order, so the same bits)
# and stores the fitted attributes under the names, types and shapes the
# shell published. fit_transform projects the X the fit converted.
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_typed, classical_host_f32
from bindings.decomp_inputs import decomp_input

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


def _admitted(model: PythonObject, context: PythonObject) raises -> Bool:
    # IncrementalPCA itself or a subclass: its MRO holds the connected class.
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        return False
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        if mro[k] is context["ipca_class"]:
            return True
    return False


def _int(value: PythonObject) raises -> Int:
    return Int(py=py_call(Python.import_module("builtins").int, value))


def _attr(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` for an instance field, with Python's AttributeError."""
    if not py_contains(fields, name):
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    return fields[name]


def _entries(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected _mojolearn_x_decomp IncrementalPCA drivers of the
    fitted numeric mode (`numeric_mode_`), as `_Kit(mode)` resolved it."""
    var b = Python.import_module("builtins")
    var mode = _attr(model, fields, "numeric_mode_")
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    var name = py_call(py_call(mode.strip).lower)
    if String(name) != "identical" and String(name) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["ipca_decomp"][name]
    if py_type(entries) is b.str:
        raise Error(String(entries))
    if py_get(entries, "x_decomp_ipca_fit") is None or py_get(entries, "x_decomp_ipca_partial") is None \
            or py_get(entries, "x_decomp_ipca_project") is None:
        raise Error("NotImplementedError: mojolearn IncrementalPCA: this build of _mojolearn_x_decomp does not export "
                    + "the x_decomp_ipca_* drivers; rebuild it with bindings/build_x_decomp.sh")
    return entries


def _set_mode(fields: PythonObject, context: PythonObject) raises:
    # `_mode(self.numeric_mode)`: the process mode when None, else the value as given.
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    py_setitem(fields, "numeric_mode_", mode)


def _f32(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    # A zero-filled C-order float32 mojolearn Array (`_M.out`).
    return classical_typed(shape, "<f4", context["output_context"])


def _requested(fields: PythonObject) raises -> Int:
    # n_components as `int(...)`, -1 for None.
    var value = fields["n_components"]
    if value is None:
        return -1
    return _int(value)


def _publish(fields: PythonObject, room: List[List[Float32]], count: Int, d: Int, noise: Float64,
             context: PythonObject) raises:
    """`_publish`: the public arrays at the kept count, from the driver's
    room (components, singular, mean, var, explained variance, ratio)."""
    var components = _f32(Python.tuple(count, d), context)
    var singular = _f32(Python.tuple(count), context)
    var mean = _f32(Python.tuple(d), context)
    var var_ = _f32(Python.tuple(d), context)
    var variance = _f32(Python.tuple(count), context)
    var ratio = _f32(Python.tuple(count), context)
    var outs = [components, singular, mean, var_, variance, ratio]
    var sizes = [count * d, count, d, d, count, count]
    for k in range(6):  # driver-loop: over the six published arrays
        if sizes[k]:
            memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(outs[k]))),
                   src=F32P(unsafe_from_address=Int(room[k].unsafe_ptr())), count=sizes[k])
    py_setitem(fields, "components_", components)
    py_setitem(fields, "singular_values_", singular)
    py_setitem(fields, "mean_", mean)
    py_setitem(fields, "var_", var_)
    py_setitem(fields, "explained_variance_", variance)
    py_setitem(fields, "explained_variance_ratio_", ratio)
    py_setitem(fields, "noise_variance_", PythonObject(noise))


def _room(d: Int) -> List[List[Float32]]:
    # The driver's outputs at the widest kept count (d components).
    var room = List[List[Float32]]()
    room.append(List[Float32](length=max(d * d, 1), fill=Float32(0)))
    for _ in range(5):  # driver-loop: over the five vector outputs
        room.append(List[Float32](length=max(d, 1), fill=Float32(0)))
    return room^


def _room_addresses(room: List[List[Float32]], params: PythonObject) raises:
    for k in range(6):  # driver-loop: over the six driver outputs
        _ = py_call(params.append, PythonObject(Int(room[k].unsafe_ptr())))


def ipca_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`IncrementalPCA.fit`; returns the converted X (kept for fit_transform)."""
    var fields = py_fields(model)
    _set_mode(fields, context)
    var names = Python.list("components_", "n_samples_seen_")
    for k in range(2):  # driver-loop: drops two fitted attribute names
        if py_contains(fields, names[k]):
            _ = py_call(fields.pop, names[k])
    var x = decomp_input(source, "IncrementalPCA", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var batch = fields["batch_size"]
    var batch_size = 5 * d if batch is None else _int(batch)
    py_setitem(fields, "batch_size_", PythonObject(batch_size))
    var requested = _requested(fields)
    var entries = _entries(model, fields, context)
    var room = _room(d)
    var params = Python.list(classical_address(x), n, d, requested, batch_size)
    _room_addresses(room, params)
    var got = py_call_typed(entries["x_decomp_ipca_fit"], params)
    var count = Int(py=got[0])
    py_setitem(fields, "n_samples_seen_", PythonObject(n))
    py_setitem(fields, "n_components_", PythonObject(count))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _publish(fields, room, count, d, Float64(py=got[1]), context)
    _ = room^
    return x


def ipca_partial_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`IncrementalPCA.partial_fit`: one more batch on the published state."""
    var fields = py_fields(model)
    if not py_contains(fields, "numeric_mode_"):
        _set_mode(fields, context)
    var x = decomp_input(source, "IncrementalPCA", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var seen_value = py_get(fields, "n_samples_seen_")
    var seen = 0 if seen_value is None else _int(seen_value)
    var entries = _entries(model, fields, context)
    var room = _room(d)
    var params = Python.list(classical_address(x), n, d, _requested(fields), seen)
    # the running state (`components_m_`, ...) is the published arrays' words
    var state = Python.list()
    var prev_count = 0
    var prev_d = d
    if seen > 0:
        var components = classical_host_f32(_attr(model, fields, "components_"), 2, "components_", context)
        prev_count = Int(py=components.shape[0])
        prev_d = Int(py=components.shape[1])
        _ = py_call(state.append, components)
        _ = py_call(state.append, classical_host_f32(_attr(model, fields, "singular_values_"), 1, "singular_values_", context))
        _ = py_call(state.append, classical_host_f32(_attr(model, fields, "mean_"), 1, "mean_", context))
        _ = py_call(state.append, classical_host_f32(_attr(model, fields, "var_"), 1, "var_", context))
    _ = py_call(params.append, prev_count)
    _ = py_call(params.append, prev_d)
    for k in range(4):  # driver-loop: over the four state arrays (unread for the first batch)
        _ = py_call(params.append, classical_address(state[k]) if seen > 0 else PythonObject(Int(room[0].unsafe_ptr())))
    _room_addresses(room, params)
    var got = py_call_typed(entries["x_decomp_ipca_partial"], params)
    _ = state^
    var count = Int(py=got[0])
    py_setitem(fields, "n_samples_seen_", PythonObject(seen + n))
    py_setitem(fields, "n_components_", PythonObject(count))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _publish(fields, room, count, d, Float64(py=got[1]), context)
    _ = room^
    _ = x^
    return model


def _check(model: PythonObject, fields: PythonObject) raises:
    if not py_contains(fields, "components_"):
        raise Error("RuntimeError: " + String(py_type(model).__name__) + " is not fitted; call fit first")


def _project(model: PythonObject, fields: PythonObject, x: PythonObject, inverse: Bool,
             context: PythonObject) raises -> PythonObject:
    """transform (X n x d) or inverse_transform (Y n x n_components) on the
    published components, mean and explained variance."""
    var entries = _entries(model, fields, context)
    var n = Int(py=x.shape[0])
    var width = Int(py=x.shape[1])
    var components = classical_host_f32(fields["components_"], 2, "components_", context)
    var nc = Int(py=components.shape[0])
    var d = Int(py=components.shape[1])
    if not inverse:
        var fitted = _int(_attr(model, fields, "n_features_in_"))
        if width != fitted:
            raise Error("X has " + String(width) + " features, but IncrementalPCA is expecting " + String(fitted))
    elif width != nc:
        raise Error("x_decomp: gemm inner dimensions " + String(width) + " and " + String(nc) + " differ")
    var mean = classical_host_f32(_attr(model, fields, "mean_"), 1, "mean_", context)
    var variance = classical_host_f32(_attr(model, fields, "explained_variance_"), 1, "explained_variance_", context)
    var out = _f32(Python.tuple(n, d if inverse else nc), context)
    var whiten = Bool(py=_attr(model, fields, "whiten"))
    _ = py_call_typed(entries["x_decomp_ipca_project"], Python.list(
        1 if inverse else 0, classical_address(x), n, d, nc, classical_address(mean), classical_address(components),
        classical_address(variance), 1 if whiten else 0, classical_address(out)))
    _ = components^
    _ = mean^
    _ = variance^
    return out


def ipca_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: ipca_native_call requires model, operation, arguments and connected context")
        var model = call[0]
        var b = Python.import_module("builtins")
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
        var result = PythonObject(None)
        if operation == "fit":
            _ = ipca_fit(model, args[0], context)
            result = model
        elif operation == "fit_transform":
            # the transform projects the X the fit converted: one conversion
            var x = ipca_fit(model, args[0], context)
            result = _project(model, py_fields(model), x, False, context)
            _ = x^
        elif operation == "partial_fit":
            result = ipca_partial_fit(model, args[0], context)
        elif operation == "transform" or operation == "inverse_transform":
            var fields = py_fields(model)
            _check(model, fields)
            var x = decomp_input(args[0], "IncrementalPCA", context)
            result = _project(model, fields, x, operation == "inverse_transform", context)
            _ = x^
        else:
            raise Error("unknown private IncrementalPCA operation")
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
