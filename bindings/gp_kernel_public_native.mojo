# SPDX-License-Identifier: Apache-2.0
# Whole-operation GP kernel metadata owner. Only import-connected, exact kernel
# classes and passive native state are admitted; no kernel method, descriptor,
# numeric protocol, or sequence protocol executes in this traversal.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.memory import bitcast
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_contains
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.native_instance_item import native_instance_item
from x_metrics.epilogue import portable_log_c


struct GPKernelSnapshot(Movable):
    var values: List[Float64]
    var lows: List[Float64]
    var highs: List[Float64]
    var owners: PythonObject
    var states: PythonObject

    def __init__(out self) raises:
        self.values = List[Float64]()
        self.lows = List[Float64]()
        self.highs = List[Float64]()
        self.owners = Python.list()
        self.states = Python.list()


def _kernel_scalar(value: PythonObject) raises -> Float64:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is not b.float and cls is not b.int and cls is not b.bool:
        raise Error("TypeError: GP kernel state requires native numeric scalars without Python conversion hooks")
    return Float64(py=value)


def _kernel_sequence(value: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is not b.list and cls is not b.tuple:
        raise Error("TypeError: GP kernel state requires exact lists or tuples without Python sequence hooks")


def _kernel_class(kernel: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(kernel)
    # Identity comparison precedes keyed lookup: an application metaclass may
    # otherwise execute __hash__ or __eq__ through a table membership test.
    var classes = py_call(native_loaded_module("builtins").list, context["family_gp_kernel_nodes"].keys())
    for i in range(len(classes)):  # driver-loop: fixed import-connected kernel types
        if cls is classes[i]:
            return cls
    raise Error("TypeError: native GP kernels require a connected built-in kernel without Python subclass protocols")


def _kernel_state(kernel: PythonObject, mut snapshot: GPKernelSnapshot) raises -> PythonObject:
    for i in range(len(snapshot.owners)):  # driver-loop: one state capture per distinct kernel owner
        if kernel is snapshot.owners[i]:
            return snapshot.states[i]
    var state = _fields(kernel)
    _ = py_call(snapshot.owners.append, kernel)
    _ = py_call(snapshot.states.append, state)
    return state


def _kernel_read(kernel: PythonObject, fields: PythonObject, name: PythonObject,
                 optional: Bool = False) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, py_type(kernel))
    for i in range(len(ancestry)):  # driver-loop: reject replaced field descriptors without invoking them
        if py_contains(_class_fields(ancestry[i]), name):
            raise Error("TypeError: native GP kernel state must use instance fields, not class fields or descriptors")
    if optional and not py_contains(fields, name):
        return PythonObject(None)
    return native_instance_item(fields, name)


def _kernel_walk(kernel: PythonObject, context: PythonObject, active: PythonObject,
                 mut snapshot: GPKernelSnapshot) raises:
    var cls = _kernel_class(kernel, context)
    for i in range(len(active)):  # driver-loop: reject cycles before descending into a composite
        if kernel is active[i]:
            raise Error("ValueError: cyclic GP kernel graph")
    if len(active) >= 512:
        raise Error("ValueError: GP kernel graph exceeds native traversal depth")
    _ = py_call(active.append, kernel)
    var fields = _kernel_state(kernel, snapshot)
    var layout = context["family_gp_kernel_nodes"][cls]
    var how = String(layout[0])
    if how == "pair":
        var left = _kernel_read(kernel, fields, layout[1])
        var right = _kernel_read(kernel, fields, layout[2])
        var op = _kernel_read(kernel, fields, layout[3])
        if py_type(op) is not py_type(PythonObject(0)) or (Int(py=op) != 4 and Int(py=op) != 5):
            raise Error("TypeError: native GP kernel operator must be the sum or product code")
        _kernel_walk(left, context, active, snapshot)
        _kernel_walk(right, context, active, snapshot)
    elif how == "nodes":
        var nodes = _kernel_read(kernel, fields, layout[1])
        _kernel_sequence(nodes)
        for i in range(len(nodes)):  # driver-loop: saved postfix nodes are fixed but must remain passive
            var node = nodes[i]
            _kernel_sequence(node)
            if len(node) != 3 or py_type(node[0]) is not py_type(PythonObject(0)):
                raise Error("TypeError: saved GP kernel nodes require an integer kind, parameter and scale list")
            _ = _kernel_scalar(node[1])
            _kernel_sequence(node[2])
            for j in range(len(node[2])):  # driver-loop: saved scale words
                _ = _kernel_scalar(node[2][j])
    elif how == "leaf":
        var values = List[Float64]()
        if layout[2] is not None:
            var value = _kernel_scalar(_kernel_read(kernel, fields, layout[2]))
            if layout[3] is None:
                values.append(value)
        if layout[3] is not None:
            var scales = _kernel_read(kernel, fields, layout[3])
            _kernel_sequence(scales)
            for i in range(len(scales)):  # driver-loop: one value per isotropic/ARD scale
                values.append(_kernel_scalar(scales[i]))
        var bounds = _kernel_read(kernel, fields, context["family_gp_kernel_bounds"][cls])
        if py_type(bounds) is py_type(PythonObject("")):
            if String(bounds) != "fixed":
                raise Error("ValueError: GP kernel bounds string must be 'fixed'")
        else:
            _kernel_sequence(bounds)
            if len(bounds) != len(values):
                raise Error("ValueError: GP kernel bounds must match the number of hyperparameters")
            for i in range(len(values)):  # driver-loop: copy each free value and its bound pair once
                var pair = bounds[i]
                _kernel_sequence(pair)
                if len(pair) != 2:
                    raise Error("ValueError: GP kernel bounds require pairs")
                snapshot.values.append(values[i])
                snapshot.lows.append(_kernel_scalar(pair[0]))
                snapshot.highs.append(_kernel_scalar(pair[1]))
    else:
        raise Error("RuntimeError: unknown connected GP kernel layout")
    _ = py_call(active.pop)


def gp_kernel_snapshot(kernel: PythonObject, context: PythonObject) raises -> GPKernelSnapshot:
    var snapshot = GPKernelSnapshot()
    _kernel_walk(kernel, context, Python.list(), snapshot)
    return snapshot^


def gp_admit_kernel(kernel: PythonObject, context: PythonObject) raises:
    # Estimator owners can share the same strict admission. They must keep the
    # GIL/owner references while consuming the admitted state without callbacks.
    var snapshot = gp_kernel_snapshot(kernel, context)
    _ = snapshot^


def _kernel_log(value: Float64) raises -> Float64:
    if value <= 0.0:
        raise Error("ValueError: math domain error")
    if value != value or bitcast[DType.uint64](value) == UInt64(0x7FF0000000000000):
        return value
    return portable_log_c(value)


def gp_kernel_operation(kernel: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    if operation == "family_gp_kernel_initialize":
        return _kernel_construct(kernel, operation, args, context)
    if operation == "family_gp_kernel_add" or operation == "family_gp_kernel_mul":
        gp_admit_kernel(kernel, context)
        gp_admit_kernel(args[0], context)
        var combined = context["family_gp_classes"]["combined"]
        _kernel_allocation_class(combined)
        var out = native_storage_allocate(combined)
        var op = 4 if operation == "family_gp_kernel_add" else 5
        _ = _kernel_construct(out, "family_gp_kernel_initialize", Python.tuple(kernel, args[0], op, "+" if op == 4 else "*"), context)
        return out
    var snapshot = gp_kernel_snapshot(kernel, context)
    if operation == "family_gp_kernel_repr":
        return PythonObject(_kernel_name(kernel, context, snapshot))
    if operation == "family_gp_kernel_n_dims":
        return PythonObject(len(snapshot.values))
    var result = Python.list()
    if operation == "family_gp_kernel_theta":
        var fitted = _kernel_read(kernel, snapshot.states[0], "_theta_fit", True)
        if fitted is not None:
            _kernel_sequence(fitted)
            for i in range(len(fitted)):  # driver-loop: fitted theta copied to a fresh finished result
                _ = py_call(result.append, PythonObject(_kernel_scalar(fitted[i])))
            return result
        for i in range(len(snapshot.values)):  # driver-loop: native logarithm for each free hyperparameter
            _ = py_call(result.append, PythonObject(_kernel_log(snapshot.values[i])))
        return result
    if operation == "family_gp_kernel_bounds":
        for i in range(len(snapshot.lows)):  # driver-loop: native logarithm of each bound pair
            var pair = Python.list()
            _ = py_call(pair.append, PythonObject(_kernel_log(snapshot.lows[i])))
            _ = py_call(pair.append, PythonObject(_kernel_log(snapshot.highs[i])))
            _ = py_call(result.append, pair)
        return result
    raise Error("RuntimeError: unknown GP kernel metadata operation")


# Constructor inputs are captured through the same admitted native buffers used
# elsewhere by the public providers. Their shape, strides and scalar bytes are
# retained for this operation, never through a user's array/float protocol.
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_is_array
from providers.python.array_metadata_pyapi import array_slot_once, array_owner_once
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.native_numpy_metadata import native_numpy_dtype
from core.native_scalar_decode import native_scalar_real
from bindings.sequential_admission import _one_of
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.py_balanced import py_get, py_setitem


struct GPKernelInput(Movable):
    var shape: List[Int]
    var values: List[Float64]

    def __init__(out self):
        self.shape = List[Int]()
        self.values = List[Float64]()


def _kernel_input(value: PythonObject, context: PythonObject, depth: Int = 0) raises -> GPKernelInput:
    if depth > 64:
        raise Error("TypeError: GP kernel numeric input exceeds native container nesting")
    var result = GPKernelInput()
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        result.shape.append(len(value))
        for i in range(len(value)):  # driver-loop: recursively capture ordinary input containers once
            var child = _kernel_input(value[i], context, depth + 1)
            if i == 0:
                for j in range(len(child.shape)):  # driver-loop: first child defines rectangular rank
                    result.shape.append(child.shape[j])
            elif len(child.shape) + 1 != len(result.shape):
                raise Error("ValueError: GP kernel numeric inputs must be rectangular")
            else:
                for j in range(len(child.shape)):  # driver-loop: preserve shape refusals before flattening
                    if child.shape[j] != result.shape[j+1]:
                        raise Error("ValueError: GP kernel numeric inputs must be rectangular")
            for j in range(len(child.values)):  # driver-loop: append captured scalar words in logical order
                result.values.append(child.values[j])
        return result^
    if cls is b.float or cls is b.int or cls is b.bool or cls is b.str or cls is b.bytes or cls is b.bytearray:
        result.values.append(Float64(py=py_call(b.float, value)))
        return result^
    if py_get(context, "ndarray") is not None:
        var dtype = native_numpy_dtype(context)
        for code in ["q", "Q", "g", "U", "S"]:  # driver-loop: native long-long/long-double and text scalar aliases
            if cls is py_call(dtype, code).type:
                if code == "U" or code == "S":
                    result.values.append(Float64(py=py_call(b.float, value)))
                else:
                    result.values.append(bitcast[DType.float64](native_scalar_buffer(value).real.float64_bits()))
                return result^
    classical_admit_value(value, context)
    if _one_of(cls, context["scalar_types"]):
        var decoded = native_scalar_buffer(value)
        if decoded.complex:
            raise Error("TypeError: GP kernel input requires real numeric values")
        result.values.append(bitcast[DType.float64](decoded.real.float64_bits()))
        return result^
    var view = PythonObject(None)
    var owner = value
    var shape = Python.tuple()
    var strides = Python.tuple()
    var width = 0
    var code = String()
    var little = True
    var address = 0
    var exported = List[Int](length=10, fill=0)
    var pointer = exported.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var acquired = False
    ref cpy = Python().cpython()
    if classical_is_array(value, context):
        shape = array_slot_once(value, "shape", context)
        strides = array_slot_once(value, "strides", context)
        var dtype = array_slot_once(value, "dtype", context)
        var item = array_slot_once(value, "itemsize", context)
        var addr = array_slot_once(value, "_addr", context)
        owner = array_owner_once(value, context)
        if py_type(dtype) is not b.str or py_type(item) is not b.int or py_type(addr) is not b.int:
            raise Error("TypeError: GP kernel Array metadata must use native scalar fields")
        width = Int(py=item)
        address = Int(py=addr)
        var text = String(dtype)
        if text.byte_length() < 3:
            raise Error("TypeError: invalid GP kernel Array dtype")
        var kind = String(text[byte=1:2])
        code = String("i") if kind == "i" else String("I")
        if kind == "f":
            code = "e" if width == 2 else "f" if width == 4 else "d"
        elif kind != "i" and kind != "u":
            raise Error("TypeError: GP kernel Array requires a real numeric dtype")
        little = String(text[byte=0:1]) != ">"
    else:
        view = native_buffer_view(value)
        shape = view.shape
        strides = view.strides
        width = Int(py=view.itemsize)
        code = String(view.format)
        if code.byte_length() > 1:
            var prefix = String(code[byte=0:1])
            if prefix == "@" or prefix == "=" or prefix == "<" or prefix == ">" or prefix == "!":
                little = prefix != ">" and prefix != "!"
                var unprefixed = String(code[byte=1:])
                code = unprefixed^
        if cls is py_get(context, "ndarray") and (code == "O" or code.endswith("w") or code.endswith("s")):
            # Read owned scalar references through native indexing; conversion
            # and flattening stay here, without ndarray.tolist/astype callbacks.
            return _kernel_boxed_array(value, shape, context, depth + 1)
        if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(view._obj_ptr, pointer, Int32(0x118)) != 0:
            raise cpy.unsafe_get_error()
        acquired = True
        address = exported[0]
    try:
        if py_type(shape) is not b.tuple or py_type(strides) is not b.tuple or len(shape) != len(strides):
            raise Error("TypeError: GP kernel input requires native shape and stride tuples")
        var count = 1
        for i in range(len(shape)):  # driver-loop: one immutable shape/stride snapshot per input buffer
            if py_type(shape[i]) is not b.int or py_type(strides[i]) is not b.int or Int(py=shape[i]) < 0:
                raise Error("TypeError: GP kernel input dimensions must be nonnegative native integers")
            var dimension = Int(py=shape[i])
            if dimension != 0 and count > ((1 << 40) // dimension):
                raise Error("ValueError: GP kernel input is too large")
            count *= dimension
            result.shape.append(dimension)
        if count > 0 and address == 0:
            raise Error("ValueError: GP kernel input has a null buffer address")
        var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
        for index in range(count):  # driver-loop: native numeric decoding in logical C order, including negative strides
            var residual = index
            var offset = 0
            for reverse in range(len(result.shape)):  # driver-loop: captured rank metadata, no input callbacks
                var axis = len(result.shape) - reverse - 1
                offset += (residual % result.shape[axis]) * Int(py=strides[axis])
                residual //= result.shape[axis]
            result.values.append(bitcast[DType.float64](native_scalar_real(data, offset, code, width, little).float64_bits()))
    finally:
        if acquired:
            cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    _ = owner
    _ = view
    return result^


def _kernel_float_list(values: List[Float64]) raises -> PythonObject:
    var result = Python.list()
    for i in range(len(values)):  # driver-loop: finished native scalar list
        _ = py_call(result.append, PythonObject(values[i]))
    return result


def _kernel_bounds_input(value: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    if py_type(value) is py_type(PythonObject("")):
        if String(value) != "fixed":
            raise Error("ValueError: GP kernel bounds must be a pair or 'fixed'")
        return PythonObject("fixed")
    var input = _kernel_input(value, context)
    var broadcast = len(input.shape) == 1 and input.shape[0] == 2
    if not broadcast and not (len(input.shape) == 2 and input.shape[0] == count and input.shape[1] == 2):
        raise Error("ValueError: GP kernel bounds must be a pair or one pair per hyperparameter")
    var result = Python.list()
    for i in range(count):  # driver-loop: expand bounds once for every scalar or ARD parameter
        var offset = 0 if broadcast else 2*i
        var low = input.values[offset]
        var high = input.values[offset+1]
        if not (low - low == 0.0 and high - high == 0.0 and 0.0 < low <= high):
            raise Error("ValueError: every GP kernel bound must be finite and positive with low <= high")
        _ = py_call(result.append, Python.tuple(PythonObject(low), PythonObject(high)))
    return result


def _kernel_release(value: PythonObject, context: PythonObject, active: PythonObject) raises:
    var classes = py_call(native_loaded_module("builtins").list, context["family_gp_kernel_nodes"].keys())
    var connected = False
    for i in range(len(classes)):  # driver-loop: exact kernel owners need recursive lifecycle admission
        connected = connected or py_type(value) is classes[i]
    if not connected:
        native_release_admit(value, context)
        return
    for i in range(len(active)):  # driver-loop: avoid cycles in retained owner graphs
        if value is active[i]:
            raise Error("TypeError: native kernel replacement cannot release a cyclic owner graph")
    native_release_weakrefs(value, context)
    var ancestry = py_call(native_loaded_module("builtins").type.__dict__["__mro__"].__get__, py_type(value))
    for i in range(len(ancestry)):  # driver-loop: destruction cannot invoke a Python finalizer
        if py_contains(_class_fields(ancestry[i]), "__del__"):
            raise Error("TypeError: native kernel replacement cannot release Python finalizers")
    _ = py_call(active.append, value)
    var fields = _fields(value)
    var children = py_call(native_loaded_module("builtins").list, fields.values())
    for i in range(len(children)):  # driver-loop: every retained field can own a finalizer, not only kernel configuration
        _kernel_release(children[i], context, active)
    _ = py_call(active.pop)


def _kernel_publish(kernel: PythonObject, updates: PythonObject, context: PythonObject) raises:
    _ = _kernel_class(kernel, context)
    var fields = _fields(kernel)
    var names = py_call(native_loaded_module("builtins").list, updates.keys())
    for i in range(len(names)):  # driver-loop: validate all replacement owners before mutating any field
        var previous = _kernel_read(kernel, fields, names[i], True)
        if previous is not None and previous is not updates[names[i]]:
            _kernel_release(previous, context, Python.list())
    for i in range(len(names)):  # driver-loop: publish already-admitted state through the native dictionary
        py_setitem(fields, names[i], updates[names[i]])


def _kernel_construct(kernel: PythonObject, operation: String, args: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var cls = _kernel_class(kernel, context)
    var layout = context["family_gp_kernel_nodes"][cls]
    var how = String(layout[0])
    var updates = Python.dict()
    if how == "leaf":
        var input = _kernel_input(args[0], context)
        if layout[3] is None:
            if len(input.shape) != 0 or len(input.values) != 1:
                raise Error("TypeError: GP scalar kernel parameter must be a native scalar")
            py_setitem(updates, layout[2], PythonObject(input.values[0]))
        else:
            if len(input.values) == 0:
                raise Error("ValueError: GP kernel length_scale is an empty sequence")
            py_setitem(updates, layout[3], _kernel_float_list(input.values))
            if layout[2] is not None:
                var nu = _kernel_input(args[2], context)
                if len(nu.shape) != 0 or len(nu.values) != 1:
                    raise Error("TypeError: Matern nu must be a native scalar")
                py_setitem(updates, layout[2], PythonObject(nu.values[0]))
        py_setitem(updates, context["family_gp_kernel_bounds"][cls], _kernel_bounds_input(args[1], len(input.values), context))
    elif how == "pair":
        gp_admit_kernel(args[0], context)
        gp_admit_kernel(args[1], context)
        var op = args[2]
        var sym = args[3]
        if py_type(op) is not py_type(PythonObject(0)) or (Int(py=op) != 4 and Int(py=op) != 5) or py_type(sym) is not py_type(PythonObject("")):
            raise Error("TypeError: native GP kernel composition requires sum/product metadata")
        py_setitem(updates, "a", args[0])
        py_setitem(updates, "b", args[1])
        py_setitem(updates, "op", op)
        py_setitem(updates, "sym", sym)
    elif how == "nodes":
        var nodes = args[0]
        _kernel_sequence(nodes)
        var saved = Python.list()
        for i in range(len(nodes)):  # driver-loop: copy saved postfix node metadata without invoking an iterator
            var node = nodes[i]
            _kernel_sequence(node)
            if len(node) != 3:
                raise Error("ValueError: a saved GP kernel node requires three entries")
            var kind = _kernel_input(node[0], context)
            var parameter = _kernel_input(node[1], context)
            var scales = _kernel_input(node[2], context)
            if len(kind.shape) != 0 or len(parameter.shape) != 0:
                raise Error("TypeError: saved GP kernel kinds and parameters must be scalars")
            var k = py_call(native_loaded_module("builtins").int, PythonObject(kind.values[0]))
            _ = py_call(saved.append, Python.tuple(k, PythonObject(parameter.values[0]), _kernel_float_list(scales.values)))
        py_setitem(updates, layout[1], saved)
    else:
        raise Error("TypeError: this GP kernel constructor has no native public schema")
    _kernel_publish(kernel, updates, context)
    return PythonObject(None)


def _kernel_name(kernel: PythonObject, context: PythonObject, snapshot: GPKernelSnapshot) raises -> String:
    var cls = _kernel_class(kernel, context)
    var fields = PythonObject(None)
    for i in range(len(snapshot.owners)):  # driver-loop: reuse state captured by admission
        if kernel is snapshot.owners[i]:
            fields = snapshot.states[i]
            break
    var layout = context["family_gp_kernel_nodes"][cls]
    var how = String(layout[0])
    var b = native_loaded_module("builtins")
    if how == "pair":
        var op = Int(py=_kernel_read(kernel, fields, "op"))
        return "(" + _kernel_name(_kernel_read(kernel, fields, "a"), context, snapshot) + (" + " if op == 4 else " * ") + _kernel_name(_kernel_read(kernel, fields, "b"), context, snapshot) + ")"
    if how == "nodes":
        return "_SavedKernel(" + String(len(_kernel_read(kernel, fields, layout[1]))) + " nodes)"
    var code = Int(py=layout[1])
    if code == 0 or code == 1:
        var value = PythonObject(_kernel_scalar(_kernel_read(kernel, fields, layout[2])))
        return ("ConstantKernel(" if code == 0 else "WhiteKernel(") + String(py_call(b.repr, value)) + ")"
    var scales = _kernel_read(kernel, fields, layout[3])
    # Admission established exact list/tuple and exact numeric leaf types.
    var name = ("RBF(" if code == 2 else "Matern(") + String(py_call(b.repr, scales))
    if code == 3:
        name += ", nu=" + String(py_call(b.repr, PythonObject(_kernel_scalar(_kernel_read(kernel, fields, "nu")))))
    return name + ")"


def _kernel_allocation_class(cls: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: native GP kernel allocation refuses Python metaclass protocols")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: an owned construction failure must not invoke a finalizer
        if py_contains(_class_fields(ancestry[i]), "__del__"):
            raise Error("TypeError: native GP kernel allocation refuses Python finalizers")


def _kernel_node_scales(source: PythonObject) raises -> PythonObject:
    _kernel_sequence(source)
    var result = Python.list()
    for i in range(len(source)):  # driver-loop: copy admitted native scale words into the finished node
        _ = py_call(result.append, PythonObject(_kernel_scalar(source[i])))
    return result


def _kernel_nodes_walk(kernel: PythonObject, context: PythonObject, active: PythonObject,
                       mut snapshot: GPKernelSnapshot, nodes: PythonObject) raises:
    var cls = _kernel_class(kernel, context)
    for i in range(len(active)):  # driver-loop: refuse cyclic postfix graph construction
        if kernel is active[i]:
            raise Error("ValueError: cyclic GP kernel graph")
    if len(active) >= 512:
        raise Error("ValueError: GP kernel graph exceeds native traversal depth")
    _ = py_call(active.append, kernel)
    var fields = _kernel_state(kernel, snapshot)
    var layout = context["family_gp_kernel_nodes"][cls]
    var how = String(layout[0])
    if how == "pair":
        var left = _kernel_read(kernel, fields, layout[1])
        var right = _kernel_read(kernel, fields, layout[2])
        var op = _kernel_read(kernel, fields, layout[3])
        if py_type(op) is not py_type(PythonObject(0)) or (Int(py=op) != 4 and Int(py=op) != 5):
            raise Error("TypeError: native GP kernel operator must be the sum or product code")
        _kernel_nodes_walk(left, context, active, snapshot, nodes)
        _kernel_nodes_walk(right, context, active, snapshot, nodes)
        _ = py_call(nodes.append, Python.tuple(op, PythonObject(0.0), Python.list()))
    elif how == "nodes":
        var source = _kernel_read(kernel, fields, layout[1])
        _kernel_sequence(source)
        for i in range(len(source)):  # driver-loop: saved postfix nodes copied without Python conversions
            var node = source[i]
            _kernel_sequence(node)
            if len(node) != 3 or py_type(node[0]) is not py_type(PythonObject(0)):
                raise Error("TypeError: saved GP kernel nodes require an integer kind, parameter and scale list")
            _ = py_call(nodes.append, Python.tuple(node[0], PythonObject(_kernel_scalar(node[1])), _kernel_node_scales(node[2])))
    elif how == "leaf":
        var parameter = Float64(0.0)
        if layout[2] is not None:
            parameter = _kernel_scalar(_kernel_read(kernel, fields, layout[2]))
        var scales = Python.list()
        if layout[3] is not None:
            scales = _kernel_node_scales(_kernel_read(kernel, fields, layout[3]))
        _ = py_call(nodes.append, Python.tuple(layout[1], PythonObject(parameter), scales))
    else:
        raise Error("RuntimeError: unknown connected GP kernel layout")
    _ = py_call(active.pop)


def gp_kernel_nodes(kernel: PythonObject, table: PythonObject) raises -> PythonObject:
    # Archive/kernel-array consumers supply only their import-connected node
    # layout table. Bounds are deliberately unnecessary for saved fixed kernels.
    # Each distinct owner dictionary is captured once and retained through the
    # entire native postfix walk; the returned tuples/lists own their scalars.
    var context = Python.dict(family_gp_kernel_nodes=table)
    var snapshot = GPKernelSnapshot()
    var nodes = Python.list()
    _kernel_nodes_walk(kernel, context, Python.list(), snapshot, nodes)
    return nodes


def _kernel_boxed_array(value: PythonObject, shape: PythonObject, context: PythonObject,
                        depth: Int) raises -> GPKernelInput:
    var result = GPKernelInput()
    var count = 1
    for axis in range(len(shape)):  # driver-loop: captured native ndarray shape
        var size = Int(py=shape[axis])
        result.shape.append(size)
        count *= size
    ref cpy = Python().cpython()
    for index in range(count):  # driver-loop: scalar storage references, never NumPy numeric conversion
        var residual = index
        var coordinates = List[Int](length=len(result.shape), fill=0)
        for reverse in range(len(result.shape)):  # driver-loop: logical row-major index from captured extents
            var axis = len(result.shape) - reverse - 1
            coordinates[axis] = residual % result.shape[axis]
            residual //= result.shape[axis]
        var words = Python.list()
        for axis in range(len(coordinates)):  # driver-loop: native scalar index tuple
            _ = py_call(words.append, PythonObject(coordinates[axis]))
        var key = py_call(py_type(Python.tuple()), words)
        var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, key._obj_ptr)
        if not raw:
            raise cpy.unsafe_get_error()
        var item = PythonObject(from_owned=raw)
        var child = _kernel_input(item, context, depth)
        if len(result.shape) == 0:
            return child^
        for scalar in child.values:  # driver-loop: admitted native scalar words only
            result.values.append(scalar)
    return result^
