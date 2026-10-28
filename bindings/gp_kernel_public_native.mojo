# SPDX-License-Identifier: Apache-2.0
# Whole-operation GP kernel metadata owner. Only import-connected, exact kernel
# classes and passive native state are admitted; no kernel method, descriptor,
# numeric protocol, or sequence protocol executes in this traversal.
from std.python import Python, PythonObject
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
    var snapshot = gp_kernel_snapshot(kernel, context)
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
