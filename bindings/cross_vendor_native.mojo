# SPDX-License-Identifier: Apache-2.0
"""Native byte admission, identity folds and state hashes for live training."""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_transport import transport_initialize, transport_send, transport_receive, _transport_exact


struct CrossVendorBoundary(Defaultable, Movable):
    var bi: PythonObject
    var arrays: PythonObject
    var array_cls: PythonObject
    var hash_type: PythonObject
    var mismatch_cls: PythonObject
    var fold_pair: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.arrays = PythonObject(None)
        self.array_cls = PythonObject(None)
        self.hash_type = PythonObject(None)
        self.mismatch_cls = PythonObject(None)
        self.fold_pair = PythonObject(None)


comptime CROSS_VENDOR_BOUNDARY = _Global[
    StorageType=CrossVendorBoundary, name="MojoCrossVendorBoundary",
    init_fn=CrossVendorBoundary.__init__,
]


def cross_vendor_initialize(array_cls: PythonObject, mismatch_cls: PythonObject, fold_pair: PythonObject) raises:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    c[].bi = Python.import_module("builtins")
    c[].arrays = Python.import_module("array").array
    c[].array_cls = array_cls
    c[].mismatch_cls = mismatch_cls
    c[].hash_type = Python.import_module("hashlib").sha256
    c[].fold_pair = fold_pair
    if c[].bi.type(c[].hash_type) is not c[].bi.type(c[].bi.len):
        raise Error("cross_vendor: SHA256 must be a native extension entry")
    if c[].bi.type(c[].fold_pair) is not c[].bi.type(c[].bi.len):
        raise Error("cross_vendor: fold must be a native extension entry")
    transport_initialize()


def cross_vendor_bytes(value: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if c[].bi.type(value) is c[].array_cls:
        return value._mv.cast("B")
    return c[].bi.memoryview(value).cast("B")


def cross_vendor_f32(value: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var raw = cross_vendor_bytes(value)
    if len(raw) % 4 != 0:
        raise Error("cross_vendor: gradient bytes must be a multiple of four")
    return raw.cast("f")


def cross_vendor_fold(gradients: PythonObject, prefix: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    var parts = c[].bi.list(gradients)
    var first = 0
    var source = prefix
    if source is None:
        if len(parts) == 0:
            raise Error("ordered_fold needs at least one gradient")
        source = parts[0]
        first = 1
    var store = c[].arrays("f")
    store.frombytes(cross_vendor_f32(source).cast("B"))
    var n = len(store)
    var total = store.buffer_info()[0]
    for k in range(first, len(parts)):  # driver-loop: over logical shard gradient buffers in immutable order
        var shard = c[].arrays("f")
        shard.frombytes(cross_vendor_f32(parts[k]).cast("B"))
        if len(shard) != n:
            raise Error("ordered_fold: gradients differ in length")
        # Retain the selected operation's existing GPU/CPU execution route.
        # Mojo owns the immutable fold; the cached compiled entry owns its
        # arithmetic, including FTZ and the one float32 rounding per add.
        _ = c[].fold_pair(total, shard.buffer_info()[0], n)
        # Explicitly retain the C array owning the address across every read;
        # converting to an integer pointer erases the compiler's origin edge.
        _ = shard^
    var result = store.tobytes()
    _ = store
    return result


def cross_vendor_state_hash(state: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if c[].bi.type(state) is not c[].bi.dict:
        raise Error("cross_vendor: state must be a dict")
    var hasher = c[].hash_type()
    for key in [String("parameters"), String("m"), String("v"), String("flags")]:
        hasher.update(cross_vendor_bytes(state[key]))
    return hasher.hexdigest()


def cross_vendor_call(operation: String, args: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_BOUNDARY.get_or_create_ptr()
    if operation == "initialize":
        cross_vendor_initialize(args[0], args[1], args[2])
    elif operation == "f32":
        return cross_vendor_f32(args[0])
    elif operation == "fold":
        return cross_vendor_fold(args[0], args[1])
    elif operation == "pair":
        var parts = c[].bi.list()
        parts.append(args[0])
        parts.append(args[1])
        return cross_vendor_fold(parts, PythonObject(None))
    elif operation == "state_hash":
        return cross_vendor_state_hash(args[0])
    elif operation == "send":
        transport_send(args[0], args[1], args[2])
    elif operation == "receive":
        return transport_receive(args[0], Int(py=args[1]))
    elif operation == "receive_exact":
        return _transport_exact(args[0], Int(py=args[1]))
    else:
        raise Error("cross_vendor: unknown native operation " + operation)
    return PythonObject(None)


def cross_vendor_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    var operation = String()
    try:
        operation = String(args[0])
        var result = cross_vendor_call(operation, args[1])
        return result.steal_data()
    except error:
        if operation == "send" or operation == "receive" or operation == "receive_exact":
            return raise_python_exception(error, ExceptionType("PyExc_ConnectionError"))
        return raise_python_exception(error, ExceptionType.ValueError)
