# SPDX-License-Identifier: Apache-2.0
# Shared sequence-layer input snapshots. Native metadata is captured once per
# argument; original owners and any prepared buffers remain alive through use.
from std.python import PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.py_typed_call import py_call_typed
from bindings.optimizer_admission import opt_value
from bindings.optimizer_public_door import py_ints
from bindings.optimizer_device_native import device_shape, shape_count
from bindings.sequence_optimizer_native import seq_binding
from bindings.classical_arrays import classical_input, classical_typed, classical_address
from bindings.classical_attributes import classical_connect_numpy
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from core.sequence_optimizer_inputs import sequence_optimizer_cast
from core.cast_in import CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _array(shape: List[Int], context: PythonObject, wide: Bool = False) raises -> PythonObject:
    _ = shape_count(shape)
    var input = context["input_context"]
    classical_connect_numpy(input)
    return classical_typed(py_ints(shape), "<i8" if wide else "<f4", input)


struct SequenceFloatBuffer(Movable):
    # The original owner, admitted exporter and any converted storage stay live
    # until the operation ends. Every data access uses this one metadata pin.
    var original: PythonObject
    var pin: AdmittedBorrowedBuffer
    var array: PythonObject
    var shape: List[Int]
    var count: Int
    var address: Int
    var readonly: Bool

    def __init__(out self, value: PythonObject, context: PythonObject,
                 copy: Bool = False, refuse_double: Bool = True, strict: Bool = False) raises:
        self.original = value
        # Array slot admission is owned by classical_input/native_array_ndarray_view;
        # avoid reading its shape/storage metadata a second time here.
        if py_type(value) is not context["array_class"]:
            opt_value(value, context)
        var input = context["input_context"]
        classical_connect_numpy(input)
        var source = classical_input(value, input, "sequence layer")
        admit_native_buffer_exporter(source)
        self.pin = AdmittedBorrowedBuffer(source)
        var layout = self.pin.layout.value().copy()
        self.shape = layout.shape.copy()
        self.count = layout.count
        self.readonly = layout.readonly
        if self.count > 2147483647:
            raise Error("ValueError: sequence layer input exceeds signed 32-bit indexing")
        var code = self.pin.format
        var native = True
        if code.byte_length() > 1:
            var prefix = String(code[byte=0:1])
            if prefix == "@" or prefix == "=" or prefix == "<" or prefix == ">" or prefix == "!":
                native = prefix != ">" and prefix != "!"
                var unprefixed = String(code[byte=1:])
                code = unprefixed^
        var kind = -1
        if code == "f" or code == "e" or code == "d":
            kind = CAST_KIND_FLOAT
            var width = 4 if code == "f" else (2 if code == "e" else 8)
            if width != layout.itemsize:
                raise Error("TypeError: sequence layer input float format and width differ")
        elif code == "b" or code == "h" or code == "i" or code == "l" or code == "q" or code == "n":
            kind = CAST_KIND_SIGNED
        elif code == "B" or code == "H" or code == "I" or code == "L" or code == "Q" or code == "N":
            kind = CAST_KIND_UNSIGNED
        elif code == "?":
            kind = CAST_KIND_BOOL
        if kind < 0 or (layout.itemsize != 1 and layout.itemsize != 2 and layout.itemsize != 4 and layout.itemsize != 8):
            raise Error("TypeError: sequence layer input requires an ordinary numeric buffer")
        if refuse_double and code == "d":
            raise Error("TypeError: sequence layer: float64 is refused; pass float32")
        if strict and (code != "f" or not native):
            raise Error("TypeError: mojolearn: to_device takes float32; convert it yourself")
        for offset in layout.suboffsets:  # driver-loop: over indirect buffer admission
            if offset >= 0:
                raise Error("TypeError: sequence layer requires direct numeric storage")
        if not copy and code == "f" and native and layout.contiguous():
            self.array = source
            self.address = layout.address
        else:
            self.array = _array(self.shape, context)
            self.address = Int(py=classical_address(self.array))
            sequence_optimizer_cast(layout.address, layout.shape, layout.strides,
                layout.itemsize, kind, native, self.address, self.count, False)
            self.readonly = False

struct SequenceResidentSnapshot(Movable):
    var owner: PythonObject
    var entries: PythonObject
    var handle: PythonObject
    var shape: List[Int]
    var count: Int

    def __init__(out self, value: PythonObject, context: PythonObject) raises:
        if py_type(value) is not context["sequence_device_class"]:
            raise Error("TypeError: sequence layer device input requires its exact native tensor")
        self.owner = value
        var get = context["sequence_device_getters"]
        self.handle = py_call(get["h"], value)
        if py_type(self.handle) is not _b().int:
            raise Error("TypeError: sequence layer device handle requires an ordinary integer")
        self.shape = device_shape(py_call(get["shape"], value), context)
        self.count = shape_count(self.shape)
        if self.count > 2147483647:
            raise Error("ValueError: sequence layer device input exceeds signed 32-bit indexing")
        self.entries = seq_binding(PythonObject(None), context, py_call(get["b"], value))

    def host(self, context: PythonObject) raises -> PythonObject:
        var result = _array(self.shape, context)
        if self.count > 0:
            _ = py_call_typed(self.entries["seq_tensor_download"], self.handle,
                               classical_address(result), PythonObject(self.count))
            _ = result
        return result

