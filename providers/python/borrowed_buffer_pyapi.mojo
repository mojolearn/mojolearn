# SPDX-License-Identifier: Apache-2.0
# Pin an ALREADY-ADMITTED native buffer exporter for an entire operation. The
# caller first applies its shared exact-type/native-exporter admission policy;
# this is not permission to invoke arbitrary Python __buffer__ protocols.
# Py_buffer is the public stable CPython structure, represented by ten native
# words on supported 64-bit little-endian targets (same as optimizer tensor_view).
from std.python import Python, PythonObject
from core.borrowed_buffer import BorrowedBufferLayout


struct AdmittedBorrowedBuffer(Movable):
    var owner: PythonObject
    var exported: List[Int]
    var acquired: Bool
    var format: String
    var layout: Optional[BorrowedBufferLayout]

    def __init__(out self, admitted_exporter: PythonObject, writable: Bool = False) raises:
        self.owner = admitted_exporter
        self.exported = List[Int](length=10, fill=0)
        self.acquired = False
        self.format = String("")
        self.layout = None
        ref cpy = Python().cpython()
        var pointer = self.exported.unsafe_ptr()
        # PyBUF_FULL(_RO): retain format, shape, strides and indirect storage.
        if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(
            admitted_exporter._obj_ptr, pointer, Int32(0x11D if writable else 0x11C)) != 0:
            raise cpy.unsafe_get_error()
        self.acquired = True
        try:
            var packed = UInt64(self.exported[4])
            var readonly = (packed & UInt64(0xffffffff)) != 0
            var rank = Int(packed >> 32)
            if rank < 0 or rank > 64:
                raise Error("native buffer: invalid rank")
            if writable and readonly:
                raise Error("TypeError: native buffer requires writable storage")
            if self.exported[5] != 0:
                self.format = String(unsafe_from_utf8_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=self.exported[5]))
            var shape = List[Int]()
            var strides = List[Int]()
            var suboffsets = List[Int]()
            var dimensions = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=self.exported[6])
            var steps = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=self.exported[7])
            var indirect = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=self.exported[8])
            if rank != 0 and (self.exported[6] == 0 or self.exported[7] == 0):
                raise Error("native buffer: exporter omitted requested shape/strides")
            for axis in range(rank):  # small-loop(admit shape/stride/suboffset words once)
                shape.append(dimensions[axis])
                strides.append(steps[axis])
                suboffsets.append(indirect[axis] if self.exported[8] != 0 else -1)
            self.layout = BorrowedBufferLayout(self.exported[0], self.exported[3], readonly,
                shape^, strides^, suboffsets^)
            if self.layout.value().count * self.layout.value().itemsize != self.exported[2]:
                raise Error("native buffer: exported size differs from its shape")
        except problem:
            self.release()
            raise problem

    def release(mut self) raises:
        if self.acquired:
            self.acquired = False
            ref cpy = Python().cpython()
            cpy.lib.get_function[NoneType]("PyBuffer_Release")(self.exported.unsafe_ptr())

    def __deinit__(deinit self):
        if self.acquired:
            try:
                ref cpy = Python().cpython()
                cpy.lib.get_function[NoneType]("PyBuffer_Release")(self.exported.unsafe_ptr())
            except:
                pass
