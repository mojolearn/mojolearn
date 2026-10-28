# SPDX-License-Identifier: Apache-2.0
# Pin an ALREADY-ADMITTED native buffer exporter for an entire operation. The
# caller first applies its shared exact-type/native-exporter admission policy;
# this is not permission to invoke arbitrary Python __buffer__ protocols.
# Py_buffer is the public stable CPython structure, represented by ten native
# words on supported 64-bit little-endian targets (same as optimizer tensor_view).
from std.python import Python, PythonObject
from std.ffi import _Global
from core.borrowed_buffer import BorrowedBufferLayout
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get, py_fields, py_call, py_contains


struct NativeMmapExporterConnection(Defaultable, Movable):
    var cls: PythonObject

    def __init__(out self):
        self.cls = PythonObject(None)


comptime NATIVE_MMAP_EXPORTER = _Global[StorageType=NativeMmapExporterConnection,
    name="MojoNativeMmapExporter", init_fn=NativeMmapExporterConnection.__init__]


def connect_native_mmap_exporter(cls: PythonObject) raises:
    # Import-time only. The product path compares a retained identity without
    # reading a mutable module namespace to decide whether admission applies.
    if cls is not native_loaded_module("mmap").mmap:
        raise Error("TypeError: native mmap connection requires its canonical extension type")
    var state = NATIVE_MMAP_EXPORTER.get_or_create_ptr()
    if state[].cls is not None and state[].cls is not cls:
        raise Error("TypeError: native mmap connection cannot be replaced")
    state[].cls = cls


def admit_native_buffer_exporter(value: PythonObject, depth: Int = 0) raises:
    # Shared no-context exporter admission. Exact memoryviews may retain a
    # Python-defined exporter, so inspect their base before acquiring a pin.
    # This routine never invokes a buffer protocol while deciding admission.
    if depth > 64:
        raise Error("TypeError: native buffer exporter nesting exceeds the supported limit")
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is bi.bytes or cls is bi.bytearray:
        return
    if cls is bi.memoryview:
        var owner = value.obj
        if owner is not None:
            admit_native_buffer_exporter(owner, depth + 1)
        return
    var sys = native_loaded_module("sys")
    var array_module = py_get(sys.modules, "array")
    if array_module is not None:
        array_module = native_loaded_module("array")
        if cls is py_get(py_fields(array_module), "array"):
            return
    if cls is NATIVE_MMAP_EXPORTER.get_or_create_ptr()[].cls:
        # Corpus batches retain their original native mmap owner. Import setup
        # pins this exact extension type; subclasses or replaced buffer slots
        # must not run Python before the operation captures its storage.
        var fields = py_call(bi.type.__dict__["__dict__"].__get__, cls)
        for name in ["__buffer__", "__release_buffer__"]:  # small-loop(native mmap buffer protocol slots)
            var descriptor = py_get(fields, PythonObject(name))
            if descriptor is not None:
                if (py_type(descriptor) is not py_type(bi.object.__repr__)
                        and py_type(descriptor) is not py_type(bi.bytes.decode)):
                    raise Error("TypeError: replaced mmap buffer protocols are unsupported")
                if descriptor.__objclass__ is not cls:
                    raise Error("TypeError: mmap buffer protocols require their own native descriptors")
        return
    var numpy_module = py_get(sys.modules, "numpy")
    if numpy_module is not None and py_type(numpy_module) is py_type(sys):
        # ndarray is a static extension type. A substituted Python class or
        # ndarray subclass cannot qualify merely by occupying this export.
        ref cpy = Python().cpython()
        var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(cls._obj_ptr)
        if (flags & UInt64(512)) == 0 and cls is py_get(py_fields(numpy_module), "ndarray"):
            return
    var ctypes_module = py_get(sys.modules, "_ctypes")
    if ctypes_module is not None and py_type(ctypes_module) is py_type(sys):
        var base = py_get(py_fields(ctypes_module), "Array")
        var bases = py_call(bi.type.__dict__["__bases__"].__get__, cls)
        if base is not None and len(bases) == 1 and bases[0] is base:
            # Same admission as Sequential: a direct native ctypes array
            # class may provide storage, but Python buffer/release hooks may not.
            var fields = py_call(bi.type.__dict__["__dict__"].__get__, cls)
            if not py_contains(fields, "__buffer__") and not py_contains(fields, "__release_buffer__"):
                return
    raise Error("TypeError: native buffers require an exact native exporter; Python buffer hooks and subclasses are not supported")


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
