# SPDX-License-Identifier: Apache-2.0
# NumPy's public NpyString storage API; formatting remains Mojo code.
from std.ffi import c_char, c_int, c_uint, c_size_t, external_call
from std.python import Python, PythonObject
from std.sys import size_of
from std.sys.info import CompilationTarget
from std.memory import bitcast
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.sequence_array_warnings import sequence_array_exception, sequence_array_report
from core.sequence_array_format import sequence_float_text
from core.sequence_array_cast import sequence_signaling, SEQ_CAST_INVALID

comptime StringStoragePointer = MutPointer[UInt8, MutUntrackedOrigin]
comptime StringStorageVersion = def() thin abi("C") -> c_uint
comptime StringStorageAcquire = def(StringStoragePointer) thin abi("C") -> Optional[StringStoragePointer]
comptime StringStorageRelease = def(StringStoragePointer) thin abi("C") -> NoneType
comptime StringStoragePack = def(StringStoragePointer, StringStoragePointer, StringStoragePointer, c_size_t) thin abi("C") -> c_int


def sequence_string_class(context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var module = py_get(context["modules"], "numpy.dtypes")
    if module is None:
        return PythonObject(None)
    if py_type(module) is not context["module_type"]:
        raise Error("TypeError: string dtype metadata requires NumPy's native module")
    var result = py_get(py_fields(module), "StringDType")
    if result is None:
        return result
    var dtype = py_call(native_numpy_dtype(context), "T")
    if py_type(dtype) is not result:
        raise Error("TypeError: StringDType native class identity has changed")
    return result


def sequence_string_na_value(value: PythonObject, classes: PythonObject, seen: PythonObject) raises:
    var b = Python.import_module("builtins")
    var kind = py_type(value)
    if value is None or kind is b.str or kind is b.bytes or kind is b.bool or kind is b.int or kind is b.float or kind is b.complex:
        return
    for i in range(len(classes)):  # driver-loop: admit native scalar metadata before descriptor copying
        if kind is classes[i][1]:
            # A raw void scalar has native equality/text metadata. Structured
            # void scalars delegate those operations to NumPy's Python
            # _promote_fields/arrayprint helpers during descriptor cloning.
            if String(kind.__name__) != "void" or (value.dtype.fields is None and not Bool(py=value.dtype.hasobject)):
                return
            raise Error("TypeError: structured void missing values require Python dtype/printing hooks; use a native scalar or string sentinel")
    if kind is b.list or kind is b.tuple or kind is b.dict:
        for i in range(len(seen)):  # driver-loop: stop native container cycles by identity
            if value is seen[i]:
                return
        py_call(seen.append, value)
        var items = py_call(b.list, value) if kind is b.dict else value
        for i in range(len(items)):  # driver-loop: audit all native missing-value metadata before its native repr
            sequence_string_na_value(items[i], classes, seen)
            if kind is b.dict:
                sequence_string_na_value(value[items[i]], classes, seen)
        return
    raise Error("TypeError: StringDType missing values require native scalar/container metadata; use None, a native number or a string instead of Python hooks")


def sequence_string_admit(dtype: PythonObject, classes: PythonObject, context: PythonObject) raises:
    if py_type(dtype) is not sequence_string_class(context):
        raise Error("TypeError: string output requires NumPy's native StringDType")
    var b = Python.import_module("builtins")
    var missing = py_call(b.getattr, dtype, "na_object", PythonObject(None))
    sequence_string_na_value(missing, classes, Python.list())


def sequence_storage_image(address: Int) raises -> Int:
    # dladdr is the supported POSIX owner lookup. Check the capsule table and
    # each function before dereferencing/calling it; a foreign capsule or a
    # Python callback trampoline is not a native NumPy storage entry.
    if address == 0:
        raise Error("TypeError: NumPy storage entry must not be null")
    var info = Array[Int, 4](uninitialized=True)
    var found = external_call["dladdr", c_int](StringStoragePointer(unsafe_from_address=address), info.unsafe_ptr())
    if found == 0:
        raise Error("TypeError: NumPy storage pointers require their native extension owner")
    return info[1]


struct SequenceStringStorage(Movable):
    var module: PythonObject
    var capsule: PythonObject
    var dtype: PythonObject
    var release: StringStorageRelease
    var pack: StringStoragePack
    var allocator: StringStoragePointer

    def __init__(out self, dtype: PythonObject, context: PythonObject) raises:
        # NpyString_acquire_allocator requires an actual StringDType
        # descriptor, never a structured parent or a requested predecessor.
        # Keep this guard at the C API boundary as well as schema admission.
        if py_type(dtype) is not sequence_string_class(context):
            raise Error("TypeError: native string storage requires its exact StringDType descriptor")
        ref cpy = Python().cpython()
        var numpy_dtype = native_numpy_dtype(context)
        var module = py_get(context["modules"], "numpy._core._multiarray_umath")
        if module is None:
            module = py_get(context["modules"], "numpy.core._multiarray_umath")
        if module is None or py_type(module) is not context["module_type"]:
            raise Error("TypeError: native StringDType storage requires NumPy's extension module")
        var capsule = py_get(py_fields(module), "_ARRAY_API")
        var retained = py_get(context, "sequence_string_capsule")
        if retained is not None and retained is not capsule:
            raise Error("TypeError: NumPy's native storage capsule has been replaced")
        var null_name = Optional[StringStoragePointer](None)
        if capsule is None or cpy.lib.get_function[c_int]("PyCapsule_IsValid")(capsule._obj_ptr, null_name) == 0:
            raise Error("TypeError: NumPy's native storage API requires its capsule")
        var pointer = cpy.lib.get_function[Optional[StringStoragePointer]]("PyCapsule_GetPointer")(capsule._obj_ptr, null_name)
        if not pointer:
            raise cpy.unsafe_get_error()
        var address = Int(pointer.value())
        var owner = sequence_storage_image(Int(numpy_dtype._obj_ptr))
        if sequence_storage_image(address) != owner:
            raise Error("TypeError: NumPy storage table has a foreign native owner")
        var words = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
        # Public stable C API indices from numpy/__multiarray_api.h:
        # GetNDArrayCFeatureVersion=211; pack=314; acquire=316; release=318.
        # The NpyString slots require NPY_2_0_API_VERSION (0x12).
        if sequence_storage_image(words[211]) != owner:
            raise Error("TypeError: NumPy storage version entry has a foreign owner")
        var version = MutPointer[StringStorageVersion, MutUntrackedOrigin](unsafe_from_address=address + 211 * size_of[Int]())[0]
        if version() < c_uint(0x12):
            raise Error("TypeError: managed string output requires NumPy 2's native storage API")
        for index in [314, 316, 318]:  # driver-loop: validate the three public native storage entry owners
            if sequence_storage_image(words[index]) != owner:
                raise Error("TypeError: NumPy storage entry has a foreign native owner")
        var acquire = MutPointer[StringStorageAcquire, MutUntrackedOrigin](unsafe_from_address=address + 316 * size_of[Int]())[0]
        var release = MutPointer[StringStorageRelease, MutUntrackedOrigin](unsafe_from_address=address + 318 * size_of[Int]())[0]
        var pack = MutPointer[StringStoragePack, MutUntrackedOrigin](unsafe_from_address=address + 314 * size_of[Int]())[0]
        # Publish retained metadata before locking, so no raising operation
        # can strand an acquired allocator during construction.
        py_setitem(context, "sequence_string_capsule", capsule)
        var allocator = acquire(StringStoragePointer(unsafe_from_address=Int(dtype._obj_ptr)))
        if not allocator:
            sequence_array_exception("MemoryError", "native string output could not acquire its allocator")
        self.module = module^
        self.capsule = capsule^
        self.dtype = dtype
        self.release = release
        self.pack = pack
        self.allocator = allocator.value()

    def __del__(deinit self):
        self.release(self.allocator)
        # Keep the descriptor and owning extension alive through release.
        _ = self.dtype^
        _ = self.capsule^
        _ = self.module^

    def write(self, bits: UInt32, destination: Int) raises:
        var text = sequence_float_text(bits)
        var data = StringStoragePointer(unsafe_from_address=Int(text.unsafe_ptr()))
        if self.pack(self.allocator, StringStoragePointer(unsafe_from_address=destination), data, c_size_t(text.byte_length())) < 0:
            sequence_array_exception("MemoryError", "native string output could not pack UTF-8 data")
        _ = text^


def sequence_string_nan_word(word: UInt64, exponent: UInt64, fraction: UInt64, quiet: UInt64) -> Int:
    if (word & exponent) == exponent and (word & fraction) != 0:
        return 1 | (4 if (word & quiet) == 0 else 0)
    return 0


def sequence_string_nan_raw(address: Int, width: Int) -> Int:
    if width == 2:
        var word = UInt64(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=address)[0])
        return sequence_string_nan_word(word, 0x7C00, 0x03FF, 0x0200)
    if width == 4:
        var word = UInt64(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)[0])
        return sequence_string_nan_word(word, 0x7F800000, 0x007FFFFF, 0x00400000)
    var low = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address)[0]
    if width == 8:
        return sequence_string_nan_word(low, 0x7FF0000000000000, 0x000FFFFFFFFFFFFF, 0x0008000000000000)
    comptime if CompilationTarget.is_x86():
        var exponent = MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=address + 8)[0] & UInt16(0x7FFF)
        if exponent == UInt16(0x7FFF) and (low & UInt64(0x7FFFFFFFFFFFFFFF)) != 0:
            return 1 | (4 if (low & UInt64(0x4000000000000000)) == 0 else 0)
    else:
        var high = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address + 8)[0]
        if (high & UInt64(0x7FFF000000000000)) == UInt64(0x7FFF000000000000) and ((high & UInt64(0x0000FFFFFFFFFFFF)) != 0 or low != 0):
            return 1 | (4 if (high & UInt64(0x0000800000000000)) == 0 else 0)
    return 0


def sequence_string_nan_info(dtype: PythonObject, classes: PythonObject, context: PythonObject) raises -> Int:
    var b = Python.import_module("builtins")
    var missing = py_call(b.getattr, dtype, "na_object", PythonObject(None))
    var kind = py_type(missing)
    if kind is b.float or kind is b.complex:
        var real = Float64(py=missing.real) if kind is b.complex else Float64(py=missing)
        var imag = Float64(py=missing.imag) if kind is b.complex else Float64(0)
        var info = sequence_string_nan_word(bitcast[DType.uint64](real), 0x7FF0000000000000, 0x000FFFFFFFFFFFFF, 0x0008000000000000)
        info |= sequence_string_nan_word(bitcast[DType.uint64](imag), 0x7FF0000000000000, 0x000FFFFFFFFFFFFF, 0x0008000000000000)
        return info | (2 if info != 0 and kind is b.complex else 0)
    for i in range(len(classes)):  # driver-loop: classify native numeric missing-value metadata
        if kind is classes[i][1]:
            var native_dtype = py_call(native_numpy_dtype(context), kind)
            var number_kind = String(native_dtype.kind)
            if number_kind != "f" and number_kind != "c":
                return 0
            var components = 2 if number_kind == "c" else 1
            var width = Int(py=native_dtype.itemsize) // components
            if width != 2 and width != 4 and width != 8 and width != 16:
                raise Error("TypeError: native missing-value scalar has an unsupported floating layout")
            # Reading the scalar's native buffer preserves signaling payloads;
            # widening through __float__ would quiet float16/float32 metadata.
            ref cpy = Python().cpython()
            var view = List[Int](length=10, fill=0)
            var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
            var owner = missing
            if cpy.lib.get_function[c_int]("PyObject_GetBuffer")(owner._obj_ptr, pointer, c_int(0)) != 0:
                raise cpy.unsafe_get_error()
            var info = 0
            for component in range(components):  # driver-loop: inspect one real or two complex native scalar words
                info |= sequence_string_nan_raw(view[0] + component * width, width)
            cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
            _ = owner^
            return info
    return 0


def sequence_string_leaf(source: Int, destination: Int, count: Int, stride: Int, offset: Int,
                         dtype: PythonObject, classes: PythonObject, settings: PythonObject, context: PythonObject, initial_status: Int) raises -> Int:
    var storage = SequenceStringStorage(dtype, context)
    var missing = sequence_string_nan_info(dtype, classes, context)
    var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source)
    var status = initial_status
    for i in range(count):  # driver-loop: store each completed native string through NumPy's public allocation API
        if (missing & 2) != 0:
            # A built-in complex NaN sentinel dispatches native scalar
            # equality through NumPy's equal ufunc. Each call clears earlier
            # FP flags, including flags from prior fields/elements. Only
            # that equality's final status survives to the outer cast.
            status = 0
        if (missing & 1) != 0 and (sequence_signaling(words[i]) or (missing & 4) != 0):
            status |= SEQ_CAST_INVALID
            if (missing & 2) != 0:
                # NumPy's na_eq_cmp clears the scalar equality exception,
                # while the final cast still reports its accumulated flag.
                # Suppress that intermediate native error without swallowing
                # callback-admission failures or altering warning delivery.
                sequence_array_report(SEQ_CAST_INVALID, settings, context, "equal", True)
        storage.write(words[i], destination + i * stride + offset)
    _ = storage^
    return status
