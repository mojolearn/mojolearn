# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A PyCapsule that owns one resident device handle of a compiled binding.

`resident_capsule(free, handle)` returns a capsule holding a strong reference
to the minting binding's release entry `free` and the handle word. When the
last reference to the capsule goes (a prepared-owner record replaced by a
refit or a new fitted array, or dropped by the model's death through the
`native_prepared_owner` weakref), its native destructor calls
`free(handle)` (no Python function runs) and so returns the device memory to
the binding that minted it. Moved out of bindings/decomp_kit_native.mojo
(lane perf2-pca-decomp) by lane ne1-a-decomp-tsa so the decomposition kit,
the TSA owners and the random-projection owner share one copy (plan section 4,
lesson 8). Nothing here touches data.
"""
from std.ffi import _Global, external_call, c_size_t
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call
from bindings.cpython_table import cpython_table


struct _ResidentNames(Defaultable, Movable):
    var capsule_name: String

    def __init__(out self):
        # PyCapsule keeps this name pointer, not a copy: the DSO-global owner
        # is never reassigned and outlives every capsule.
        self.capsule_name = String("mojolearn.decomp_kit.resident.v1")


comptime _RESIDENT_NAMES = _Global[
    StorageType=_ResidentNames, name="MojolearnDecompResidentNamesV1",
    init_fn=_ResidentNames.__init__,
]

comptime _ResidentWords = MutPointer[Int, MutUntrackedOrigin]
comptime _ResidentBytes = MutPointer[UInt8, MutUntrackedOrigin]


def _resident_destroy(capsule: PyObjectPtr) abi("C") -> NoneType:
    # Native destructor: returns the device buffer through the binding's
    # compiled `x_decomp_dev_free` (no Python function runs). A pending
    # exception of the code that dropped the record is kept; a failing free
    # (interpreter shutdown: the pool dies with the process) is quiet, as
    # the Python `_DevBuf.__del__` was.
    try:
        ref cpy = Python().cpython()
        var names = _RESIDENT_NAMES.get_or_create_ptr()
        var raw = cpy.lib.get_function[Optional[_ResidentBytes]]("PyCapsule_GetPointer")(
            capsule, names[].capsule_name.as_c_string_slice().unsafe_ptr())
        if not raw:
            cpython_table()[].PyErr_Clear()  # main's form (pass 2, C4)
            return None
        var words = _ResidentWords(unsafe_from_address=Int(raw.value()))
        var free = words[0]
        var handle = words[1]
        words[0] = 0
        if free != 0:
            var pending = cpy.PyErr_FetchTriple()
            var argument = cpy.lib.get_function[PyObjectPtr]("PyLong_FromSsize_t")(handle)
            if argument:
                # both arguments as address words, as words[0] is (no call
                # on main mixes an Int and a PyObjectPtr argument; pass 2 C8)
                var result = cpy.lib.get_function[PyObjectPtr]("PyObject_CallOneArg")(free, Int(argument))
                if result:
                    cpy.lib.get_function[NoneType]("Py_DecRef")(result)
                cpy.lib.get_function[NoneType]("Py_DecRef")(argument)
            cpython_table()[].PyErr_Clear()  # main's form (pass 2, C4)
            cpy.lib.get_function[NoneType]("Py_DecRef")(free)
            cpy.PyErr_Restore(pending[0], pending[1], pending[2])
        external_call["free", NoneType](raw.value())
    except:
        pass
    return None


def resident_capsule_owned(value: PythonObject) raises -> Bool:
    """True when `value` (already known to be an exact capsule) is one of
    these resident capsules: its name is this module's capsule name. Its
    destructor is `_resident_destroy` above (the binding's compiled free; no
    Python function runs), so dropping it is admitted by the owner release."""
    var names = _RESIDENT_NAMES.get_or_create_ptr()
    var valid = cpython_table()[].PyCapsule_IsValid(
        value._obj_ptr, Int(names[].capsule_name.as_c_string_slice().unsafe_ptr()))
    return Int(valid) != 0


def resident_capsule(free: PythonObject, handle: Int) raises -> PythonObject:
    # Two words: a strong reference to the minting binding's free entry and
    # the device buffer id. On failure the buffer is returned at once.
    ref cpy = Python().cpython()
    var raw = external_call["malloc", Optional[_ResidentBytes]](c_size_t(16))
    if not raw:
        _ = py_call(free, handle)
        raise Error("resident device record allocation failed")
    var words = _ResidentWords(unsafe_from_address=Int(raw.value()))
    words[0] = Int(cpy.Py_NewRef(free._obj_ptr))
    words[1] = handle
    var names = _RESIDENT_NAMES.get_or_create_ptr()
    var result = cpy.lib.get_function[PyObjectPtr]("PyCapsule_New")(
        raw.value(), names[].capsule_name.as_c_string_slice().unsafe_ptr(), _resident_destroy)
    if not result:
        cpy.lib.get_function[NoneType]("Py_DecRef")(words[0])
        external_call["free", NoneType](raw.value())
        var failure = cpy.unsafe_get_error()
        _ = py_call(free, handle)
        raise failure^
    return PythonObject(from_owned=result)
