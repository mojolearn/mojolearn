# SPDX-License-Identifier: Apache-2.0
"""A PyCapsule that owns one splitter spec of the spec store
(bindings/msel_split_spec_run.mojo `msel_spec_mint`; lane mf-split5,
2026-11-23).

An x_linear CV member's admission (providers/python/member_xlinear_boundary_door.mojo)
keeps its splitter's spec in the store, its handle in the member's words and
this capsule in the owner's pins: every caller keeps a member's pins as long
as any copy of its plan can make a state. When the last reference goes, the
native destructor releases the spec (`msel_spec_release`, typed; no Python
function runs). The form of bindings/meta_plan_capsule_pack.mojo.
"""
from std.ffi import external_call, c_size_t
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.msel_split_spec_run import msel_spec_release
from core.shared_slot import SharedSlot

comptime _Words = MutPointer[Int, MutUntrackedOrigin]
comptime _Bytes = MutPointer[UInt8, MutUntrackedOrigin]


struct _SpecNames(Defaultable, Movable):
    var capsule_name: String

    def __init__(out self):
        # PyCapsule keeps this name pointer, not a copy: the DSO-global owner
        # is never reassigned and outlives every capsule.
        self.capsule_name = String("mojolearn.msel_split.spec.v1")


comptime _NAMES = SharedSlot[_SpecNames, "MojolearnMselSplitSpecNamesV1"]


def _spec_destroy(capsule: PyObjectPtr) abi("C") -> NoneType:
    # Native destructor: one typed release, no Python call. A pending
    # exception of the code that dropped the pins is untouched.
    try:
        ref cpy = Python().cpython()
        var names = _NAMES.get_or_create_ptr()
        var raw = cpy.lib.get_function[Optional[_Bytes]]("PyCapsule_GetPointer")(
            capsule, names[].capsule_name.as_c_string_slice().unsafe_ptr())
        if not raw:
            cpython_table()[].PyErr_Clear()
            return None
        var words = _Words(unsafe_from_address=Int(raw.value()))
        msel_spec_release(words[0])
        words[0] = 0
        external_call["free", NoneType](raw.value())
    except:
        pass
    return None


def msel_spec_capsule(handle: Int) raises -> PythonObject:
    """UNPACK: the capsule that releases spec `handle` when the last
    reference to it goes. On failure the spec is released at once."""
    ref cpy = Python().cpython()
    var raw = external_call["malloc", Optional[_Bytes]](c_size_t(8))
    if not raw:
        msel_spec_release(handle)
        raise Error("MemoryError: splitter spec record allocation failed")
    var words = _Words(unsafe_from_address=Int(raw.value()))
    words[0] = handle
    var names = _NAMES.get_or_create_ptr()
    var result = cpy.lib.get_function[PyObjectPtr]("PyCapsule_New")(
        raw.value(), names[].capsule_name.as_c_string_slice().unsafe_ptr(), _spec_destroy)
    if not result:
        external_call["free", NoneType](raw.value())
        msel_spec_release(handle)
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)
