# SPDX-License-Identifier: Apache-2.0
"""A PyCapsule that owns one META member plan entry (core/ensemble_member.mojo
`meta_plan_mint`; lane mf-meta3, 2026-11-23).

The META kind's admission (providers/python/member_ensemble_boundary_door.mojo)
keeps the nested plans in the process-wide plan store and the capsule in the
owner's pins: every caller keeps a member's pins as long as any copy of its
plan can make a state (the member set's pins, a held state's owner). When
the last reference goes, the native destructor releases the entry
(`meta_plan_release`, typed; no Python function runs). The form of
bindings/native_resident_capsule_pack.mojo, without a Python free entry.
"""
from std.ffi import external_call, c_size_t
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table
from core.ensemble_member import meta_plan_release
from core.shared_slot import SharedSlot

comptime _Words = MutPointer[Int, MutUntrackedOrigin]
comptime _Bytes = MutPointer[UInt8, MutUntrackedOrigin]


struct _MetaPlanNames(Defaultable, Movable):
    var capsule_name: String

    def __init__(out self):
        # PyCapsule keeps this name pointer, not a copy: the DSO-global owner
        # is never reassigned and outlives every capsule.
        self.capsule_name = String("mojolearn.ensemble_member.plan.v1")


comptime _NAMES = SharedSlot[_MetaPlanNames, "MojolearnEnsembleMemberPlanNamesV1"]


def _meta_plan_destroy(capsule: PyObjectPtr) abi("C") -> NoneType:
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
        meta_plan_release(words[0])
        words[0] = 0
        external_call["free", NoneType](raw.value())
    except:
        pass
    return None


def meta_plan_capsule(handle: Int) raises -> PythonObject:
    """UNPACK: the capsule that releases plan entry `handle` when the last
    reference to it goes. On failure the entry is released at once."""
    ref cpy = Python().cpython()
    var raw = external_call["malloc", Optional[_Bytes]](c_size_t(8))
    if not raw:
        meta_plan_release(handle)
        raise Error("MemoryError: meta member plan record allocation failed")
    var words = _Words(unsafe_from_address=Int(raw.value()))
    words[0] = handle
    var names = _NAMES.get_or_create_ptr()
    var result = cpy.lib.get_function[PyObjectPtr]("PyCapsule_New")(
        raw.value(), names[].capsule_name.as_c_string_slice().unsafe_ptr(), _meta_plan_destroy)
    if not result:
        external_call["free", NoneType](raw.value())
        meta_plan_release(handle)
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)
