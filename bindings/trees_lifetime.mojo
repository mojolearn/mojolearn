# SPDX-License-Identifier: Apache-2.0
# The lifetime of a device-resident tree model snapshot held by a public
# estimator (a forest's resident groves or strict snapshot). It replaces the
# Python `weakref.finalize` of `_forest_protocol._ResidentForest`: the
# snapshot's compiled release entry runs when the last reference to this
# object goes (the estimator dropped, refitted, or its holder replaced).
from std.python import Python, PythonObject
from bindings.native_owner_registry import borrowed_native_owner
from bindings.py_balanced import py_call


struct TreesResident(Movable, Writable):
    var release: PythonObject
    var handle: Int

    def __init__(out self, release: PythonObject, handle: Int):
        self.release = release
        self.handle = handle

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_TreesResident(handle=", self.handle, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.handle < 1:
            return
        try:
            _ = py_call(self.release, PythonObject(self.handle))
        except:
            # Quiet, as the former finalizer: a binding torn down at
            # interpreter exit has nothing left to release.
            pass


def trees_resident_new(release: PythonObject, handle: Int) raises -> PythonObject:
    """A new lifetime owner of `handle`, released through `release`."""
    return PythonObject(alloc=TreesResident(release, handle))


def trees_resident_pins(value: Int, index: Int) abi("C") -> Int:
    # Native registry inspection only; the caller retains the exact owner.
    if value == 0 or index < -1 or index >= 1:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[TreesResident]()
        var result = 1
        if index == 0:
            result = Int(native[].release._obj_ptr)
        _ = owner
        return result
    except:
        return -1
