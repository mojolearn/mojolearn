# SPDX-License-Identifier: Apache-2.0
# The robust family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct RobustKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_ComplexWarning: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k_modules: PythonObject
    var k_ndarray: PythonObject
    var k_numpy: PythonObject
    var k_scalar_types: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_ComplexWarning = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numpy = PythonObject(None)
        self.k_scalar_types = PythonObject(None)


comptime _ROBUST_KEYS = SharedSlot[RobustKeys, "MojolearnRobustKeysV1"]


def _robust_keys_fill(nk: UnsafePointer[RobustKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_ComplexWarning = py_key("ComplexWarning")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k_modules = py_key("modules")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numpy = py_key("numpy")
    nk[].k_scalar_types = py_key("scalar_types")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def robust_keys() raises -> UnsafePointer[RobustKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _ROBUST_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _robust_keys_fill(nk)
    return nk
