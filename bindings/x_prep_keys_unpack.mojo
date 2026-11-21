# SPDX-License-Identifier: Apache-2.0
# The x_prep family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct XPrepKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k___class__: PythonObject
    var k___del__: PythonObject
    var k___dict__: PythonObject
    var k___getattribute__: PythonObject
    var k___mro__: PythonObject
    var k__data: PythonObject
    var k__mgr: PythonObject
    var k__mv: PythonObject
    var k_array_class: PythonObject
    var k_array_fields: PythonObject
    var k_base: PythonObject
    var k_blocks: PythonObject
    var k_dtype: PythonObject
    var k_indices: PythonObject
    var k_maxabs_gather_rows: PythonObject
    var k_maxabs_rows_class: PythonObject
    var k_maxabs_rows_owner: PythonObject
    var k_ndarray: PythonObject
    var k_order: PythonObject
    var k_output_context: PythonObject
    var k_scalar_types: PythonObject
    var k_shape: PythonObject
    var k_vendor: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k___class__ = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___getattribute__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k__data = PythonObject(None)
        self.k__mgr = PythonObject(None)
        self.k__mv = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_fields = PythonObject(None)
        self.k_base = PythonObject(None)
        self.k_blocks = PythonObject(None)
        self.k_dtype = PythonObject(None)
        self.k_indices = PythonObject(None)
        self.k_maxabs_gather_rows = PythonObject(None)
        self.k_maxabs_rows_class = PythonObject(None)
        self.k_maxabs_rows_owner = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_order = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_shape = PythonObject(None)
        self.k_vendor = PythonObject(None)


comptime _X_PREP_KEYS = SharedSlot[XPrepKeys, "MojolearnXPrepKeysV1"]


def _x_prep_keys_fill(nk: UnsafePointer[XPrepKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k___class__ = py_key("__class__")
    nk[].k___del__ = py_key("__del__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___getattribute__ = py_key("__getattribute__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k__data = py_key("_data")
    nk[].k__mgr = py_key("_mgr")
    nk[].k__mv = py_key("_mv")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_fields = py_key("array_fields")
    nk[].k_base = py_key("base")
    nk[].k_blocks = py_key("blocks")
    nk[].k_dtype = py_key("dtype")
    nk[].k_indices = py_key("indices")
    nk[].k_maxabs_gather_rows = py_key("maxabs_gather_rows")
    nk[].k_maxabs_rows_class = py_key("maxabs_rows_class")
    nk[].k_maxabs_rows_owner = py_key("maxabs_rows_owner")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_order = py_key("order")
    nk[].k_output_context = py_key("output_context")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_shape = py_key("shape")
    nk[].k_vendor = py_key("vendor")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def x_prep_keys() raises -> UnsafePointer[XPrepKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _X_PREP_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _x_prep_keys_fill(nk)
    return nk
