# SPDX-License-Identifier: Apache-2.0
# The mode family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime MODE_OPN_IS_FITTED = 0
comptime MODE_OPN_NUMERIC_MODE_USED = 1
comptime MODE_OPN_VENDOR_USED = 2


struct ModeKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__BINDING: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k___dict__: PythonObject
    var k___get__: PythonObject
    var k___mro__: PythonObject
    var k__bind: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__mojolearn_fitting: PythonObject
    var k__mojolearn_params_reset: PythonObject
    var k_backend_fields: PythonObject
    var k_bind: PythonObject
    var k_binding: PythonObject
    var k_error: PythonObject
    var k_legacy_mode: PythonObject
    var k_mode: PythonObject
    var k_mode_codes: PythonObject
    var k_modes: PythonObject
    var k_plans: PythonObject
    var k_route: PythonObject
    var k_vendor: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__BINDING = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___get__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k__bind = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__mojolearn_fitting = PythonObject(None)
        self.k__mojolearn_params_reset = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_bind = PythonObject(None)
        self.k_binding = PythonObject(None)
        self.k_error = PythonObject(None)
        self.k_legacy_mode = PythonObject(None)
        self.k_mode = PythonObject(None)
        self.k_mode_codes = PythonObject(None)
        self.k_modes = PythonObject(None)
        self.k_plans = PythonObject(None)
        self.k_route = PythonObject(None)
        self.k_vendor = PythonObject(None)


comptime _MODE_KEYS = SharedSlot[ModeKeys, "MojolearnModeKeysV1"]


def _mode_keys_fill(nk: UnsafePointer[ModeKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__BINDING = py_key("_BINDING")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___get__ = py_key("__get__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k__bind = py_key("_bind")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__mojolearn_fitting = py_key("_mojolearn_fitting")
    nk[].k__mojolearn_params_reset = py_key("_mojolearn_params_reset")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_bind = py_key("bind")
    nk[].k_binding = py_key("binding")
    nk[].k_error = py_key("error")
    nk[].k_legacy_mode = py_key("legacy_mode")
    nk[].k_mode = py_key("mode")
    nk[].k_mode_codes = py_key("mode_codes")
    nk[].k_modes = py_key("modes")
    nk[].k_plans = py_key("plans")
    nk[].k_route = py_key("route")
    nk[].k_vendor = py_key("vendor")
    var ops = Python.dict()
    py_op_add(ops, "is_fitted", MODE_OPN_IS_FITTED)
    py_op_add(ops, "numeric_mode_used", MODE_OPN_NUMERIC_MODE_USED)
    py_op_add(ops, "vendor_used", MODE_OPN_VENDOR_USED)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def mode_keys() raises -> UnsafePointer[ModeKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _MODE_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _mode_keys_fill(nk)
    return nk
