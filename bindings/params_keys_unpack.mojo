# SPDX-License-Identifier: Apache-2.0
# The params family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct ParamsKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k___getattribute__: PythonObject
    var k___mro__: PythonObject
    var k__has_constructor_parameters: PythonObject
    var k__mojolearn_params_reset: PythonObject
    var k_class_name: PythonObject
    var k_constructor: PythonObject
    var k_fields_only: PythonObject
    var k_fitted_fields: PythonObject
    var k_gbdt_init: PythonObject
    var k_get_params: PythonObject
    var k_hooks: PythonObject
    var k_kind: PythonObject
    var k_missing: PythonObject
    var k_missing_error: PythonObject
    var k_mro: PythonObject
    var k_n_repeats: PythonObject
    var k_names: PythonObject
    var k_ownership: PythonObject
    var k_parameter_metadata: PythonObject
    var k_parameters: PythonObject
    var k_pending: PythonObject
    var k_pinned: PythonObject
    var k_plans: PythonObject
    var k_protected: PythonObject
    var k_random_state: PythonObject
    var k_ready: PythonObject
    var k_recurse: PythonObject
    var k_splitter_plans: PythonObject
    var k__get_param_names: PythonObject
    var k___getattr__: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k___getattribute__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k__has_constructor_parameters = PythonObject(None)
        self.k__mojolearn_params_reset = PythonObject(None)
        self.k_class_name = PythonObject(None)
        self.k_constructor = PythonObject(None)
        self.k_fields_only = PythonObject(None)
        self.k_fitted_fields = PythonObject(None)
        self.k_gbdt_init = PythonObject(None)
        self.k_get_params = PythonObject(None)
        self.k_hooks = PythonObject(None)
        self.k_kind = PythonObject(None)
        self.k_missing = PythonObject(None)
        self.k_missing_error = PythonObject(None)
        self.k_mro = PythonObject(None)
        self.k_n_repeats = PythonObject(None)
        self.k_names = PythonObject(None)
        self.k_ownership = PythonObject(None)
        self.k_parameter_metadata = PythonObject(None)
        self.k_parameters = PythonObject(None)
        self.k_pending = PythonObject(None)
        self.k_pinned = PythonObject(None)
        self.k_plans = PythonObject(None)
        self.k_protected = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_ready = PythonObject(None)
        self.k_recurse = PythonObject(None)
        self.k_splitter_plans = PythonObject(None)
        self.k__get_param_names = PythonObject(None)
        self.k___getattr__ = PythonObject(None)


comptime _PARAMS_KEYS = SharedSlot[ParamsKeys, "MojolearnParamsKeysV1"]


def _params_keys_fill(nk: UnsafePointer[ParamsKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k___getattribute__ = py_key("__getattribute__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k__has_constructor_parameters = py_key("_has_constructor_parameters")
    nk[].k__mojolearn_params_reset = py_key("_mojolearn_params_reset")
    nk[].k_class_name = py_key("class_name")
    nk[].k_constructor = py_key("constructor")
    nk[].k_fields_only = py_key("fields_only")
    nk[].k_fitted_fields = py_key("fitted_fields")
    nk[].k_gbdt_init = py_key("gbdt_init")
    nk[].k_get_params = py_key("get_params")
    nk[].k_hooks = py_key("hooks")
    nk[].k_kind = py_key("kind")
    nk[].k_missing = py_key("missing")
    nk[].k_missing_error = py_key("missing_error")
    nk[].k_mro = py_key("mro")
    nk[].k_n_repeats = py_key("n_repeats")
    nk[].k_names = py_key("names")
    nk[].k_ownership = py_key("ownership")
    nk[].k_parameter_metadata = py_key("parameter_metadata")
    nk[].k_parameters = py_key("parameters")
    nk[].k_pending = py_key("pending")
    nk[].k_pinned = py_key("pinned")
    nk[].k_plans = py_key("plans")
    nk[].k_protected = py_key("protected")
    nk[].k_random_state = py_key("random_state")
    nk[].k_ready = py_key("ready")
    nk[].k_recurse = py_key("recurse")
    nk[].k_splitter_plans = py_key("splitter_plans")
    nk[].k__get_param_names = py_key("_get_param_names")
    nk[].k___getattr__ = py_key("__getattr__")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def params_keys() raises -> UnsafePointer[ParamsKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _PARAMS_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _params_keys_fill(nk)
    return nk
