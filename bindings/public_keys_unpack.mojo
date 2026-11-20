# SPDX-License-Identifier: Apache-2.0
# The public family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct PublicKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k___dict__: PythonObject
    var k___getattr__: PythonObject
    var k___getattribute__: PythonObject
    var k___mojolearn_internal__: PythonObject
    var k___mojolearn_user_boundary__: PythonObject
    var k___name__: PythonObject
    var k__mojolearn_after_fit: PythonObject
    var k__mojolearn_fitting: PythonObject
    var k_arguments: PythonObject
    var k_array_class: PythonObject
    var k_attribute_name: PythonObject
    var k_availability: PythonObject
    var k_bindings: PythonObject
    var k_check: PythonObject
    var k_classes_: PythonObject
    var k_code: PythonObject
    var k_condition: PythonObject
    var k_context: PythonObject
    var k_descriptor: PythonObject
    var k_entry: PythonObject
    var k_fn: PythonObject
    var k_function: PythonObject
    var k_hooks: PythonObject
    var k_member: PythonObject
    var k_missing: PythonObject
    var k_mro: PythonObject
    var k_n_features_in_: PythonObject
    var k_native_property_recipes: PythonObject
    var k_ndarray: PythonObject
    var k_operation: PythonObject
    var k_plans: PythonObject
    var k_properties: PythonObject
    var k_svm_owner: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___getattr__ = PythonObject(None)
        self.k___getattribute__ = PythonObject(None)
        self.k___mojolearn_internal__ = PythonObject(None)
        self.k___mojolearn_user_boundary__ = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k__mojolearn_after_fit = PythonObject(None)
        self.k__mojolearn_fitting = PythonObject(None)
        self.k_arguments = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_attribute_name = PythonObject(None)
        self.k_availability = PythonObject(None)
        self.k_bindings = PythonObject(None)
        self.k_check = PythonObject(None)
        self.k_classes_ = PythonObject(None)
        self.k_code = PythonObject(None)
        self.k_condition = PythonObject(None)
        self.k_context = PythonObject(None)
        self.k_descriptor = PythonObject(None)
        self.k_entry = PythonObject(None)
        self.k_fn = PythonObject(None)
        self.k_function = PythonObject(None)
        self.k_hooks = PythonObject(None)
        self.k_member = PythonObject(None)
        self.k_missing = PythonObject(None)
        self.k_mro = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_native_property_recipes = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_operation = PythonObject(None)
        self.k_plans = PythonObject(None)
        self.k_properties = PythonObject(None)
        self.k_svm_owner = PythonObject(None)


comptime _PUBLIC_KEYS = SharedSlot[PublicKeys, "MojolearnPublicKeysV1"]


def _public_keys_fill(nk: UnsafePointer[PublicKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___getattr__ = py_key("__getattr__")
    nk[].k___getattribute__ = py_key("__getattribute__")
    nk[].k___mojolearn_internal__ = py_key("__mojolearn_internal__")
    nk[].k___mojolearn_user_boundary__ = py_key("__mojolearn_user_boundary__")
    nk[].k___name__ = py_key("__name__")
    nk[].k__mojolearn_after_fit = py_key("_mojolearn_after_fit")
    nk[].k__mojolearn_fitting = py_key("_mojolearn_fitting")
    nk[].k_arguments = py_key("arguments")
    nk[].k_array_class = py_key("array_class")
    nk[].k_attribute_name = py_key("attribute_name")
    nk[].k_availability = py_key("availability")
    nk[].k_bindings = py_key("bindings")
    nk[].k_check = py_key("check")
    nk[].k_classes_ = py_key("classes_")
    nk[].k_code = py_key("code")
    nk[].k_condition = py_key("condition")
    nk[].k_context = py_key("context")
    nk[].k_descriptor = py_key("descriptor")
    nk[].k_entry = py_key("entry")
    nk[].k_fn = py_key("fn")
    nk[].k_function = py_key("function")
    nk[].k_hooks = py_key("hooks")
    nk[].k_member = py_key("member")
    nk[].k_missing = py_key("missing")
    nk[].k_mro = py_key("mro")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_native_property_recipes = py_key("native_property_recipes")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_operation = py_key("operation")
    nk[].k_plans = py_key("plans")
    nk[].k_properties = py_key("properties")
    nk[].k_svm_owner = py_key("svm_owner")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def public_keys() raises -> UnsafePointer[PublicKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _PUBLIC_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _public_keys_fill(nk)
    return nk
