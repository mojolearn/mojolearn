# SPDX-License-Identifier: Apache-2.0
# The lowbit family's held name keys (bindings/py_keys.mojo has the form and
# its cost rule). Written by lane bw2-linalg in the form of the boundary
# codemod of lane fx-boundary-core; a name the family starts to read is added
# in the three places below (field, None, fill), and the slot name keeps its
# V<n> until a wheel that holds the old layout could meet this one in a
# process (bump `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add


struct LowbitKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_bits: PythonObject
    var k_codes: PythonObject
    var k_exponents: PythonObject
    var k_family_low_bf16: PythonObject
    var k_family_low_int8: PythonObject
    var k_shape: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_bits = PythonObject(None)
        self.k_codes = PythonObject(None)
        self.k_exponents = PythonObject(None)
        self.k_family_low_bf16 = PythonObject(None)
        self.k_family_low_int8 = PythonObject(None)
        self.k_shape = PythonObject(None)


comptime _LOWBIT_KEYS = SharedSlot[LowbitKeys, "MojolearnLowbitKeysV1"]


def _lowbit_keys_fill(nk: UnsafePointer[LowbitKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_bits = py_key("bits")
    nk[].k_codes = py_key("codes")
    nk[].k_exponents = py_key("exponents")
    nk[].k_family_low_bf16 = py_key("family_low_bf16")
    nk[].k_family_low_int8 = py_key("family_low_int8")
    nk[].k_shape = py_key("shape")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def lowbit_keys() raises -> UnsafePointer[LowbitKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _LOWBIT_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _lowbit_keys_fill(nk)
    return nk
