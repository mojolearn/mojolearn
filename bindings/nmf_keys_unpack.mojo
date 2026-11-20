# SPDX-License-Identifier: Apache-2.0
# The nmf family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct NmfKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_H: PythonObject
    var k_W: PythonObject
    var k___and__: PythonObject
    var k_alpha_H: PythonObject
    var k_alpha_W: PythonObject
    var k_array_class: PythonObject
    var k_array_storage_slots: PythonObject
    var k_beta_loss: PythonObject
    var k_components_: PythonObject
    var k_init: PythonObject
    var k_l1_ratio: PythonObject
    var k_max_iter: PythonObject
    var k_n_components: PythonObject
    var k_n_components_: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_ndarray: PythonObject
    var k_nmf_bindings: PythonObject
    var k_output_context: PythonObject
    var k_random_state: PythonObject
    var k_reconstruction_err_: PythonObject
    var k_shuffle: PythonObject
    var k_solver: PythonObject
    var k_tol: PythonObject
    var k_vendor: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_H = PythonObject(None)
        self.k_W = PythonObject(None)
        self.k___and__ = PythonObject(None)
        self.k_alpha_H = PythonObject(None)
        self.k_alpha_W = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_storage_slots = PythonObject(None)
        self.k_beta_loss = PythonObject(None)
        self.k_components_ = PythonObject(None)
        self.k_init = PythonObject(None)
        self.k_l1_ratio = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_n_components = PythonObject(None)
        self.k_n_components_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_nmf_bindings = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_reconstruction_err_ = PythonObject(None)
        self.k_shuffle = PythonObject(None)
        self.k_solver = PythonObject(None)
        self.k_tol = PythonObject(None)
        self.k_vendor = PythonObject(None)


comptime _NMF_KEYS = SharedSlot[NmfKeys, "MojolearnNmfKeysV1"]


def _nmf_keys_fill(nk: UnsafePointer[NmfKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_H = py_key("H")
    nk[].k_W = py_key("W")
    nk[].k___and__ = py_key("__and__")
    nk[].k_alpha_H = py_key("alpha_H")
    nk[].k_alpha_W = py_key("alpha_W")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_storage_slots = py_key("array_storage_slots")
    nk[].k_beta_loss = py_key("beta_loss")
    nk[].k_components_ = py_key("components_")
    nk[].k_init = py_key("init")
    nk[].k_l1_ratio = py_key("l1_ratio")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_n_components = py_key("n_components")
    nk[].k_n_components_ = py_key("n_components_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_nmf_bindings = py_key("nmf_bindings")
    nk[].k_output_context = py_key("output_context")
    nk[].k_random_state = py_key("random_state")
    nk[].k_reconstruction_err_ = py_key("reconstruction_err_")
    nk[].k_shuffle = py_key("shuffle")
    nk[].k_solver = py_key("solver")
    nk[].k_tol = py_key("tol")
    nk[].k_vendor = py_key("vendor")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def nmf_keys() raises -> UnsafePointer[NmfKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _NMF_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _nmf_keys_fill(nk)
    return nk
