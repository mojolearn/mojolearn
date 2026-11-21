# SPDX-License-Identifier: Apache-2.0
# The dict family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime DICT_OPN_DL_FIT = 0
comptime DICT_OPN_DL_FIT_TRANSFORM = 1
comptime DICT_OPN_MBDL_FIT = 2
comptime DICT_OPN_MBDL_FIT_TRANSFORM = 3
comptime DICT_OPN_CODING_TRANSFORM = 4
comptime DICT_OPN_CODING_INVERSE = 5
comptime DICT_OPN_CODER_FIT = 6
comptime DICT_OPN_CODER_TRANSFORM = 7
comptime DICT_OPN_CODER_FIT_TRANSFORM = 8
comptime DICT_OPN_CODER_INVERSE = 9
comptime DICT_OPN_SPCA_FIT = 10
comptime DICT_OPN_SPCA_FIT_TRANSFORM = 11
comptime DICT_OPN_SPCA_TRANSFORM = 12
comptime DICT_OPN_SPCA_INVERSE = 13
comptime DICT_OPN_SPARSE_ENCODE = 14
comptime DICT_OPN_DEVICE_SET = 15
comptime DICT_OPN_VENDOR_USED = 16


struct DictKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_U_init: PythonObject
    var k_V_init: PythonObject
    var k___mro__: PythonObject
    var k_alpha: PythonObject
    var k_batch_size: PythonObject
    var k_callback: PythonObject
    var k_code_init: PythonObject
    var k_components_: PythonObject
    var k_dict_init: PythonObject
    var k_dictionary: PythonObject
    var k_error_: PythonObject
    var k_fit_algorithm: PythonObject
    var k_max_iter: PythonObject
    var k_max_no_improvement: PythonObject
    var k_mean_: PythonObject
    var k_method: PythonObject
    var k_minibatch_sparse_pca_class: PythonObject
    var k_n_components: PythonObject
    var k_n_components_: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_n_steps_: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_mode_: PythonObject
    var k_output_context: PythonObject
    var k_positive_code: PythonObject
    var k_positive_dict: PythonObject
    var k_random_state: PythonObject
    var k_ridge_alpha: PythonObject
    var k_shuffle: PythonObject
    var k_split_sign: PythonObject
    var k_tol: PythonObject
    var k_transform_algorithm: PythonObject
    var k_transform_alpha: PythonObject
    var k_transform_max_iter: PythonObject
    var k_transform_n_nonzero_coefs: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_U_init = PythonObject(None)
        self.k_V_init = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k_alpha = PythonObject(None)
        self.k_batch_size = PythonObject(None)
        self.k_callback = PythonObject(None)
        self.k_code_init = PythonObject(None)
        self.k_components_ = PythonObject(None)
        self.k_dict_init = PythonObject(None)
        self.k_dictionary = PythonObject(None)
        self.k_error_ = PythonObject(None)
        self.k_fit_algorithm = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_max_no_improvement = PythonObject(None)
        self.k_mean_ = PythonObject(None)
        self.k_method = PythonObject(None)
        self.k_minibatch_sparse_pca_class = PythonObject(None)
        self.k_n_components = PythonObject(None)
        self.k_n_components_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_n_steps_ = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_mode_ = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_positive_code = PythonObject(None)
        self.k_positive_dict = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_ridge_alpha = PythonObject(None)
        self.k_shuffle = PythonObject(None)
        self.k_split_sign = PythonObject(None)
        self.k_tol = PythonObject(None)
        self.k_transform_algorithm = PythonObject(None)
        self.k_transform_alpha = PythonObject(None)
        self.k_transform_max_iter = PythonObject(None)
        self.k_transform_n_nonzero_coefs = PythonObject(None)


comptime _DICT_KEYS = SharedSlot[DictKeys, "MojolearnDictKeysV1"]


def _dict_keys_fill(nk: UnsafePointer[DictKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_U_init = py_key("U_init")
    nk[].k_V_init = py_key("V_init")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k_alpha = py_key("alpha")
    nk[].k_batch_size = py_key("batch_size")
    nk[].k_callback = py_key("callback")
    nk[].k_code_init = py_key("code_init")
    nk[].k_components_ = py_key("components_")
    nk[].k_dict_init = py_key("dict_init")
    nk[].k_dictionary = py_key("dictionary")
    nk[].k_error_ = py_key("error_")
    nk[].k_fit_algorithm = py_key("fit_algorithm")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_max_no_improvement = py_key("max_no_improvement")
    nk[].k_mean_ = py_key("mean_")
    nk[].k_method = py_key("method")
    nk[].k_minibatch_sparse_pca_class = py_key("minibatch_sparse_pca_class")
    nk[].k_n_components = py_key("n_components")
    nk[].k_n_components_ = py_key("n_components_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_n_steps_ = py_key("n_steps_")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_mode_ = py_key("numeric_mode_")
    nk[].k_output_context = py_key("output_context")
    nk[].k_positive_code = py_key("positive_code")
    nk[].k_positive_dict = py_key("positive_dict")
    nk[].k_random_state = py_key("random_state")
    nk[].k_ridge_alpha = py_key("ridge_alpha")
    nk[].k_shuffle = py_key("shuffle")
    nk[].k_split_sign = py_key("split_sign")
    nk[].k_tol = py_key("tol")
    nk[].k_transform_algorithm = py_key("transform_algorithm")
    nk[].k_transform_alpha = py_key("transform_alpha")
    nk[].k_transform_max_iter = py_key("transform_max_iter")
    nk[].k_transform_n_nonzero_coefs = py_key("transform_n_nonzero_coefs")
    var ops = Python.dict()
    py_op_add(ops, "dl_fit", DICT_OPN_DL_FIT)
    py_op_add(ops, "dl_fit_transform", DICT_OPN_DL_FIT_TRANSFORM)
    py_op_add(ops, "mbdl_fit", DICT_OPN_MBDL_FIT)
    py_op_add(ops, "mbdl_fit_transform", DICT_OPN_MBDL_FIT_TRANSFORM)
    py_op_add(ops, "coding_transform", DICT_OPN_CODING_TRANSFORM)
    py_op_add(ops, "coding_inverse", DICT_OPN_CODING_INVERSE)
    py_op_add(ops, "coder_fit", DICT_OPN_CODER_FIT)
    py_op_add(ops, "coder_transform", DICT_OPN_CODER_TRANSFORM)
    py_op_add(ops, "coder_fit_transform", DICT_OPN_CODER_FIT_TRANSFORM)
    py_op_add(ops, "coder_inverse", DICT_OPN_CODER_INVERSE)
    py_op_add(ops, "spca_fit", DICT_OPN_SPCA_FIT)
    py_op_add(ops, "spca_fit_transform", DICT_OPN_SPCA_FIT_TRANSFORM)
    py_op_add(ops, "spca_transform", DICT_OPN_SPCA_TRANSFORM)
    py_op_add(ops, "spca_inverse", DICT_OPN_SPCA_INVERSE)
    py_op_add(ops, "sparse_encode", DICT_OPN_SPARSE_ENCODE)
    py_op_add(ops, "device_set", DICT_OPN_DEVICE_SET)
    py_op_add(ops, "vendor_used", DICT_OPN_VENDOR_USED)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def dict_keys() raises -> UnsafePointer[DictKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _DICT_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _dict_keys_fill(nk)
    return nk
