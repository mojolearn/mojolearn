# SPDX-License-Identifier: Apache-2.0
# The mani family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct ManiKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_ComplexWarning: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__Kc: PythonObject
    var k__fit_X: PythonObject
    var k__iso_v_m_: PythonObject
    var k__k_all: PythonObject
    var k__k_col: PythonObject
    var k__kind: PythonObject
    var k__knn: PythonObject
    var k__mojolearn_after_fit: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__mojolearn_params_reset: PythonObject
    var k__norm: PythonObject
    var k__pw: PythonObject
    var k__shape: PythonObject
    var k__transform_config: PythonObject
    var k__transform_embedding: PythonObject
    var k__transform_mode: PythonObject
    var k__transform_training: PythonObject
    var k_args: PythonObject
    var k_array_class: PythonObject
    var k_array_type: PythonObject
    var k_backend_fields: PythonObject
    var k_dissimilarity_matrix_: PythonObject
    var k_dist_matrix_: PythonObject
    var k_dist_matrix_m_: PythonObject
    var k_eigenvalues_: PythonObject
    var k_eigenvalues_m_: PythonObject
    var k_eigenvectors_m_: PythonObject
    var k_embedding_: PythonObject
    var k_embedding_m_: PythonObject
    var k_family_mani_native_dev_one: PythonObject
    var k_family_mani_numpy_release: PythonObject
    var k_family_mani_pd_kind: PythonObject
    var k_family_mani_scalar_cache: PythonObject
    var k_family_mani_shared: PythonObject
    var k_feature_names_in_: PythonObject
    var k_input_copied_: PythonObject
    var k_kl_divergence_: PythonObject
    var k_learning_rate_: PythonObject
    var k_metric: PythonObject
    var k_module_type: PythonObject
    var k_modules: PythonObject
    var k_n_connected_components_: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_ndarray: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_mode_: PythonObject
    var k_numpy: PythonObject
    var k_output_context: PythonObject
    var k_reconstruction_error_: PythonObject
    var k_scalar_types: PythonObject
    var k_stress_: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_ComplexWarning = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__Kc = PythonObject(None)
        self.k__fit_X = PythonObject(None)
        self.k__iso_v_m_ = PythonObject(None)
        self.k__k_all = PythonObject(None)
        self.k__k_col = PythonObject(None)
        self.k__kind = PythonObject(None)
        self.k__knn = PythonObject(None)
        self.k__mojolearn_after_fit = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__mojolearn_params_reset = PythonObject(None)
        self.k__norm = PythonObject(None)
        self.k__pw = PythonObject(None)
        self.k__shape = PythonObject(None)
        self.k__transform_config = PythonObject(None)
        self.k__transform_embedding = PythonObject(None)
        self.k__transform_mode = PythonObject(None)
        self.k__transform_training = PythonObject(None)
        self.k_args = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_dissimilarity_matrix_ = PythonObject(None)
        self.k_dist_matrix_ = PythonObject(None)
        self.k_dist_matrix_m_ = PythonObject(None)
        self.k_eigenvalues_ = PythonObject(None)
        self.k_eigenvalues_m_ = PythonObject(None)
        self.k_eigenvectors_m_ = PythonObject(None)
        self.k_embedding_ = PythonObject(None)
        self.k_embedding_m_ = PythonObject(None)
        self.k_family_mani_native_dev_one = PythonObject(None)
        self.k_family_mani_numpy_release = PythonObject(None)
        self.k_family_mani_pd_kind = PythonObject(None)
        self.k_family_mani_scalar_cache = PythonObject(None)
        self.k_family_mani_shared = PythonObject(None)
        self.k_feature_names_in_ = PythonObject(None)
        self.k_input_copied_ = PythonObject(None)
        self.k_kl_divergence_ = PythonObject(None)
        self.k_learning_rate_ = PythonObject(None)
        self.k_metric = PythonObject(None)
        self.k_module_type = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_n_connected_components_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_mode_ = PythonObject(None)
        self.k_numpy = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_reconstruction_error_ = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_stress_ = PythonObject(None)


comptime _MANI_KEYS = SharedSlot[ManiKeys, "MojolearnManiKeysV1"]


def _mani_keys_fill(nk: UnsafePointer[ManiKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_ComplexWarning = py_key("ComplexWarning")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__Kc = py_key("_Kc")
    nk[].k__fit_X = py_key("_fit_X")
    nk[].k__iso_v_m_ = py_key("_iso_v_m_")
    nk[].k__k_all = py_key("_k_all")
    nk[].k__k_col = py_key("_k_col")
    nk[].k__kind = py_key("_kind")
    nk[].k__knn = py_key("_knn")
    nk[].k__mojolearn_after_fit = py_key("_mojolearn_after_fit")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__mojolearn_params_reset = py_key("_mojolearn_params_reset")
    nk[].k__norm = py_key("_norm")
    nk[].k__pw = py_key("_pw")
    nk[].k__shape = py_key("_shape")
    nk[].k__transform_config = py_key("_transform_config")
    nk[].k__transform_embedding = py_key("_transform_embedding")
    nk[].k__transform_mode = py_key("_transform_mode")
    nk[].k__transform_training = py_key("_transform_training")
    nk[].k_args = py_key("args")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_type = py_key("array_type")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_dissimilarity_matrix_ = py_key("dissimilarity_matrix_")
    nk[].k_dist_matrix_ = py_key("dist_matrix_")
    nk[].k_dist_matrix_m_ = py_key("dist_matrix_m_")
    nk[].k_eigenvalues_ = py_key("eigenvalues_")
    nk[].k_eigenvalues_m_ = py_key("eigenvalues_m_")
    nk[].k_eigenvectors_m_ = py_key("eigenvectors_m_")
    nk[].k_embedding_ = py_key("embedding_")
    nk[].k_embedding_m_ = py_key("embedding_m_")
    nk[].k_family_mani_native_dev_one = py_key("family_mani_native_dev_one")
    nk[].k_family_mani_numpy_release = py_key("family_mani_numpy_release")
    nk[].k_family_mani_pd_kind = py_key("family_mani_pd_kind")
    nk[].k_family_mani_scalar_cache = py_key("family_mani_scalar_cache")
    nk[].k_family_mani_shared = py_key("family_mani_shared")
    nk[].k_feature_names_in_ = py_key("feature_names_in_")
    nk[].k_input_copied_ = py_key("input_copied_")
    nk[].k_kl_divergence_ = py_key("kl_divergence_")
    nk[].k_learning_rate_ = py_key("learning_rate_")
    nk[].k_metric = py_key("metric")
    nk[].k_module_type = py_key("module_type")
    nk[].k_modules = py_key("modules")
    nk[].k_n_connected_components_ = py_key("n_connected_components_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_mode_ = py_key("numeric_mode_")
    nk[].k_numpy = py_key("numpy")
    nk[].k_output_context = py_key("output_context")
    nk[].k_reconstruction_error_ = py_key("reconstruction_error_")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_stress_ = py_key("stress_")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def mani_keys() raises -> UnsafePointer[ManiKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _MANI_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _mani_keys_fill(nk)
    return nk
