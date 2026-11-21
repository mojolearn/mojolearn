# SPDX-License-Identifier: Apache-2.0
# The pca family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime PCA_OPN_LOAD = 0
comptime PCA_OPN_TRANSFORM = 1
comptime PCA_OPN_INVERSE_TRANSFORM = 2
comptime PCA_OPN_FIT = 3
comptime PCA_OPN_FIT_TRANSFORM = 4
comptime PCA_OPN_SAVE = 5
comptime PCA_OPN_INITIALIZE = 6
comptime PCA_OPN_GET_PARAMS = 7
comptime PCA_OPN_SET_PARAMS = 8
comptime PCA_OPN_IS_FITTED = 9
comptime PCA_OPN_NUMERIC_MODE_USED = 10
comptime PCA_OPN_VENDOR_USED = 11


struct PcaKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_C_CONTIGUOUS: PythonObject
    var k_ComplexWarning: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__HOST_INFERENCE_ONLY: PythonObject
    var k___and__: PythonObject
    var k___del__: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__mojolearn_fitting: PythonObject
    var k__pca_fit_owners: PythonObject
    var k__pca_query_scalar_owners: PythonObject
    var k_archive_npy_dtypes: PythonObject
    var k_array_class: PythonObject
    var k_array_type: PythonObject
    var k_backend_fields: PythonObject
    var k_components_: PythonObject
    var k_copy: PythonObject
    var k_estimator: PythonObject
    var k_explained_variance_: PythonObject
    var k_explained_variance_ratio_: PythonObject
    var k_float32: PythonObject
    var k_float64: PythonObject
    var k_format: PythonObject
    var k_input_copied_: PythonObject
    var k_iterated_power: PythonObject
    var k_mean_: PythonObject
    var k_meta: PythonObject
    var k_module_type: PythonObject
    var k_modules: PythonObject
    var k_n_components: PythonObject
    var k_n_components_: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_oversamples: PythonObject
    var k_n_samples_: PythonObject
    var k_native_bytearray_type: PythonObject
    var k_native_bytes_type: PythonObject
    var k_ndarray: PythonObject
    var k_noise_variance: PythonObject
    var k_noise_variance_: PythonObject
    var k_numeric_mode: PythonObject
    var k_pca_bindings: PythonObject
    var k_pca_cpu_bindings: PythonObject
    var k_pca_class: PythonObject
    var k_pca_dictionary_descriptor: PythonObject
    var k_pca_fit_scalar_context: PythonObject
    var k_pca_fold_factory: PythonObject
    var k_pca_parameters: PythonObject
    var k_pca_state: PythonObject
    var k_power_iteration_normalizer: PythonObject
    var k_random_state: PythonObject
    var k_scalar_types: PythonObject
    var k_singular_values_: PythonObject
    var k_svd_solver: PythonObject
    var k_tol: PythonObject
    var k_vendor: PythonObject
    var k_whiten: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_C_CONTIGUOUS = PythonObject(None)
        self.k_ComplexWarning = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__HOST_INFERENCE_ONLY = PythonObject(None)
        self.k___and__ = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__mojolearn_fitting = PythonObject(None)
        self.k__pca_fit_owners = PythonObject(None)
        self.k__pca_query_scalar_owners = PythonObject(None)
        self.k_archive_npy_dtypes = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_components_ = PythonObject(None)
        self.k_copy = PythonObject(None)
        self.k_estimator = PythonObject(None)
        self.k_explained_variance_ = PythonObject(None)
        self.k_explained_variance_ratio_ = PythonObject(None)
        self.k_float32 = PythonObject(None)
        self.k_float64 = PythonObject(None)
        self.k_format = PythonObject(None)
        self.k_input_copied_ = PythonObject(None)
        self.k_iterated_power = PythonObject(None)
        self.k_mean_ = PythonObject(None)
        self.k_meta = PythonObject(None)
        self.k_module_type = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_n_components = PythonObject(None)
        self.k_n_components_ = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_oversamples = PythonObject(None)
        self.k_n_samples_ = PythonObject(None)
        self.k_native_bytearray_type = PythonObject(None)
        self.k_native_bytes_type = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_noise_variance = PythonObject(None)
        self.k_noise_variance_ = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_pca_bindings = PythonObject(None)
        self.k_pca_cpu_bindings = PythonObject(None)
        self.k_pca_class = PythonObject(None)
        self.k_pca_dictionary_descriptor = PythonObject(None)
        self.k_pca_fit_scalar_context = PythonObject(None)
        self.k_pca_fold_factory = PythonObject(None)
        self.k_pca_parameters = PythonObject(None)
        self.k_pca_state = PythonObject(None)
        self.k_power_iteration_normalizer = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_singular_values_ = PythonObject(None)
        self.k_svd_solver = PythonObject(None)
        self.k_tol = PythonObject(None)
        self.k_vendor = PythonObject(None)
        self.k_whiten = PythonObject(None)


comptime _PCA_KEYS = SharedSlot[PcaKeys, "MojolearnPcaKeysV1"]


def _pca_keys_fill(nk: UnsafePointer[PcaKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_C_CONTIGUOUS = py_key("C_CONTIGUOUS")
    nk[].k_ComplexWarning = py_key("ComplexWarning")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__HOST_INFERENCE_ONLY = py_key("_HOST_INFERENCE_ONLY")
    nk[].k___and__ = py_key("__and__")
    nk[].k___del__ = py_key("__del__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__mojolearn_fitting = py_key("_mojolearn_fitting")
    nk[].k__pca_fit_owners = py_key("_pca_fit_owners")
    nk[].k__pca_query_scalar_owners = py_key("_pca_query_scalar_owners")
    nk[].k_archive_npy_dtypes = py_key("archive_npy_dtypes")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_type = py_key("array_type")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_components_ = py_key("components_")
    nk[].k_copy = py_key("copy")
    nk[].k_estimator = py_key("estimator")
    nk[].k_explained_variance_ = py_key("explained_variance_")
    nk[].k_explained_variance_ratio_ = py_key("explained_variance_ratio_")
    nk[].k_float32 = py_key("float32")
    nk[].k_float64 = py_key("float64")
    nk[].k_format = py_key("format")
    nk[].k_input_copied_ = py_key("input_copied_")
    nk[].k_iterated_power = py_key("iterated_power")
    nk[].k_mean_ = py_key("mean_")
    nk[].k_meta = py_key("meta")
    nk[].k_module_type = py_key("module_type")
    nk[].k_modules = py_key("modules")
    nk[].k_n_components = py_key("n_components")
    nk[].k_n_components_ = py_key("n_components_")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_oversamples = py_key("n_oversamples")
    nk[].k_n_samples_ = py_key("n_samples_")
    nk[].k_native_bytearray_type = py_key("native_bytearray_type")
    nk[].k_native_bytes_type = py_key("native_bytes_type")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_noise_variance = py_key("noise_variance")
    nk[].k_noise_variance_ = py_key("noise_variance_")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_pca_bindings = py_key("pca_bindings")
    nk[].k_pca_cpu_bindings = py_key("pca_cpu_bindings")
    nk[].k_pca_class = py_key("pca_class")
    nk[].k_pca_dictionary_descriptor = py_key("pca_dictionary_descriptor")
    nk[].k_pca_fit_scalar_context = py_key("pca_fit_scalar_context")
    nk[].k_pca_fold_factory = py_key("pca_fold_factory")
    nk[].k_pca_parameters = py_key("pca_parameters")
    nk[].k_pca_state = py_key("pca_state")
    nk[].k_power_iteration_normalizer = py_key("power_iteration_normalizer")
    nk[].k_random_state = py_key("random_state")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_singular_values_ = py_key("singular_values_")
    nk[].k_svd_solver = py_key("svd_solver")
    nk[].k_tol = py_key("tol")
    nk[].k_vendor = py_key("vendor")
    nk[].k_whiten = py_key("whiten")
    var ops = Python.dict()
    py_op_add(ops, "load", PCA_OPN_LOAD)
    py_op_add(ops, "transform", PCA_OPN_TRANSFORM)
    py_op_add(ops, "inverse_transform", PCA_OPN_INVERSE_TRANSFORM)
    py_op_add(ops, "fit", PCA_OPN_FIT)
    py_op_add(ops, "fit_transform", PCA_OPN_FIT_TRANSFORM)
    py_op_add(ops, "save", PCA_OPN_SAVE)
    py_op_add(ops, "initialize", PCA_OPN_INITIALIZE)
    py_op_add(ops, "get_params", PCA_OPN_GET_PARAMS)
    py_op_add(ops, "set_params", PCA_OPN_SET_PARAMS)
    py_op_add(ops, "is_fitted", PCA_OPN_IS_FITTED)
    py_op_add(ops, "numeric_mode_used", PCA_OPN_NUMERIC_MODE_USED)
    py_op_add(ops, "vendor_used", PCA_OPN_VENDOR_USED)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def pca_keys() raises -> UnsafePointer[PcaKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _PCA_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _pca_keys_fill(nk)
    return nk
