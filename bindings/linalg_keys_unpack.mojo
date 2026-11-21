# SPDX-License-Identifier: Apache-2.0
# The linalg family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime LINALG_OPN_RANDOMIZED_SVD = 0
comptime LINALG_OPN_SOLVE = 1
comptime LINALG_OPN_LSTSQ = 2
comptime LINALG_OPN_LU_FACTOR = 3
comptime LINALG_OPN_LU_SOLVE = 4


struct LinalgKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_INT15_MAX_K: PythonObject
    var k_INT15_PROFILE_VERSION: PythonObject
    var k_LOWBIT_PROFILE_VERSION: PythonObject
    var k_L_: PythonObject
    var k_PROFILE: PythonObject
    var k_PROFILE_VERSION: PythonObject
    var k_PROFILE_INT15: PythonObject
    var k__BUILD_SCRIPT: PythonObject
    var k__CODE_MODE: PythonObject
    var k__CPU_ONLY: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__EXPECTED_PROFILE_ENV: PythonObject
    var k__EXPERIMENT_PROFILES: PythonObject
    var k__SVD_NULL_RTOL_MODULE: PythonObject
    var k___file__: PythonObject
    var k__binding_cache: PythonObject
    var k__logdet: PythonObject
    var k__mode_binding: PythonObject
    var k__mode_cache: PythonObject
    var k__mojolearn_after_fit: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__native_provider: PythonObject
    var k__profile_cache: PythonObject
    var k_backend_fields: PythonObject
    var k_binary: PythonObject
    var k_certification_source: PythonObject
    var k_certified_shapes: PythonObject
    var k_certified_vendors: PythonObject
    var k_closed_form: PythonObject
    var k_confidence_interval: PythonObject
    var k_contract: PythonObject
    var k_count_greater: PythonObject
    var k_count_less: PythonObject
    var k_decomp_modules: PythonObject
    var k_distribution: PythonObject
    var k_dtype: PythonObject
    var k_experiment_source: PythonObject
    var k_family_chol_mode_codes: PythonObject
    var k_family_chol_par_class: PythonObject
    var k_family_lin_result_schemas: PythonObject
    var k_family_lin_state: PythonObject
    var k_family_rs_codes: PythonObject
    var k_identity_claimed: PythonObject
    var k_identity_required: PythonObject
    var k_info_: PythonObject
    var k_integral: PythonObject
    var k_jitter: PythonObject
    var k_jitter_: PythonObject
    var k_kernel: PythonObject
    var k_lu_device_buffer: PythonObject
    var k_lu_pair_class: PythonObject
    var k_mean: PythonObject
    var k_n_: PythonObject
    var k_nb_: PythonObject
    var k_ndarray: PythonObject
    var k_null_distribution: PythonObject
    var k_numeric_mode: PythonObject
    var k_ops: PythonObject
    var k_oracle: PythonObject
    var k_order_high: PythonObject
    var k_order_low: PythonObject
    var k_output_context: PythonObject
    var k_point_estimate: PythonObject
    var k_profile: PythonObject
    var k_profile_version: PythonObject
    var k_pvalue: PythonObject
    var k_qualification: PythonObject
    var k_sorted_distribution: PythonObject
    var k_standard_error: PythonObject
    var k_statistic: PythonObject
    var k_volume: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_INT15_MAX_K = PythonObject(None)
        self.k_INT15_PROFILE_VERSION = PythonObject(None)
        self.k_LOWBIT_PROFILE_VERSION = PythonObject(None)
        self.k_L_ = PythonObject(None)
        self.k_PROFILE = PythonObject(None)
        self.k_PROFILE_VERSION = PythonObject(None)
        self.k_PROFILE_INT15 = PythonObject(None)
        self.k__BUILD_SCRIPT = PythonObject(None)
        self.k__CODE_MODE = PythonObject(None)
        self.k__CPU_ONLY = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__EXPECTED_PROFILE_ENV = PythonObject(None)
        self.k__EXPERIMENT_PROFILES = PythonObject(None)
        self.k__SVD_NULL_RTOL_MODULE = PythonObject(None)
        self.k___file__ = PythonObject(None)
        self.k__binding_cache = PythonObject(None)
        self.k__logdet = PythonObject(None)
        self.k__mode_binding = PythonObject(None)
        self.k__mode_cache = PythonObject(None)
        self.k__mojolearn_after_fit = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__native_provider = PythonObject(None)
        self.k__profile_cache = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_binary = PythonObject(None)
        self.k_certification_source = PythonObject(None)
        self.k_certified_shapes = PythonObject(None)
        self.k_certified_vendors = PythonObject(None)
        self.k_closed_form = PythonObject(None)
        self.k_confidence_interval = PythonObject(None)
        self.k_contract = PythonObject(None)
        self.k_count_greater = PythonObject(None)
        self.k_count_less = PythonObject(None)
        self.k_decomp_modules = PythonObject(None)
        self.k_distribution = PythonObject(None)
        self.k_dtype = PythonObject(None)
        self.k_experiment_source = PythonObject(None)
        self.k_family_chol_mode_codes = PythonObject(None)
        self.k_family_chol_par_class = PythonObject(None)
        self.k_family_lin_result_schemas = PythonObject(None)
        self.k_family_lin_state = PythonObject(None)
        self.k_family_rs_codes = PythonObject(None)
        self.k_identity_claimed = PythonObject(None)
        self.k_identity_required = PythonObject(None)
        self.k_info_ = PythonObject(None)
        self.k_integral = PythonObject(None)
        self.k_jitter = PythonObject(None)
        self.k_jitter_ = PythonObject(None)
        self.k_kernel = PythonObject(None)
        self.k_lu_device_buffer = PythonObject(None)
        self.k_lu_pair_class = PythonObject(None)
        self.k_mean = PythonObject(None)
        self.k_n_ = PythonObject(None)
        self.k_nb_ = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_null_distribution = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_ops = PythonObject(None)
        self.k_oracle = PythonObject(None)
        self.k_order_high = PythonObject(None)
        self.k_order_low = PythonObject(None)
        self.k_output_context = PythonObject(None)
        self.k_point_estimate = PythonObject(None)
        self.k_profile = PythonObject(None)
        self.k_profile_version = PythonObject(None)
        self.k_pvalue = PythonObject(None)
        self.k_qualification = PythonObject(None)
        self.k_sorted_distribution = PythonObject(None)
        self.k_standard_error = PythonObject(None)
        self.k_statistic = PythonObject(None)
        self.k_volume = PythonObject(None)


# V2 (lane dw-decomp, 2026-11-21): `k__native_provider_binding` removed (the door's Python-entry
# fallback that compared it is deleted), so the layout changed.
comptime _LINALG_KEYS = SharedSlot[LinalgKeys, "MojolearnLinalgKeysV2"]


def _linalg_keys_fill(nk: UnsafePointer[LinalgKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_INT15_MAX_K = py_key("INT15_MAX_K")
    nk[].k_INT15_PROFILE_VERSION = py_key("INT15_PROFILE_VERSION")
    nk[].k_LOWBIT_PROFILE_VERSION = py_key("LOWBIT_PROFILE_VERSION")
    nk[].k_L_ = py_key("L_")
    nk[].k_PROFILE = py_key("PROFILE")
    nk[].k_PROFILE_VERSION = py_key("PROFILE_VERSION")
    nk[].k_PROFILE_INT15 = py_key("PROFILE_INT15")
    nk[].k__BUILD_SCRIPT = py_key("_BUILD_SCRIPT")
    nk[].k__CODE_MODE = py_key("_CODE_MODE")
    nk[].k__CPU_ONLY = py_key("_CPU_ONLY")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__EXPECTED_PROFILE_ENV = py_key("_EXPECTED_PROFILE_ENV")
    nk[].k__EXPERIMENT_PROFILES = py_key("_EXPERIMENT_PROFILES")
    nk[].k__SVD_NULL_RTOL_MODULE = py_key("_SVD_NULL_RTOL_MODULE")
    nk[].k___file__ = py_key("__file__")
    nk[].k__binding_cache = py_key("_binding_cache")
    nk[].k__logdet = py_key("_logdet")
    nk[].k__mode_binding = py_key("_mode_binding")
    nk[].k__mode_cache = py_key("_mode_cache")
    nk[].k__mojolearn_after_fit = py_key("_mojolearn_after_fit")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__native_provider = py_key("_native_provider")
    nk[].k__profile_cache = py_key("_profile_cache")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_binary = py_key("binary")
    nk[].k_certification_source = py_key("certification_source")
    nk[].k_certified_shapes = py_key("certified_shapes")
    nk[].k_certified_vendors = py_key("certified_vendors")
    nk[].k_closed_form = py_key("closed_form")
    nk[].k_confidence_interval = py_key("confidence_interval")
    nk[].k_contract = py_key("contract")
    nk[].k_count_greater = py_key("count_greater")
    nk[].k_count_less = py_key("count_less")
    nk[].k_decomp_modules = py_key("decomp_modules")
    nk[].k_distribution = py_key("distribution")
    nk[].k_dtype = py_key("dtype")
    nk[].k_experiment_source = py_key("experiment_source")
    nk[].k_family_chol_mode_codes = py_key("family_chol_mode_codes")
    nk[].k_family_chol_par_class = py_key("family_chol_par_class")
    nk[].k_family_lin_result_schemas = py_key("family_lin_result_schemas")
    nk[].k_family_lin_state = py_key("family_lin_state")
    nk[].k_family_rs_codes = py_key("family_rs_codes")
    nk[].k_identity_claimed = py_key("identity_claimed")
    nk[].k_identity_required = py_key("identity_required")
    nk[].k_info_ = py_key("info_")
    nk[].k_integral = py_key("integral")
    nk[].k_jitter = py_key("jitter")
    nk[].k_jitter_ = py_key("jitter_")
    nk[].k_kernel = py_key("kernel")
    nk[].k_lu_device_buffer = py_key("lu_device_buffer")
    nk[].k_lu_pair_class = py_key("lu_pair_class")
    nk[].k_mean = py_key("mean")
    nk[].k_n_ = py_key("n_")
    nk[].k_nb_ = py_key("nb_")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_null_distribution = py_key("null_distribution")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_ops = py_key("ops")
    nk[].k_oracle = py_key("oracle")
    nk[].k_order_high = py_key("order_high")
    nk[].k_order_low = py_key("order_low")
    nk[].k_output_context = py_key("output_context")
    nk[].k_point_estimate = py_key("point_estimate")
    nk[].k_profile = py_key("profile")
    nk[].k_profile_version = py_key("profile_version")
    nk[].k_pvalue = py_key("pvalue")
    nk[].k_qualification = py_key("qualification")
    nk[].k_sorted_distribution = py_key("sorted_distribution")
    nk[].k_standard_error = py_key("standard_error")
    nk[].k_statistic = py_key("statistic")
    nk[].k_volume = py_key("volume")
    var ops = Python.dict()
    py_op_add(ops, "randomized_svd", LINALG_OPN_RANDOMIZED_SVD)
    py_op_add(ops, "solve", LINALG_OPN_SOLVE)
    py_op_add(ops, "lstsq", LINALG_OPN_LSTSQ)
    py_op_add(ops, "lu_factor", LINALG_OPN_LU_FACTOR)
    py_op_add(ops, "lu_solve", LINALG_OPN_LU_SOLVE)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def linalg_keys() raises -> UnsafePointer[LinalgKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _LINALG_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _linalg_keys_fill(nk)
    return nk
