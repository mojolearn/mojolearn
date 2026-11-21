# SPDX-License-Identifier: Apache-2.0
# The tsa family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime TSA_OPN_KPSS_TEST = 0
comptime TSA_OPN_SELECT_D = 1
comptime TSA_OPN_HW_INITIALIZE = 2
comptime TSA_OPN_HW_FORECAST = 3
comptime TSA_OPN_HW_ATTRIBUTE = 4
comptime TSA_OPN_HW_COMPONENT = 5
comptime TSA_OPN_HW_SAVE = 6
comptime TSA_OPN_HW_FIT = 7
comptime TSA_OPN_HW_PREDICT = 8
comptime TSA_OPN_HW_SCORE = 9
comptime TSA_OPN_ARIMA_INITIALIZE = 10
comptime TSA_OPN_ARIMA_PARAMETER = 11
comptime TSA_OPN_ARIMA_REPR = 12
comptime TSA_OPN_ARIMA_FIT = 13
comptime TSA_OPN_ARIMA_FORECAST = 14
comptime TSA_OPN_ARIMA_PREDICT = 15
comptime TSA_OPN_AUTOARIMA_SEARCH = 16
comptime TSA_OPN_AUTOARIMA_INITIALIZE = 17
comptime TSA_OPN_AUTOARIMA_FORECAST = 18
comptime TSA_OPN_AUTOARIMA_FIT = 19
comptime TSA_OPN_AUTOARIMA_PREDICT = 20
comptime TSA_OPN_HW_VENDOR_USED = 21


struct TsaKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_HostExponentialSmoothing: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__HOST_INFERENCE_ONLY: PythonObject
    var k___del__: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k__comp_layout: PythonObject
    var k__components_len: PythonObject
    var k__comps: PythonObject
    var k__data: PythonObject
    var k__endog: PythonObject
    var k__exog: PythonObject
    var k__fitted: PythonObject
    var k__ic: PythonObject
    var k__ids: PythonObject
    var k__mojolearn_fitted: PythonObject
    var k__reuse: PythonObject
    var k__root_check: PythonObject
    var k__search_maxiter: PythonObject
    var k__time_major: PythonObject
    var k__y: PythonObject
    var k__y_ndim: PythonObject
    var k_aic_: PythonObject
    var k_arima_bindings: PythonObject
    var k_arima_class: PythonObject
    var k_arima_entry_names: PythonObject
    var k_arima_forecast_resident: PythonObject
    var k_arima_methods: PythonObject
    var k_arima_ndiffs_kpss: PythonObject
    var k_arima_numeric_mode: PythonObject
    var k_arima_order_search: PythonObject
    var k_arima_order_search_device: PythonObject
    var k_arima_order_search_fit: PythonObject
    var k_arima_predict_resident: PythonObject
    var k_arima_resident_prepare: PythonObject
    var k_arima_resident_release: PythonObject
    var k_array_class: PythonObject
    var k_array_type: PythonObject
    var k_autoarima_class: PythonObject
    var k_backend_fields: PythonObject
    var k_base_gather_rows_bytes: PythonObject
    var k_base_scatter_rows_bytes: PythonObject
    var k_base_strided_copy_bytes: PythonObject
    var k_batch_size: PythonObject
    var k_batch_size_: PythonObject
    var k_bic_: PythonObject
    var k_complexity_: PythonObject
    var k_components: PythonObject
    var k_criterion_: PythonObject
    var k_d_: PythonObject
    var k_endog: PythonObject
    var k_eps: PythonObject
    var k_fit_executed_flag: PythonObject
    var k_fx_: PythonObject
    var k_hw_bindings: PythonObject
    var k_device: PythonObject
    var k_hw_cpu_bindings: PythonObject
    var k_hw_install_vendor: PythonObject
    var k_hw_cell_family: PythonObject
    var k_hw_cell_mode: PythonObject
    var k_hw_cell_vendor: PythonObject
    var k_hw_cell_module: PythonObject
    var k_hw_class: PythonObject
    var k_hw_host_module: PythonObject
    var k_hw_init_codes: PythonObject
    var k_ic_: PythonObject
    var k_initialization_method: PythonObject
    var k_input_copied_: PythonObject
    var k_integer_types: PythonObject
    var k_k_: PythonObject
    var k_kpss_test: PythonObject
    var k_llf_: PythonObject
    var k_maxiter: PythonObject
    var k_meta: PythonObject
    var k_method: PythonObject
    var k_models: PythonObject
    var k_module_name: PythonObject
    var k_modules: PythonObject
    var k_n: PythonObject
    var k_n_exog_: PythonObject
    var k_n_iter_: PythonObject
    var k_n_obs: PythonObject
    var k_n_obs_: PythonObject
    var k_ndarray: PythonObject
    var k_numeric_dtypes: PythonObject
    var k_numeric_mode: PythonObject
    var k_order: PythonObject
    var k_order_: PythonObject
    var k_output_type: PythonObject
    var k_params_: PythonObject
    var k_retcode_: PythonObject
    var k_scalar_types: PythonObject
    var k_seasonal: PythonObject
    var k_seasonal_order: PythonObject
    var k_seasonal_periods: PythonObject
    var k_select_d: PythonObject
    var k_sse_: PythonObject
    var k_start_periods: PythonObject
    var k_trend: PythonObject
    var k_ts_num: PythonObject
    var k_tsa_bindings: PythonObject
    var k_tsa_owner_context: PythonObject
    var k_verbose: PythonObject
    var k_x0_: PythonObject
    var k_x_: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_HostExponentialSmoothing = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__HOST_INFERENCE_ONLY = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k__comp_layout = PythonObject(None)
        self.k__components_len = PythonObject(None)
        self.k__comps = PythonObject(None)
        self.k__data = PythonObject(None)
        self.k__endog = PythonObject(None)
        self.k__exog = PythonObject(None)
        self.k__fitted = PythonObject(None)
        self.k__ic = PythonObject(None)
        self.k__ids = PythonObject(None)
        self.k__mojolearn_fitted = PythonObject(None)
        self.k__reuse = PythonObject(None)
        self.k__root_check = PythonObject(None)
        self.k__search_maxiter = PythonObject(None)
        self.k__time_major = PythonObject(None)
        self.k__y = PythonObject(None)
        self.k__y_ndim = PythonObject(None)
        self.k_aic_ = PythonObject(None)
        self.k_arima_bindings = PythonObject(None)
        self.k_arima_class = PythonObject(None)
        self.k_arima_entry_names = PythonObject(None)
        self.k_arima_forecast_resident = PythonObject(None)
        self.k_arima_methods = PythonObject(None)
        self.k_arima_ndiffs_kpss = PythonObject(None)
        self.k_arima_numeric_mode = PythonObject(None)
        self.k_arima_order_search = PythonObject(None)
        self.k_arima_order_search_device = PythonObject(None)
        self.k_arima_order_search_fit = PythonObject(None)
        self.k_arima_predict_resident = PythonObject(None)
        self.k_arima_resident_prepare = PythonObject(None)
        self.k_arima_resident_release = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_autoarima_class = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k_base_gather_rows_bytes = PythonObject(None)
        self.k_base_scatter_rows_bytes = PythonObject(None)
        self.k_base_strided_copy_bytes = PythonObject(None)
        self.k_batch_size = PythonObject(None)
        self.k_batch_size_ = PythonObject(None)
        self.k_bic_ = PythonObject(None)
        self.k_complexity_ = PythonObject(None)
        self.k_components = PythonObject(None)
        self.k_criterion_ = PythonObject(None)
        self.k_d_ = PythonObject(None)
        self.k_endog = PythonObject(None)
        self.k_eps = PythonObject(None)
        self.k_fit_executed_flag = PythonObject(None)
        self.k_fx_ = PythonObject(None)
        self.k_hw_bindings = PythonObject(None)
        self.k_device = PythonObject(None)
        self.k_hw_cpu_bindings = PythonObject(None)
        self.k_hw_install_vendor = PythonObject(None)
        self.k_hw_cell_family = PythonObject(None)
        self.k_hw_cell_mode = PythonObject(None)
        self.k_hw_cell_vendor = PythonObject(None)
        self.k_hw_cell_module = PythonObject(None)
        self.k_hw_class = PythonObject(None)
        self.k_hw_host_module = PythonObject(None)
        self.k_hw_init_codes = PythonObject(None)
        self.k_ic_ = PythonObject(None)
        self.k_initialization_method = PythonObject(None)
        self.k_input_copied_ = PythonObject(None)
        self.k_integer_types = PythonObject(None)
        self.k_k_ = PythonObject(None)
        self.k_kpss_test = PythonObject(None)
        self.k_llf_ = PythonObject(None)
        self.k_maxiter = PythonObject(None)
        self.k_meta = PythonObject(None)
        self.k_method = PythonObject(None)
        self.k_models = PythonObject(None)
        self.k_module_name = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_n = PythonObject(None)
        self.k_n_exog_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_n_obs = PythonObject(None)
        self.k_n_obs_ = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numeric_dtypes = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_order = PythonObject(None)
        self.k_order_ = PythonObject(None)
        self.k_output_type = PythonObject(None)
        self.k_params_ = PythonObject(None)
        self.k_retcode_ = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_seasonal = PythonObject(None)
        self.k_seasonal_order = PythonObject(None)
        self.k_seasonal_periods = PythonObject(None)
        self.k_select_d = PythonObject(None)
        self.k_sse_ = PythonObject(None)
        self.k_start_periods = PythonObject(None)
        self.k_trend = PythonObject(None)
        self.k_ts_num = PythonObject(None)
        self.k_tsa_bindings = PythonObject(None)
        self.k_tsa_owner_context = PythonObject(None)
        self.k_verbose = PythonObject(None)
        self.k_x0_ = PythonObject(None)
        self.k_x_ = PythonObject(None)


comptime _TSA_KEYS = SharedSlot[TsaKeys, "MojolearnTsaKeysV1"]


def _tsa_keys_fill(nk: UnsafePointer[TsaKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_HostExponentialSmoothing = py_key("HostExponentialSmoothing")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__HOST_INFERENCE_ONLY = py_key("_HOST_INFERENCE_ONLY")
    nk[].k___del__ = py_key("__del__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k__comp_layout = py_key("_comp_layout")
    nk[].k__components_len = py_key("_components_len")
    nk[].k__comps = py_key("_comps")
    nk[].k__data = py_key("_data")
    nk[].k__endog = py_key("_endog")
    nk[].k__exog = py_key("_exog")
    nk[].k__fitted = py_key("_fitted")
    nk[].k__ic = py_key("_ic")
    nk[].k__ids = py_key("_ids")
    nk[].k__mojolearn_fitted = py_key("_mojolearn_fitted")
    nk[].k__reuse = py_key("_reuse")
    nk[].k__root_check = py_key("_root_check")
    nk[].k__search_maxiter = py_key("_search_maxiter")
    nk[].k__time_major = py_key("_time_major")
    nk[].k__y = py_key("_y")
    nk[].k__y_ndim = py_key("_y_ndim")
    nk[].k_aic_ = py_key("aic_")
    nk[].k_arima_bindings = py_key("arima_bindings")
    nk[].k_arima_class = py_key("arima_class")
    nk[].k_arima_entry_names = py_key("arima_entry_names")
    nk[].k_arima_forecast_resident = py_key("arima_forecast_resident")
    nk[].k_arima_methods = py_key("arima_methods")
    nk[].k_arima_ndiffs_kpss = py_key("arima_ndiffs_kpss")
    nk[].k_arima_numeric_mode = py_key("arima_numeric_mode")
    nk[].k_arima_order_search = py_key("arima_order_search")
    nk[].k_arima_order_search_device = py_key("arima_order_search_device")
    nk[].k_arima_order_search_fit = py_key("arima_order_search_fit")
    nk[].k_arima_predict_resident = py_key("arima_predict_resident")
    nk[].k_arima_resident_prepare = py_key("arima_resident_prepare")
    nk[].k_arima_resident_release = py_key("arima_resident_release")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_type = py_key("array_type")
    nk[].k_autoarima_class = py_key("autoarima_class")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k_base_gather_rows_bytes = py_key("base_gather_rows_bytes")
    nk[].k_base_scatter_rows_bytes = py_key("base_scatter_rows_bytes")
    nk[].k_base_strided_copy_bytes = py_key("base_strided_copy_bytes")
    nk[].k_batch_size = py_key("batch_size")
    nk[].k_batch_size_ = py_key("batch_size_")
    nk[].k_bic_ = py_key("bic_")
    nk[].k_complexity_ = py_key("complexity_")
    nk[].k_components = py_key("components")
    nk[].k_criterion_ = py_key("criterion_")
    nk[].k_d_ = py_key("d_")
    nk[].k_endog = py_key("endog")
    nk[].k_eps = py_key("eps")
    nk[].k_fit_executed_flag = py_key("fit_executed_flag")
    nk[].k_fx_ = py_key("fx_")
    nk[].k_hw_bindings = py_key("hw_bindings")
    nk[].k_device = py_key("device")
    nk[].k_hw_cpu_bindings = py_key("hw_cpu_bindings")
    nk[].k_hw_install_vendor = py_key("hw_install_vendor")
    nk[].k_hw_cell_family = py_key("family")
    nk[].k_hw_cell_mode = py_key("mode")
    nk[].k_hw_cell_vendor = py_key("vendor")
    nk[].k_hw_cell_module = py_key("module")
    nk[].k_hw_class = py_key("hw_class")
    nk[].k_hw_host_module = py_key("hw_host_module")
    nk[].k_hw_init_codes = py_key("hw_init_codes")
    nk[].k_ic_ = py_key("ic_")
    nk[].k_initialization_method = py_key("initialization_method")
    nk[].k_input_copied_ = py_key("input_copied_")
    nk[].k_integer_types = py_key("integer_types")
    nk[].k_k_ = py_key("k_")
    nk[].k_kpss_test = py_key("kpss_test")
    nk[].k_llf_ = py_key("llf_")
    nk[].k_maxiter = py_key("maxiter")
    nk[].k_meta = py_key("meta")
    nk[].k_method = py_key("method")
    nk[].k_models = py_key("models")
    nk[].k_module_name = py_key("module_name")
    nk[].k_modules = py_key("modules")
    nk[].k_n = py_key("n")
    nk[].k_n_exog_ = py_key("n_exog_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_n_obs = py_key("n_obs")
    nk[].k_n_obs_ = py_key("n_obs_")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numeric_dtypes = py_key("numeric_dtypes")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_order = py_key("order")
    nk[].k_order_ = py_key("order_")
    nk[].k_output_type = py_key("output_type")
    nk[].k_params_ = py_key("params_")
    nk[].k_retcode_ = py_key("retcode_")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_seasonal = py_key("seasonal")
    nk[].k_seasonal_order = py_key("seasonal_order")
    nk[].k_seasonal_periods = py_key("seasonal_periods")
    nk[].k_select_d = py_key("select_d")
    nk[].k_sse_ = py_key("sse_")
    nk[].k_start_periods = py_key("start_periods")
    nk[].k_trend = py_key("trend")
    nk[].k_ts_num = py_key("ts_num")
    nk[].k_tsa_bindings = py_key("tsa_bindings")
    nk[].k_tsa_owner_context = py_key("tsa_owner_context")
    nk[].k_verbose = py_key("verbose")
    nk[].k_x0_ = py_key("x0_")
    nk[].k_x_ = py_key("x_")
    var ops = Python.dict()
    py_op_add(ops, "kpss_test", TSA_OPN_KPSS_TEST)
    py_op_add(ops, "select_d", TSA_OPN_SELECT_D)
    py_op_add(ops, "hw_initialize", TSA_OPN_HW_INITIALIZE)
    py_op_add(ops, "hw_forecast", TSA_OPN_HW_FORECAST)
    py_op_add(ops, "hw_attribute", TSA_OPN_HW_ATTRIBUTE)
    py_op_add(ops, "hw_component", TSA_OPN_HW_COMPONENT)
    py_op_add(ops, "hw_save", TSA_OPN_HW_SAVE)
    py_op_add(ops, "hw_fit", TSA_OPN_HW_FIT)
    py_op_add(ops, "hw_predict", TSA_OPN_HW_PREDICT)
    py_op_add(ops, "hw_score", TSA_OPN_HW_SCORE)
    py_op_add(ops, "hw_vendor_used", TSA_OPN_HW_VENDOR_USED)
    py_op_add(ops, "arima_initialize", TSA_OPN_ARIMA_INITIALIZE)
    py_op_add(ops, "arima_parameter", TSA_OPN_ARIMA_PARAMETER)
    py_op_add(ops, "arima_repr", TSA_OPN_ARIMA_REPR)
    py_op_add(ops, "arima_fit", TSA_OPN_ARIMA_FIT)
    py_op_add(ops, "arima_forecast", TSA_OPN_ARIMA_FORECAST)
    py_op_add(ops, "arima_predict", TSA_OPN_ARIMA_PREDICT)
    py_op_add(ops, "autoarima_search", TSA_OPN_AUTOARIMA_SEARCH)
    py_op_add(ops, "autoarima_initialize", TSA_OPN_AUTOARIMA_INITIALIZE)
    py_op_add(ops, "autoarima_forecast", TSA_OPN_AUTOARIMA_FORECAST)
    py_op_add(ops, "autoarima_fit", TSA_OPN_AUTOARIMA_FIT)
    py_op_add(ops, "autoarima_predict", TSA_OPN_AUTOARIMA_PREDICT)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def tsa_keys() raises -> UnsafePointer[TsaKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _TSA_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _tsa_keys_fill(nk)
    return nk
