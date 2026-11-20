# SPDX-License-Identifier: Apache-2.0
# The forecast family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct ForecastKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k___del__: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k__endog: PythonObject
    var k__numeric_mode: PythonObject
    var k__one: PythonObject
    var k_alias: PythonObject
    var k_alpha: PythonObject
    var k_beta: PythonObject
    var k_coefs: PythonObject
    var k_damped: PythonObject
    var k_decomposition_type: PythonObject
    var k_df_resid: PythonObject
    var k_dist: PythonObject
    var k_endog: PythonObject
    var k_error: PythonObject
    var k_exog: PythonObject
    var k_fittedvalues: PythonObject
    var k_gamma: PythonObject
    var k_initial_smoothed: PythonObject
    var k_intercept: PythonObject
    var k_k_ar: PythonObject
    var k_k_trend: PythonObject
    var k_low_pass: PythonObject
    var k_low_pass_deg: PythonObject
    var k_low_pass_jump: PythonObject
    var k_mean: PythonObject
    var k_model: PythonObject
    var k_ndarray: PythonObject
    var k_neqs: PythonObject
    var k_nobs: PythonObject
    var k_numeric_mode: PythonObject
    var k_o: PythonObject
    var k_p: PythonObject
    var k_params: PythonObject
    var k_period: PythonObject
    var k_phi: PythonObject
    var k_power: PythonObject
    var k_q: PythonObject
    var k_resid: PythonObject
    var k_robust: PythonObject
    var k_scalar_types: PythonObject
    var k_season_length: PythonObject
    var k_seasonal: PythonObject
    var k_seasonal_deg: PythonObject
    var k_seasonal_jump: PythonObject
    var k_sigma_u: PythonObject
    var k_theta: PythonObject
    var k_trend: PythonObject
    var k_trend_deg: PythonObject
    var k_trend_jump: PythonObject
    var k_xseq_forecast_ownership: PythonObject
    var k_xseq_lifecycle_context: PythonObject
    var k_xseq_theta_codes: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k__endog = PythonObject(None)
        self.k__numeric_mode = PythonObject(None)
        self.k__one = PythonObject(None)
        self.k_alias = PythonObject(None)
        self.k_alpha = PythonObject(None)
        self.k_beta = PythonObject(None)
        self.k_coefs = PythonObject(None)
        self.k_damped = PythonObject(None)
        self.k_decomposition_type = PythonObject(None)
        self.k_df_resid = PythonObject(None)
        self.k_dist = PythonObject(None)
        self.k_endog = PythonObject(None)
        self.k_error = PythonObject(None)
        self.k_exog = PythonObject(None)
        self.k_fittedvalues = PythonObject(None)
        self.k_gamma = PythonObject(None)
        self.k_initial_smoothed = PythonObject(None)
        self.k_intercept = PythonObject(None)
        self.k_k_ar = PythonObject(None)
        self.k_k_trend = PythonObject(None)
        self.k_low_pass = PythonObject(None)
        self.k_low_pass_deg = PythonObject(None)
        self.k_low_pass_jump = PythonObject(None)
        self.k_mean = PythonObject(None)
        self.k_model = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_neqs = PythonObject(None)
        self.k_nobs = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_o = PythonObject(None)
        self.k_p = PythonObject(None)
        self.k_params = PythonObject(None)
        self.k_period = PythonObject(None)
        self.k_phi = PythonObject(None)
        self.k_power = PythonObject(None)
        self.k_q = PythonObject(None)
        self.k_resid = PythonObject(None)
        self.k_robust = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_season_length = PythonObject(None)
        self.k_seasonal = PythonObject(None)
        self.k_seasonal_deg = PythonObject(None)
        self.k_seasonal_jump = PythonObject(None)
        self.k_sigma_u = PythonObject(None)
        self.k_theta = PythonObject(None)
        self.k_trend = PythonObject(None)
        self.k_trend_deg = PythonObject(None)
        self.k_trend_jump = PythonObject(None)
        self.k_xseq_forecast_ownership = PythonObject(None)
        self.k_xseq_lifecycle_context = PythonObject(None)
        self.k_xseq_theta_codes = PythonObject(None)


comptime _FORECAST_KEYS = SharedSlot[ForecastKeys, "MojolearnForecastKeysV1"]


def _forecast_keys_fill(nk: UnsafePointer[ForecastKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k___del__ = py_key("__del__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k__endog = py_key("_endog")
    nk[].k__numeric_mode = py_key("_numeric_mode")
    nk[].k__one = py_key("_one")
    nk[].k_alias = py_key("alias")
    nk[].k_alpha = py_key("alpha")
    nk[].k_beta = py_key("beta")
    nk[].k_coefs = py_key("coefs")
    nk[].k_damped = py_key("damped")
    nk[].k_decomposition_type = py_key("decomposition_type")
    nk[].k_df_resid = py_key("df_resid")
    nk[].k_dist = py_key("dist")
    nk[].k_endog = py_key("endog")
    nk[].k_error = py_key("error")
    nk[].k_exog = py_key("exog")
    nk[].k_fittedvalues = py_key("fittedvalues")
    nk[].k_gamma = py_key("gamma")
    nk[].k_initial_smoothed = py_key("initial_smoothed")
    nk[].k_intercept = py_key("intercept")
    nk[].k_k_ar = py_key("k_ar")
    nk[].k_k_trend = py_key("k_trend")
    nk[].k_low_pass = py_key("low_pass")
    nk[].k_low_pass_deg = py_key("low_pass_deg")
    nk[].k_low_pass_jump = py_key("low_pass_jump")
    nk[].k_mean = py_key("mean")
    nk[].k_model = py_key("model")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_neqs = py_key("neqs")
    nk[].k_nobs = py_key("nobs")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_o = py_key("o")
    nk[].k_p = py_key("p")
    nk[].k_params = py_key("params")
    nk[].k_period = py_key("period")
    nk[].k_phi = py_key("phi")
    nk[].k_power = py_key("power")
    nk[].k_q = py_key("q")
    nk[].k_resid = py_key("resid")
    nk[].k_robust = py_key("robust")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_season_length = py_key("season_length")
    nk[].k_seasonal = py_key("seasonal")
    nk[].k_seasonal_deg = py_key("seasonal_deg")
    nk[].k_seasonal_jump = py_key("seasonal_jump")
    nk[].k_sigma_u = py_key("sigma_u")
    nk[].k_theta = py_key("theta")
    nk[].k_trend = py_key("trend")
    nk[].k_trend_deg = py_key("trend_deg")
    nk[].k_trend_jump = py_key("trend_jump")
    nk[].k_xseq_forecast_ownership = py_key("xseq_forecast_ownership")
    nk[].k_xseq_lifecycle_context = py_key("xseq_lifecycle_context")
    nk[].k_xseq_theta_codes = py_key("xseq_theta_codes")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def forecast_keys() raises -> UnsafePointer[ForecastKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _FORECAST_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _forecast_keys_fill(nk)
    return nk
