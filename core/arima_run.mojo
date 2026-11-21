# SPDX-License-Identifier: Apache-2.0
"""RUN of ARIMA's and AutoARIMA's public operations: one call through the
estimator's resolved operations entry (core/device_route.mojo `DeviceRoute`,
slot `entry`) over typed addresses and words. No Python import, no Python
object; the boundary (bindings/tsa_arima_native_door.mojo and the AutoARIMA
doors) admits the buffers, pins their owners and releases the GIL around
each call.

Each function is one request of core/arima_entry_table.mojo; a request has
no independent units to split here (the kernels batch the series and the
candidate orders themselves).
"""
from core.arima_entry_table import (
    ARIMA_OP_FLAGS, ARIMA_OP_VENDOR, ARIMA_OP_MODE, ARIMA_OP_FIT, ARIMA_OP_PREDICT, ARIMA_OP_FORECAST,
    ARIMA_OP_IC, ARIMA_OP_RESIDENT_PREPARE, ARIMA_OP_RESIDENT_RELEASE, ARIMA_OP_RESIDENT_COUNT,
    ARIMA_OP_PREDICT_RESIDENT, ARIMA_OP_FORECAST_RESIDENT, ARIMA_OP_NDIFFS, ARIMA_OP_ORDER_SEARCH,
    ARIMA_OP_ORDER_SEARCH_FIT, ARIMA_OP_ORDER_SEARCH_DEVICE, ARIMA_OP_FAST_SEARCH_MODE, ARIMA_OP_ORDER_CAPS,
    ARIMA_OP_SEARCH_SWITCHES, ARIMA_HAS_FIT, ARIMA_HAS_RESIDENT, ARIMA_HAS_SEARCH, ARIMA_HAS_IC,
    arima_ops_resolve,
)
from core.device_route import DeviceRoute
from core.entry_table import entry_ref_call


def arima_route_resolve(device: Int, family: Int, mode: Int, vendor: Int) raises -> DeviceRoute:
    """Read the operations entry of (`family`, `mode`, `vendor`) once: the
    compiled entry must report the vendor of the key it was published under
    (the tier is admitted by the caller, in the door's words). `flags` holds
    its ARIMA_HAS_* bits; `mode` and `vendor` are the entry's own answers. A
    metadata operation at an estimator's first use; nothing to split."""
    var route = DeviceRoute(device)
    var ops = arima_ops_resolve(family, mode, vendor)
    var got_vendor = entry_ref_call(ops, ARIMA_OP_VENDOR, List[Int](), List[Int](), List[Float64]())
    if got_vendor != vendor:
        raise Error("RuntimeError: ARIMA native vendor differs from selected execution")
    route.mode = entry_ref_call(ops, ARIMA_OP_MODE, List[Int](), List[Int](), List[Float64]())
    route.flags = entry_ref_call(ops, ARIMA_OP_FLAGS, List[Int](), List[Int](), List[Float64]())
    route.vendor = got_vendor
    route.entry = ops
    return route


@always_inline
def arima_route_fits(route: DeviceRoute) -> Bool:
    return (route.flags & ARIMA_HAS_FIT) != 0


@always_inline
def arima_route_resident(route: DeviceRoute) -> Bool:
    return (route.flags & ARIMA_HAS_RESIDENT) != 0


@always_inline
def arima_route_searches(route: DeviceRoute) -> Bool:
    return (route.flags & ARIMA_HAS_SEARCH) != 0


@always_inline
def arima_route_has_ic(route: DeviceRoute) -> Bool:
    return (route.flags & ARIMA_HAS_IC) != 0


def arima_fit_run(route: DeviceRoute, y: Int, exog: Int, params: Int, x: Int, x0: Int, stats: Int, flags: Int,
    ic: Int, batch: Int, n_obs: Int, order: List[Int], k: Int, n_exog: Int, method: Int, max_iterations: Int,
    pen_aic: Float64, pen_bic: Float64) raises -> Int:
    """The fit: `N * batch` float32 at each of `params`, `x`, `x0`, the
    statistics at `stats`, the flags at `flags`, AIC then BIC (`2 * batch`
    float64) at `ic`; returns `N * batch`. `order` is [p, d, q, P, D, Q, s]."""
    var a: List[Int] = [y, exog, params, x, x0, stats, flags, ic]
    var p: List[Int] = [batch, n_obs, order[0], order[1], order[2], order[3], order[4], order[5], order[6], k,
                        n_exog, method, max_iterations]
    var f: List[Float64] = [pen_aic, pen_bic]
    return entry_ref_call(route.entry, ARIMA_OP_FIT, a, p, f)


def arima_predict_run(route: DeviceRoute, y: Int, exog: Int, exog_fut: Int, params: Int, dst: Int, batch: Int,
    n_obs: Int, start: Int, end: Int, order: List[Int], k: Int, n_exog: Int) raises -> Int:
    """Steps `[start, end)` at `dst`, series-major."""
    var a: List[Int] = [y, exog, exog_fut, params, dst]
    var p: List[Int] = [batch, n_obs, start, end, order[0], order[1], order[2], order[3], order[4], order[5],
                        order[6], k, n_exog]
    return entry_ref_call(route.entry, ARIMA_OP_PREDICT, a, p, List[Float64]())


def arima_forecast_run(route: DeviceRoute, y: Int, exog: Int, exog_fut: Int, params: Int, dst: Int, batch: Int,
    n_obs: Int, n_steps: Int, order: List[Int], k: Int, n_exog: Int) raises -> Int:
    """`n_steps` forecast steps per series at `dst`, series-major."""
    var a: List[Int] = [y, exog, exog_fut, params, dst]
    var p: List[Int] = [batch, n_obs, n_steps, order[0], order[1], order[2], order[3], order[4], order[5],
                        order[6], k, n_exog]
    return entry_ref_call(route.entry, ARIMA_OP_FORECAST, a, p, List[Float64]())


def arima_ic_run(route: DeviceRoute, loglike: Int, ic: Int, batch: Int, pen_aic: Float64,
    pen_bic: Float64) raises -> Int:
    """AIC then BIC (`2 * batch` float64 at `ic`) from `batch` float32
    log-likelihoods, by the plain fit's kernel; returns `batch`."""
    var a: List[Int] = [loglike, ic]
    var p: List[Int] = [batch]
    var f: List[Float64] = [pen_aic, pen_bic]
    return entry_ref_call(route.entry, ARIMA_OP_IC, a, p, f)


def arima_resident_prepare_run(route: DeviceRoute, y: Int, exog: Int, params: Int, words: List[Int]) raises -> Int:
    """Upload the fitted state once; `words` is the eleven plan words of
    arima/tsa_arima_resident_run.mojo. Returns the resident handle."""
    var a: List[Int] = [y, exog, params]
    return entry_ref_call(route.entry, ARIMA_OP_RESIDENT_PREPARE, a, words, List[Float64]())


def arima_resident_release_run(route: DeviceRoute, handle: Int) raises:
    var p: List[Int] = [handle]
    _ = entry_ref_call(route.entry, ARIMA_OP_RESIDENT_RELEASE, List[Int](), p, List[Float64]())


def arima_resident_count_run(route: DeviceRoute) raises -> Int:
    """Live resident handles of the route's binding (0 on a CPU binding)."""
    return entry_ref_call(route.entry, ARIMA_OP_RESIDENT_COUNT, List[Int](), List[Int](), List[Float64]())


def arima_resident_query_run(route: DeviceRoute, handle: Int, exog_fut: Int, dst: Int, start: Int, end: Int,
    width: Int, forecast: Bool) raises -> Int:
    """One resident prediction (`[start, end)`) or forecast (`width`
    steps) at `dst`."""
    var a: List[Int] = [exog_fut, dst]
    if forecast:
        var steps: List[Int] = [handle, width]
        return entry_ref_call(route.entry, ARIMA_OP_FORECAST_RESIDENT, a, steps, List[Float64]())
    var p: List[Int] = [handle, start, end]
    return entry_ref_call(route.entry, ARIMA_OP_PREDICT_RESIDENT, a, p, List[Float64]())


def arima_fast_search_mode_run(route: DeviceRoute) raises -> Int:
    return entry_ref_call(route.entry, ARIMA_OP_FAST_SEARCH_MODE, List[Int](), List[Int](), List[Float64]())


def arima_order_caps_run(route: DeviceRoute) raises -> Int:
    return entry_ref_call(route.entry, ARIMA_OP_ORDER_CAPS, List[Int](), List[Int](), List[Float64]())


def arima_search_switches_run(route: DeviceRoute) raises -> Int:
    """ARIMA_SWITCH_* bits of the route's binding, read per search."""
    return entry_ref_call(route.entry, ARIMA_OP_SEARCH_SWITCHES, List[Int](), List[Int](), List[Float64]())


def arima_ndiffs_run(route: DeviceRoute, y: Int, dst: Int, batch: Int, n_obs: Int, d_max: Int) raises -> Int:
    var a: List[Int] = [y, dst]
    var p: List[Int] = [batch, n_obs, d_max]
    return entry_ref_call(route.entry, ARIMA_OP_NDIFFS, a, p, List[Float64]())


def arima_order_search_run(route: DeviceRoute, y: Int, dst: Int, plan: List[Int]) raises -> Int:
    """The grouped likelihood search: `plan` is [batch, n_obs, d, maxiter]
    then one (p, q, k) triple per candidate order. 0: take per-order fits."""
    var a: List[Int] = [y, dst]
    return entry_ref_call(route.entry, ARIMA_OP_ORDER_SEARCH, a, plan, List[Float64]())


def arima_order_search_fit_run(route: DeviceRoute, y: Int, dst: Int, fit_f32: Int, fit_i32: Int,
    plan: List[Int]) raises -> Int:
    """`arima_order_search_run` that also writes each order's fit block."""
    var a: List[Int] = [y, dst, fit_f32, fit_i32]
    return entry_ref_call(route.entry, ARIMA_OP_ORDER_SEARCH_FIT, a, plan, List[Float64]())


def arima_order_search_device_run(route: DeviceRoute, y: Int, dst: Int, best: Int, plan: List[Int],
    penalties: List[Float64]) raises -> Int:
    """The device search: `plan` is [batch, n_obs, d, D, s, maxiter,
    want_ic] then one (p, q, P, Q, k) row per order; `penalties` one
    criterion penalty per order. 0: take per-order fits."""
    var a: List[Int] = [y, dst, best]
    return entry_ref_call(route.entry, ARIMA_OP_ORDER_SEARCH_DEVICE, a, plan, penalties)
