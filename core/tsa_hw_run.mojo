# SPDX-License-Identifier: Apache-2.0
"""RUN of ExponentialSmoothing's public operations: one call through the
estimator's resolved operations entry (core/device_route.mojo `DeviceRoute`, slot `entry`) over typed
addresses and words. No Python import, no Python object; the boundary
(bindings/tsa_hw_native_door.mojo) admits the buffers, pins their owners and
releases the GIL around each call.

Each function is one request of core/tsa_entry_table.mojo; a request has no
independent units to split here (the kernels batch the series themselves).
"""
from core.entry_table import entry_ref_call
from core.tsa_entry_table import (
    TSA_OP_FLAGS, TSA_OP_VENDOR, TSA_OP_MODE, TSA_HAS_FIT, TSA_HAS_RESIDENT, tsa_ops_resolve,
    TSA_OP_HW_FIT, TSA_OP_HW_FORECAST, TSA_OP_HW_PREDICT, TSA_OP_HW_RESIDENT_PREPARE,
    TSA_OP_HW_RESIDENT_RELEASE, TSA_OP_HW_RESIDENT_COUNT, TSA_OP_HW_QUERY_RESIDENT,
    TSA_OP_KPSS_TEST, TSA_OP_SELECT_D,
)
from core.device_route import DeviceRoute


def tsa_hw_route_resolve(device: Int, family: Int, mode: Int, vendor: Int) raises -> DeviceRoute:
    """Read the operations entry of (`family`, `mode`, `vendor`) once and
    admit it: the compiled entry must report the vendor and tier of the key
    it was published under. `flags` holds its TSA_HAS_* bits. A metadata
    operation at an estimator's first use; nothing to split."""
    var route = DeviceRoute(device)
    var ops = tsa_ops_resolve(family, mode, vendor)
    var got_vendor = entry_ref_call(ops, TSA_OP_VENDOR, List[Int](), List[Int](), List[Float64]())
    var got_mode = entry_ref_call(ops, TSA_OP_MODE, List[Int](), List[Int](), List[Float64]())
    if got_vendor != vendor:
        raise Error("RuntimeError: Holt-Winters native vendor differs from selected execution")
    if got_mode != mode:
        raise Error("RuntimeError: Holt-Winters native mode differs from selected execution")
    route.flags = entry_ref_call(ops, TSA_OP_FLAGS, List[Int](), List[Int](), List[Float64]())
    route.mode = got_mode
    route.vendor = got_vendor
    route.entry = ops
    return route


@always_inline
def tsa_hw_route_fits(route: DeviceRoute) -> Bool:
    return (route.flags & TSA_HAS_FIT) != 0


@always_inline
def tsa_hw_route_resident(route: DeviceRoute) -> Bool:
    return (route.flags & TSA_HAS_RESIDENT) != 0


def tsa_hw_fit_run(route: DeviceRoute, data: Int, comps: Int, stats: Int, flags: Int, n: Int, batch: Int,
    frequency: Int, start_periods: Int, init_method: Int, seasonal: Int, eps: Float64) raises -> Int:
    """The fit: writes `3 * components_len` float32 at `comps`, the
    per-series statistics at `stats` and the flags at `flags`; returns
    `components_len`."""
    var a: List[Int] = [data, comps, stats, flags]
    var p: List[Int] = [n, batch, frequency, start_periods, init_method, seasonal]
    var f: List[Float64] = [eps]
    return entry_ref_call(route.entry, TSA_OP_HW_FIT, a, p, f)


def tsa_hw_forecast_run(route: DeviceRoute, comps: Int, dst: Int, n: Int, batch: Int, frequency: Int,
    h: Int, seasonal: Int) raises -> Int:
    """`h * batch` forecast float32 at `dst`, time-major."""
    var a: List[Int] = [comps, dst]
    var p: List[Int] = [n, batch, frequency, h, seasonal]
    return entry_ref_call(route.entry, TSA_OP_HW_FORECAST, a, p, List[Float64]())


def tsa_hw_predict_run(route: DeviceRoute, comps: Int, dst: Int, n: Int, batch: Int, frequency: Int,
    start: Int, end: Int, seasonal: Int) raises -> Int:
    """The in-sample steps `[start, end)` at `dst`, time-major."""
    var a: List[Int] = [comps, dst]
    var p: List[Int] = [n, batch, frequency, start, end, seasonal]
    return entry_ref_call(route.entry, TSA_OP_HW_PREDICT, a, p, List[Float64]())


def tsa_hw_resident_prepare_run(route: DeviceRoute, comps: Int, n: Int, batch: Int, frequency: Int) raises -> Int:
    """Upload the fitted components once; returns the resident handle."""
    var a: List[Int] = [comps]
    var p: List[Int] = [n, batch, frequency]
    return entry_ref_call(route.entry, TSA_OP_HW_RESIDENT_PREPARE, a, p, List[Float64]())


def tsa_hw_resident_release_run(route: DeviceRoute, handle: Int) raises:
    var p: List[Int] = [handle]
    _ = entry_ref_call(route.entry, TSA_OP_HW_RESIDENT_RELEASE, List[Int](), p, List[Float64]())


def tsa_hw_resident_count_run(route: DeviceRoute) raises -> Int:
    """Live resident handles of the route's binding (0 on a CPU binding)."""
    return entry_ref_call(route.entry, TSA_OP_HW_RESIDENT_COUNT, List[Int](), List[Int](), List[Float64]())


def tsa_hw_query_resident_run(route: DeviceRoute, handle: Int, dst: Int, start: Int, end: Int,
    seasonal: Int) raises -> Int:
    """Steps `[start, end)` over the resident components at `dst`: the
    in-sample steps, then the forecast steps, time-major."""
    var a: List[Int] = [dst]
    var p: List[Int] = [handle, start, end, seasonal]
    return entry_ref_call(route.entry, TSA_OP_HW_QUERY_RESIDENT, a, p, List[Float64]())


def tsa_kpss_run(route: DeviceRoute, y: Int, flags: Int, stat: Int, batch: Int, n_obs: Int, d: Int,
    seasonal_d: Int, period: Int, pval_threshold: Float64) raises -> Int:
    """`kpss_test`: `batch` uint8 flags (1 stationary) at `flags` and
    `batch` float32 statistics at `stat` from the series-major float32 `y`."""
    var a: List[Int] = [y, flags, stat]
    var p: List[Int] = [batch, n_obs, d, seasonal_d, period]
    var f: List[Float64] = [pval_threshold]
    return entry_ref_call(route.entry, TSA_OP_KPSS_TEST, a, p, f)


def tsa_kpss_run(route: DeviceRoute, y: Int, flags: Int, stat: Int, batch: Int, n_obs: Int, d: Int,
    seasonal_d: Int, period: Int, pval_threshold: Float64, scope_block: Int, layout: Int) raises -> Int:
    """With the door's `resident()` scope block (lane gd-resident-2; 0: the
    plain request) and the series layout (1: the caller's C order, read
    from the GPU image's resident slot)."""
    if scope_block == 0:
        return tsa_kpss_run(route, y, flags, stat, batch, n_obs, d, seasonal_d, period, pval_threshold)
    var a: List[Int] = [y, flags, stat]
    var p: List[Int] = [batch, n_obs, d, seasonal_d, period, scope_block, layout]
    var f: List[Float64] = [pval_threshold]
    return entry_ref_call(route.entry, TSA_OP_KPSS_TEST, a, p, f)


def tsa_select_d_run(route: DeviceRoute, y: Int, dst: Int, batch: Int, n_obs: Int, seasonal_d: Int,
    period: Int, d_max: Int, pval_threshold: Float64) raises -> Int:
    """`select_d`: `batch` int32 differencing orders at `dst`."""
    var a: List[Int] = [y, dst]
    var p: List[Int] = [batch, n_obs, seasonal_d, period, d_max]
    var f: List[Float64] = [pval_threshold]
    return entry_ref_call(route.entry, TSA_OP_SELECT_D, a, p, f)


def tsa_select_d_run(route: DeviceRoute, y: Int, dst: Int, batch: Int, n_obs: Int, seasonal_d: Int,
    period: Int, d_max: Int, pval_threshold: Float64, scope_block: Int, layout: Int) raises -> Int:
    """`tsa_kpss_run`'s scope block and layout words for `select_d`."""
    if scope_block == 0:
        return tsa_select_d_run(route, y, dst, batch, n_obs, seasonal_d, period, d_max, pval_threshold)
    var a: List[Int] = [y, dst]
    var p: List[Int] = [batch, n_obs, seasonal_d, period, d_max, scope_block, layout]
    var f: List[Float64] = [pval_threshold]
    return entry_ref_call(route.entry, TSA_OP_SELECT_D, a, p, f)
