# SPDX-License-Identifier: Apache-2.0
"""RUN of ExponentialSmoothing's public operations: one call through the
estimator's resolved operations entry (core/tsa_route.mojo) over typed
addresses and words. No Python import, no Python object; the boundary
(bindings/tsa_hw_native_door.mojo) admits the buffers, pins their owners and
releases the GIL around each call.

Each function is one request of core/tsa_entry_table.mojo; a request has no
independent units to split here (the kernels batch the series themselves).
"""
from core.entry_table import entry_ref_call
from core.tsa_entry_table import (
    TSA_OP_HW_FIT, TSA_OP_HW_FORECAST, TSA_OP_HW_PREDICT, TSA_OP_HW_RESIDENT_PREPARE,
    TSA_OP_HW_RESIDENT_RELEASE, TSA_OP_HW_RESIDENT_COUNT, TSA_OP_HW_QUERY_RESIDENT,
)
from core.tsa_route import TsaHwRoute


def tsa_hw_fit_run(route: TsaHwRoute, data: Int, comps: Int, stats: Int, flags: Int, n: Int, batch: Int,
    frequency: Int, start_periods: Int, init_method: Int, seasonal: Int, eps: Float64) raises -> Int:
    """The fit: writes `3 * components_len` float32 at `comps`, the
    per-series statistics at `stats` and the flags at `flags`; returns
    `components_len`."""
    var a: List[Int] = [data, comps, stats, flags]
    var p: List[Int] = [n, batch, frequency, start_periods, init_method, seasonal]
    var f: List[Float64] = [eps]
    return entry_ref_call(route.ops, TSA_OP_HW_FIT, a, p, f)


def tsa_hw_forecast_run(route: TsaHwRoute, comps: Int, dst: Int, n: Int, batch: Int, frequency: Int,
    h: Int, seasonal: Int) raises -> Int:
    """`h * batch` forecast float32 at `dst`, time-major."""
    var a: List[Int] = [comps, dst]
    var p: List[Int] = [n, batch, frequency, h, seasonal]
    return entry_ref_call(route.ops, TSA_OP_HW_FORECAST, a, p, List[Float64]())


def tsa_hw_predict_run(route: TsaHwRoute, comps: Int, dst: Int, n: Int, batch: Int, frequency: Int,
    start: Int, end: Int, seasonal: Int) raises -> Int:
    """The in-sample steps `[start, end)` at `dst`, time-major."""
    var a: List[Int] = [comps, dst]
    var p: List[Int] = [n, batch, frequency, start, end, seasonal]
    return entry_ref_call(route.ops, TSA_OP_HW_PREDICT, a, p, List[Float64]())


def tsa_hw_resident_prepare_run(route: TsaHwRoute, comps: Int, n: Int, batch: Int, frequency: Int) raises -> Int:
    """Upload the fitted components once; returns the resident handle."""
    var a: List[Int] = [comps]
    var p: List[Int] = [n, batch, frequency]
    return entry_ref_call(route.ops, TSA_OP_HW_RESIDENT_PREPARE, a, p, List[Float64]())


def tsa_hw_resident_release_run(route: TsaHwRoute, handle: Int) raises:
    var p: List[Int] = [handle]
    _ = entry_ref_call(route.ops, TSA_OP_HW_RESIDENT_RELEASE, List[Int](), p, List[Float64]())


def tsa_hw_resident_count_run(route: TsaHwRoute) raises -> Int:
    """Live resident handles of the route's binding (0 on a CPU binding)."""
    return entry_ref_call(route.ops, TSA_OP_HW_RESIDENT_COUNT, List[Int](), List[Int](), List[Float64]())


def tsa_hw_query_resident_run(route: TsaHwRoute, handle: Int, dst: Int, start: Int, end: Int,
    seasonal: Int) raises -> Int:
    """Steps `[start, end)` over the resident components at `dst`: the
    in-sample steps, then the forecast steps, time-major."""
    var a: List[Int] = [dst]
    var p: List[Int] = [handle, start, end, seasonal]
    return entry_ref_call(route.ops, TSA_OP_HW_QUERY_RESIDENT, a, p, List[Float64]())
