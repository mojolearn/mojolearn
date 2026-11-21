# SPDX-License-Identifier: Apache-2.0
"""The device column of the ARIMA operations table
(core/arima_entry_table.mojo): the typed entry `_mojolearn_arima` publishes
at initialization. Each operation is the body of the former Python export of
the same name with its argument words read from the request spans (no Python
object, no Python list); the caller releases the GIL around the call.

One request is one fit, forecast, query or grouped search; it has no
independent units to split here (the kernels below batch the series and the
candidate orders themselves). The loops below walk the candidate orders of a
search plan, never series data.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.arima_entry_table import (
    ARIMA_OP_FLAGS, ARIMA_OP_VENDOR, ARIMA_OP_MODE, ARIMA_OP_FIT, ARIMA_OP_PREDICT, ARIMA_OP_FORECAST,
    ARIMA_OP_IC, ARIMA_OP_RESIDENT_PREPARE, ARIMA_OP_RESIDENT_RELEASE, ARIMA_OP_RESIDENT_COUNT,
    ARIMA_OP_PREDICT_RESIDENT, ARIMA_OP_FORECAST_RESIDENT, ARIMA_OP_NDIFFS, ARIMA_OP_ORDER_SEARCH,
    ARIMA_OP_ORDER_SEARCH_FIT, ARIMA_OP_ORDER_SEARCH_DEVICE, ARIMA_OP_FAST_SEARCH_MODE, ARIMA_OP_ORDER_CAPS,
    ARIMA_OP_SEARCH_SWITCHES, ARIMA_HAS_FIT, ARIMA_HAS_RESIDENT, ARIMA_HAS_SEARCH, ARIMA_HAS_IC,
    ARIMA_SWITCH_ORDER_BATCH, ARIMA_SWITCH_SEARCH_REUSE, ARIMA_TABLE_ARIMA,
    arima_ops_check_spans, arima_ops_publish,
)
from core.hostptr import f32_ptr, i32_ptr
from core.identity_trace import IdentityTrace
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from arima.estimator import (
    arima_fit_ptr_host,
    arima_ic_from_loglike_ptr_host,
    arima_forecast_ptr_host,
    arima_predict_ptr_host,
)
from arima.impl.fast_arima_quality import ndiffs_kpss
from arima.impl.fast_order_search import (
    order_search_caps, order_search_device, order_search_loglike,
    order_search_fit, ARIMA_FAST_SEARCH_REUSE,
    fast_search_mode,
)
from arima.impl.fast_order_state import ARIMA_ORDER_BATCH
from arima.impl.tsa.arima_common import ARIMAOrder
from arima.mojolearn_arima_run import _f32_ptr, _i32_ptr
from arima.resident_state import (
    arima_resident_prepare,
    arima_resident_release,
    arima_resident_count,
    arima_predict_resident,
)

comptime _VENDOR_CODE = (1 if COMPILED_VENDOR == "metal" else 2 if COMPILED_VENDOR == "cuda" else 3 if COMPILED_VENDOR == "hip" else -1)


def _plain_orders(p: MutPointer[Int, MutUntrackedOrigin], np: Int, d: Int) -> List[ARIMAOrder]:
    """The (p, q, k) triples after the four fixed words, as non-seasonal
    orders of differencing `d`."""
    var orders = List[ARIMAOrder]()
    for i in range((np - 4) // 3):  # small-loop(grid: candidate p, q, k orders of the search plan): plan entries, not series data
        orders.append(ARIMAOrder(p[4 + 3 * i], d, p[4 + 3 * i + 1], 0, 0, 0, 0, p[4 + 3 * i + 2], 0))
    return orders^


def arima_ops_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        arima_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == ARIMA_OP_FLAGS:
            r[] = ARIMA_HAS_FIT | ARIMA_HAS_RESIDENT | ARIMA_HAS_SEARCH | ARIMA_HAS_IC
        elif operation == ARIMA_OP_VENDOR:
            r[] = _VENDOR_CODE
        elif operation == ARIMA_OP_MODE:
            r[] = Int(GLOBAL_NUMERIC_MODE)
        elif operation == ARIMA_OP_FIT:
            # exog (a[1]) must be a live address even when n_exog == 0.
            _ = _f32_ptr(a[1])
            r[] = arima_fit_ptr_host(
                _f32_ptr(a[0]), a[1], _f32_ptr(a[2]), _f32_ptr(a[3]), _f32_ptr(a[4]), _f32_ptr(a[5]), _i32_ptr(a[6]),
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12],
                a[7], f[0], f[1],
            )
        elif operation == ARIMA_OP_PREDICT:
            _ = _f32_ptr(a[1])
            _ = _f32_ptr(a[2])
            r[] = arima_predict_ptr_host(
                _f32_ptr(a[0]), a[1], a[2], _f32_ptr(a[3]), _f32_ptr(a[4]), p[0], p[1], p[2], p[3],
                p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12],
            )
        elif operation == ARIMA_OP_FORECAST:
            _ = _f32_ptr(a[1])
            _ = _f32_ptr(a[2])
            r[] = arima_forecast_ptr_host(
                _f32_ptr(a[0]), a[1], a[2], _f32_ptr(a[3]), _f32_ptr(a[4]), p[0], p[1], p[2],
                p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
            )
        elif operation == ARIMA_OP_IC:
            r[] = arima_ic_from_loglike_ptr_host(_f32_ptr(a[0]), a[1], p[0], f[0], f[1])
        elif operation == ARIMA_OP_RESIDENT_PREPARE:
            _ = _f32_ptr(a[1])
            r[] = arima_resident_prepare(
                _f32_ptr(a[0]), a[1], _f32_ptr(a[2]), p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
            )
        elif operation == ARIMA_OP_RESIDENT_RELEASE:
            arima_resident_release(p[0])
            r[] = 0
        elif operation == ARIMA_OP_RESIDENT_COUNT:
            r[] = arima_resident_count()
        elif operation == ARIMA_OP_PREDICT_RESIDENT:
            _ = _f32_ptr(a[0])
            r[] = arima_predict_resident(p[0], a[0], _f32_ptr(a[1]), p[1], p[2], False, "arima_predict")
        elif operation == ARIMA_OP_FORECAST_RESIDENT:
            if p[1] < 1:
                raise Error("arima_forecast: n_steps must be >= 1 (n_steps=" + String(p[1]) + ")")
            _ = _f32_ptr(a[0])
            r[] = arima_predict_resident(p[0], a[0], _f32_ptr(a[1]), 0, p[1], True, "arima_forecast")
        elif operation == ARIMA_OP_NDIFFS:
            r[] = ndiffs_kpss(f32_ptr(a[0]), i32_ptr(a[1]), p[0], p[1], p[2])
        elif operation == ARIMA_OP_ORDER_SEARCH:
            var orders = _plain_orders(p, np, p[2])
            r[] = order_search_loglike(f32_ptr(a[0]), f32_ptr(a[1]), orders, p[0], p[1], p[3])
        elif operation == ARIMA_OP_ORDER_SEARCH_FIT:
            var orders = _plain_orders(p, np, p[2])
            r[] = order_search_fit(f32_ptr(a[0]), f32_ptr(a[1]), orders, p[0], p[1], p[3], a[2], a[3])
        elif operation == ARIMA_OP_ORDER_SEARCH_DEVICE:
            # Each order's period is `s` when P + D + Q > 0, else 0.
            var d = p[2]
            var seasonal_d = p[3]
            var s = p[4]
            var want_ic = p[6] != 0
            var orders = List[ARIMAOrder]()
            var pen = List[Float32]()
            for i in range((np - 7) // 5):  # small-loop(grid: candidate seasonal orders of the search plan): plan entries, not series data
                var sp = p[7 + 5 * i + 2]
                var sq = p[7 + 5 * i + 3]
                var period = s if (sp + seasonal_d + sq) > 0 else 0
                orders.append(ARIMAOrder(p[7 + 5 * i], d, p[7 + 5 * i + 1], sp, seasonal_d, sq, period, p[7 + 5 * i + 4], 0))
            if want_ic:
                if nf != len(orders):
                    raise Error("arima_order_search_device: one penalty per order")
                for i in range(len(orders)):  # small-loop(orders: one criterion penalty per candidate order): plan parameters, not series data
                    pen.append(Float32(f[i]))
            r[] = order_search_device(f32_ptr(a[0]), f32_ptr(a[1]), i32_ptr(a[2]), orders, pen, p[0], p[1], p[5], want_ic)
        elif operation == ARIMA_OP_FAST_SEARCH_MODE:
            # 0 with an identity trace on.
            var trace = IdentityTrace()
            r[] = 0 if trace.enabled else fast_search_mode()
        elif operation == ARIMA_OP_ORDER_CAPS:
            # With a trace on only bit 2: the search runs as per-order fits,
            # the criterion stays the build's.
            var trace = IdentityTrace()
            r[] = (order_search_caps() & 4) if trace.enabled else order_search_caps()
        elif operation == ARIMA_OP_SEARCH_SWITCHES:
            var trace = IdentityTrace()
            var word = 0
            if ARIMA_ORDER_BATCH and not trace.enabled:
                word |= ARIMA_SWITCH_ORDER_BATCH
            if ARIMA_FAST_SEARCH_REUSE and not trace.enabled:
                word |= ARIMA_SWITCH_SEARCH_REUSE
            r[] = word
        else:
            raise Error("arima ops: unknown operation " + String(operation))
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ArimaOpsDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(arima_ops_device_v1)


comptime _OPS_SLOT = "MojolearnArimaOpsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnArimaOpsDeviceFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime ARIMA_OPS_DEVICE_API = SharedSlot[ArimaOpsDeviceAPI, _OPS_SLOT]


def publish_arima_ops_device() raises:
    """Publish this image's table; called once, after the module is ready."""
    var api = ARIMA_OPS_DEVICE_API.get_or_create_ptr()
    arima_ops_publish[ARIMA_TABLE_ARIMA](Int(GLOBAL_NUMERIC_MODE), _VENDOR_CODE, Int(api[].entries.unsafe_ptr()),
        arima_ops_device_v1)
