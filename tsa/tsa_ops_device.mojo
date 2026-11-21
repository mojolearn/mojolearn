# SPDX-License-Identifier: Apache-2.0
"""The device column of the Holt-Winters operations table
(core/tsa_entry_table.mojo): the typed entry `_mojolearn_tsa` publishes at
initialization. Each operation is the body of the former Python export of
the same name with its argument words read from the request spans (no Python
object, no Python list); the caller releases the GIL around the call.

One request is one fit, forecast or query; it has no independent units to
split here (the kernels below batch the series themselves).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.hostptr import f32_ptr, i32_ptr
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.tsa_entry_table import (
    TSA_OP_FLAGS, TSA_OP_VENDOR, TSA_OP_MODE, TSA_OP_HW_FIT, TSA_OP_HW_FORECAST, TSA_OP_HW_PREDICT,
    TSA_OP_HW_RESIDENT_PREPARE, TSA_OP_HW_RESIDENT_RELEASE, TSA_OP_HW_RESIDENT_COUNT,
    TSA_OP_HW_QUERY_RESIDENT, TSA_OP_KPSS_TEST, TSA_OP_SELECT_D, TSA_HAS_FIT, TSA_HAS_RESIDENT, TSA_TABLE_TSA,
    tsa_ops_check_spans, tsa_ops_publish,
)
from holtwinters.estimator import holtwinters_fit_ptr, holtwinters_forecast_ptr, holtwinters_predict_ptr
from holtwinters.impl.tsa.holtwinters_params import seasonal_name
from holtwinters.resident_state import (
    holtwinters_resident_prepare, holtwinters_resident_release, holtwinters_resident_count,
    holtwinters_query_resident,
)

from tsa.estimator import kpss_test_host, select_d_host
from tsa.mojolearn_tsa_run import _f32_ptr, _i32_ptr

comptime _VENDOR_CODE = (1 if COMPILED_VENDOR == "metal" else 2 if COMPILED_VENDOR == "cuda" else 3 if COMPILED_VENDOR == "hip" else -1)


def tsa_hw_ops_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        tsa_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == TSA_OP_FLAGS:
            r[] = TSA_HAS_FIT | TSA_HAS_RESIDENT
        elif operation == TSA_OP_VENDOR:
            r[] = _VENDOR_CODE
        elif operation == TSA_OP_MODE:
            r[] = Int(GLOBAL_NUMERIC_MODE)
        elif operation == TSA_OP_HW_FIT:
            if p[4] != 0 and p[4] != 1:
                raise Error("holtwinters_fit: init_method must be 0 (heuristic) or 1 (estimated), got " + String(p[4]))
            r[] = holtwinters_fit_ptr(f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), i32_ptr(a[3]),
                p[0], p[1], p[2], p[3], seasonal_name(p[5]), Float32(f[0]), p[4])
        elif operation == TSA_OP_HW_FORECAST:
            r[] = holtwinters_forecast_ptr(f32_ptr(a[0]), f32_ptr(a[1]), p[0], p[1], p[2], seasonal_name(p[4]), p[3])
        elif operation == TSA_OP_HW_PREDICT:
            r[] = holtwinters_predict_ptr(f32_ptr(a[0]), f32_ptr(a[1]), p[0], p[1], p[2], seasonal_name(p[5]), p[3], p[4])
        elif operation == TSA_OP_HW_RESIDENT_PREPARE:
            r[] = holtwinters_resident_prepare(f32_ptr(a[0]), p[0], p[1], p[2])
        elif operation == TSA_OP_HW_RESIDENT_RELEASE:
            holtwinters_resident_release(p[0])
            r[] = 0
        elif operation == TSA_OP_HW_RESIDENT_COUNT:
            r[] = holtwinters_resident_count()
        elif operation == TSA_OP_HW_QUERY_RESIDENT:
            r[] = holtwinters_query_resident(p[0], f32_ptr(a[0]), seasonal_name(p[3]), p[1], p[2])
        elif operation == TSA_OP_KPSS_TEST:
            r[] = kpss_test_host(_f32_ptr(a[0]), _i32_ptr(a[1]), _f32_ptr(a[2]), p[0], p[1], p[2], p[3], p[4],
                Float32(f[0]))
        elif operation == TSA_OP_SELECT_D:
            r[] = select_d_host(_f32_ptr(a[0]), _i32_ptr(a[1]), p[0], p[1], p[2], p[3], p[4], Float32(f[0]))
        else:
            raise Error("tsa ops: unknown operation " + String(operation))
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct TsaHwOpsDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(tsa_hw_ops_device_v1)


comptime _OPS_SLOT = "MojolearnTsaHwOpsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnTsaHwOpsDeviceFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime TSA_HW_OPS_DEVICE_API = SharedSlot[TsaHwOpsDeviceAPI, _OPS_SLOT]


def publish_tsa_hw_ops_device() raises:
    """Publish this image's table; called once, after the module is ready."""
    var api = TSA_HW_OPS_DEVICE_API.get_or_create_ptr()
    tsa_ops_publish[TSA_TABLE_TSA](Int(GLOBAL_NUMERIC_MODE), _VENDOR_CODE, Int(api[].entries.unsafe_ptr()),
        tsa_hw_ops_device_v1)
