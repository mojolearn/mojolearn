# SPDX-License-Identifier: Apache-2.0
"""The CPU reference image's column of the ARIMA operations table
(core/arima_entry_table.mojo, family ARIMA_TABLE_ARIMA): the entry
`_mojolearn_arima_host` publishes. The fit is `arima/host/`'s restatement
(`_refuse_method`, then `arima_host_fit_into`); prediction and forecast are
the shared host bodies. One request is one fit or one prediction; it has no
independent units to split here.
"""
from arima.arima_ops_host_inference import arima_ops_host_absent, arima_ops_host_inference
from arima.host.arima_host_entries import arima_host_fit_into
from arima.host.mojolearn_arima_host_run import _refuse_method
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.arima_entry_table import (
    ARIMA_OP_FLAGS, ARIMA_OP_VENDOR, ARIMA_OP_MODE, ARIMA_OP_FIT, ARIMA_OP_ORDER_CAPS, ARIMA_HAS_FIT,
    ARIMA_TABLE_ARIMA, arima_ops_check_spans, arima_ops_publish,
)
from core.hostptr import f32_ptr, i32_ptr
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot


def arima_ops_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        arima_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == ARIMA_OP_FLAGS:
            r[] = ARIMA_HAS_FIT
        elif operation == ARIMA_OP_VENDOR:
            r[] = 0
        elif operation == ARIMA_OP_MODE:
            r[] = Int(GLOBAL_NUMERIC_MODE)
        elif operation == ARIMA_OP_ORDER_CAPS:
            # Bit 2 (the float32 criterion) only; no grouped search runs here.
            r[] = 4 if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else 0
        elif operation == ARIMA_OP_FIT:
            var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
            var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
            var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
            _ = f32_ptr(a[0])
            _ = f32_ptr(a[1])
            _refuse_method(p[11])
            r[] = arima_host_fit_into(
                a[0], a[1], f32_ptr(a[2]), f32_ptr(a[3]), f32_ptr(a[4]), f32_ptr(a[5]), i32_ptr(a[6]),
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[12],
                a[7], f[0], f[1],
            )
        elif not arima_ops_host_inference(operation, aa, pp, result):
            arima_ops_host_absent(operation)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ArimaOpsHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(arima_ops_host_v1)


comptime _OPS_SLOT = "MojolearnArimaOpsHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnArimaOpsHostFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime ARIMA_OPS_HOST_API = SharedSlot[ArimaOpsHostAPI, _OPS_SLOT]


def publish_arima_ops_host() raises:
    """Publish this image's table; called once, after the module is ready."""
    var api = ARIMA_OPS_HOST_API.get_or_create_ptr()
    arima_ops_publish[ARIMA_TABLE_ARIMA](Int(GLOBAL_NUMERIC_MODE), 0, Int(api[].entries.unsafe_ptr()), arima_ops_host_v1)
