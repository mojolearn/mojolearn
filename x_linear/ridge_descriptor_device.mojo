# SPDX-License-Identifier: Apache-2.0
"""The weighted/one-feature Ridge kernel exposed to native linear RUN."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.linear_fit_types import RIDGE_FIT_ABI, ridge_fit_check_spans
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from x_linear.device import fit_device
from x_linear.dispatch import ALGO_RIDGE
from x_linear.finite_device import XLIN_IDN_DEV_FINITE
from x_linear.ops import FP
from x_linear.x_linear_input_impl import x_linear_finite_device


def ridge_fit_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        ridge_fit_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var rows = p[0]
        var cols = p[1]
        var n_y = rows * (2 if p[3] else 1)
        var ip: List[Int32] = [1, Int32(p[2]), 1, Int32(p[3])]
        var fp: List[Float32] = [Float32(f[0])]
        var x = FP(unsafe_from_address=a[0])
        var y = FP(unsafe_from_address=a[1])
        comptime if not XLIN_IDN_DEV_FINITE:
            x_linear_finite_device(x, rows * cols, "X")
            x_linear_finite_device(y, n_y, "y")
        fit_device(ALGO_RIDGE, x, rows * cols, y, n_y, rows, cols,
            ip, fp, cols + 5, max(3 * cols * cols + 4 * cols + 1 + 2 * rows, 1), 1,
            FP(unsafe_from_address=a[2]))
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = cols + 5
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct RidgeFitDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(RIDGE_FIT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(ridge_fit_device_v1)


comptime _API_SLOT = "MojolearnRidgeFitDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnRidgeFitDeviceFastV1"
comptime RIDGE_FIT_DEVICE_API = SharedSlot[RidgeFitDeviceAPI, _API_SLOT]
