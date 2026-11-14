# SPDX-License-Identifier: Apache-2.0
"""RMS-norm and linear training primitives on the device
(training_primitives_cases.mojo), through `native_primitive_operation`
(operations 2..5). The two fixtures run in order: every call shares the one
process training context (`neural_ctx`); the kernels keep their own
parallelism."""
from training.native_primitive_dispatch import native_primitive_operation
from checks.native_tests.training_primitives_cases import (
    RMS_M, RMS_D, LIN_M, LIN_K, LIN_N, rms_x, rms_w, rms_dy, check_rms, lin_a, lin_w, lin_dc, check_linear,
)
from checks.native_tests.training_check_common import require

comptime PRIMITIVE_FIXTURES = 2


def _rms() raises:
    var x = rms_x()
    var w = rms_w()
    var dy = rms_dy()
    var y = List[Float32](length=RMS_M * RMS_D, fill=0)
    var dx = List[Float32](length=RMS_M * RMS_D, fill=0)
    var dw = List[Float32](length=RMS_D, fill=0)
    var eps: List[Float64] = [0.0]
    _ = native_primitive_operation(2, [Int(y.unsafe_ptr()), Int(x.unsafe_ptr()), Int(w.unsafe_ptr())], [RMS_M, RMS_D], eps)
    _ = native_primitive_operation(3, [Int(dx.unsafe_ptr()), Int(dw.unsafe_ptr()), Int(dy.unsafe_ptr()),
                                       Int(x.unsafe_ptr()), Int(w.unsafe_ptr())], [RMS_M, RMS_D], eps)
    check_rms(x, w, dy, y, dx, dw)


def _linear() raises:
    var a = lin_a()
    var w = lin_w()
    var dc = lin_dc()
    var c = List[Float32](length=LIN_M * LIN_N, fill=0)
    var c1 = List[Float32](length=LIN_N, fill=0)
    var da = List[Float32](length=LIN_M * LIN_K, fill=0)
    var dw = List[Float32](length=LIN_N * LIN_K, fill=0)
    var none = List[Float64]()
    _ = native_primitive_operation(4, [Int(c.unsafe_ptr()), Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], [LIN_M, LIN_N, LIN_K], none)
    _ = native_primitive_operation(4, [Int(c1.unsafe_ptr()), Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], [1, LIN_N, LIN_K], none)
    _ = native_primitive_operation(5, [Int(da.unsafe_ptr()), Int(dw.unsafe_ptr()), Int(dc.unsafe_ptr()),
                                       Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], [LIN_M, LIN_N, LIN_K], none)
    check_linear(a, w, dc, c, da, dw, c1)


def run_checks() raises:
    var failed = 0
    for unit in range(PRIMITIVE_FIXTURES):  # ordered: one shared device context
        try:
            if unit == 0:
                _rms()
            else:
                _linear()
        except error:
            failed += 1
            print("FAIL training_primitives_gpu fixture", unit, String(error))
    require(failed == 0, "training_primitives_gpu failed " + String(failed) + " fixtures")
    print("PASS training_primitives_gpu:", PRIMITIVE_FIXTURES, "fixtures (rms_norm, linear)")
