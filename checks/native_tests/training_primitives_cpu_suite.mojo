# SPDX-License-Identifier: Apache-2.0
"""RMS-norm and linear training primitives on the host column
(training_primitives_cases.mojo); the two independent fixtures run across
host cores."""
from core.host_parallel import host_parallelize
from core.mojolearn_training_host_impl import (
    rms_norm_forward_host_into, rms_norm_backward_host_into, linear_forward_host_into, linear_backward_host_into,
)
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
    _ = rms_norm_forward_host_into([Int(y.unsafe_ptr()), Int(x.unsafe_ptr()), Int(w.unsafe_ptr())], RMS_M, RMS_D, Float32(0))
    _ = rms_norm_backward_host_into([Int(dx.unsafe_ptr()), Int(dw.unsafe_ptr()), Int(dy.unsafe_ptr()),
                                     Int(x.unsafe_ptr()), Int(w.unsafe_ptr())], RMS_M, RMS_D, Float32(0))
    check_rms(x, w, dy, y, dx, dw)


def _linear() raises:
    var a = lin_a()
    var w = lin_w()
    var dc = lin_dc()
    var c = List[Float32](length=LIN_M * LIN_N, fill=0)
    var c1 = List[Float32](length=LIN_N, fill=0)
    var da = List[Float32](length=LIN_M * LIN_K, fill=0)
    var dw = List[Float32](length=LIN_N * LIN_K, fill=0)
    _ = linear_forward_host_into([Int(c.unsafe_ptr()), Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], LIN_M, LIN_N, LIN_K)
    _ = linear_forward_host_into([Int(c1.unsafe_ptr()), Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], 1, LIN_N, LIN_K)
    _ = linear_backward_host_into([Int(da.unsafe_ptr()), Int(dw.unsafe_ptr()), Int(dc.unsafe_ptr()),
                                   Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], LIN_M, LIN_N, LIN_K)
    check_linear(a, w, dc, c, da, dw, c1)


def run_checks() raises:
    var failures = List[Int](length=PRIMITIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit == 0:
                _rms()
            else:
                _linear()
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL training_primitives_cpu fixture", unit, String(error))
    host_parallelize(task, PRIMITIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "training_primitives_cpu failed " + String(failed) + " fixtures")
    print("PASS training_primitives_cpu:", PRIMITIVE_FIXTURES, "fixtures (rms_norm, linear)")
