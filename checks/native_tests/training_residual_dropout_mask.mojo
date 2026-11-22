# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The residual-dropout layer's forward and backward obey one mask, moved
from test_public_verifier_gap_methods.py
`test_residual_dropout_forward_and_backward_obey_same_mask` (2026-11-15).

values = (1..32) * 0.125 as 4 x 8 float32, residual 0.5, upstream gradient
1.0; seed 42 (seed words 42, 0, as the door's `_rd_params` splits it),
stream 3, offset 17; the calls the PACK doors make: `residual_dropout_host`
(training/residual_dropout_host.mojo) on the host column and
`residual_dropout_device` (training/residual_dropout.mojo) on the device
column, forward then backward with the same coordinates.

  0  p 0      d_values are all 1.0
  1  p 0.375  d_values hold at least one 0 and at least one nonzero
In both: d_residual equals the upstream gradient (1.0) by value, and the
forward output's words equal residual + values * d_values (an unfused
float32 product, then the float32 add, as NumPy computes it).
The probability refusals stay in the Python test (they gate its ValueError).
"""
from std.memory import bitcast
from checks.numerics import identical_mul
from core.hostptr import f32_ptr
from core.neural_context import neural_ctx
from training.residual_dropout import residual_dropout_device
from training.residual_dropout_boundary_run import _RD_DEV_CTX
from training.residual_dropout_host import residual_dropout_host
from std.testing import TestSuite, assert_true as require
from checks.native_tests.testsuite_cases import check_cases
from std.sys.compile import is_defined


comptime RD_MASK_FIXTURES = 2
comptime RD_N = 32
comptime RD_OFFSET = 17
comptime RD_SEED_LO = 42
comptime RD_SEED_HI = 0
comptime RD_STREAM = 3


def rd_probability(unit: Int) -> Float32:
    return Float32(0.375) if unit == 1 else Float32(0.0)


def rd_values() -> List[Float32]:
    var values = List[Float32](length=RD_N, fill=0.0)
    for i in range(RD_N):  # small-loop(fixture: 32 test values)
        values[i] = Float32(i + 1) * Float32(0.125)
    return values^


struct RdOutputs(Movable):
    var forward: List[Float32]
    var dvalues: List[Float32]
    var dresidual: List[Float32]

    def __init__(out self):
        self.forward = List[Float32](length=RD_N, fill=-7.0)
        self.dvalues = List[Float32](length=RD_N, fill=-7.0)
        self.dresidual = List[Float32](length=RD_N, fill=-7.0)


def check_outputs(unit: Int, outputs: RdOutputs, values: List[Float32]) raises:
    var tag = "residual dropout p " + String(rd_probability(unit))
    var zeros = 0
    var nonzeros = 0
    for i in range(RD_N):  # small-loop(fixture compare)
        require(outputs.dresidual[i] == Float32(1.0), tag + ": d_residual at " + String(i) + " is "
                + String(outputs.dresidual[i]) + ", expected the upstream 1.0")
        var d = outputs.dvalues[i]
        if d == Float32(0.0):
            zeros += 1
        else:
            nonzeros += 1
        if unit == 0:
            require(d == Float32(1.0), tag + ": d_values at " + String(i) + " is " + String(d) + ", expected 1.0")
        var want = Float32(0.5) + identical_mul(values[i], d)
        require(bitcast[DType.uint32](outputs.forward[i]) == bitcast[DType.uint32](want),
                tag + ": forward at " + String(i) + " is " + String(outputs.forward[i]) + ", expected "
                + String(want) + " = residual + values * d_values")
    if unit == 1:
        require(zeros > 0 and nonzeros > 0, tag + ": the mask keeps " + String(nonzeros) + " and drops "
                + String(zeros) + "; expected both")


# ---- The host column ----
# The two probabilities are independent fixtures across host cores, each
# with its own buffers and failure slot.
def run_host_fixture(unit: Int) raises:
    var values = rd_values()
    var residual = List[Float32](length=RD_N, fill=0.5)
    var ones = List[Float32](length=RD_N, fill=1.0)
    var outputs = RdOutputs()
    var p = rd_probability(unit)
    var yp = f32_ptr(Int(outputs.forward.unsafe_ptr()))
    _ = residual_dropout_host[False](yp, yp, f32_ptr(Int(values.unsafe_ptr())), f32_ptr(Int(residual.unsafe_ptr())),
                                     RD_N, RD_OFFSET, p, RD_SEED_LO, RD_SEED_HI, RD_STREAM)
    var gp = f32_ptr(Int(ones.unsafe_ptr()))
    _ = residual_dropout_host[True](f32_ptr(Int(outputs.dvalues.unsafe_ptr())),
                                    f32_ptr(Int(outputs.dresidual.unsafe_ptr())), gp, gp,
                                    RD_N, RD_OFFSET, p, RD_SEED_LO, RD_SEED_HI, RD_STREAM)
    check_outputs(unit, outputs, values)
    _ = residual^
    _ = ones^


# ---- The device column ----
# The fixtures share this process's neural training context and run in
# order; each launch keeps its own parallelism.
def run_device_fixture(unit: Int) raises:
    var values = rd_values()
    var residual = List[Float32](length=RD_N, fill=0.5)
    var ones = List[Float32](length=RD_N, fill=1.0)
    var outputs = RdOutputs()
    var p = rd_probability(unit)
    var ctx = neural_ctx[_RD_DEV_CTX]()
    var yp = f32_ptr(Int(outputs.forward.unsafe_ptr()))
    _ = residual_dropout_device[False](ctx, yp, yp, f32_ptr(Int(values.unsafe_ptr())),
                                       f32_ptr(Int(residual.unsafe_ptr())),
                                       RD_N, RD_OFFSET, p, RD_SEED_LO, RD_SEED_HI, RD_STREAM)
    var gp = f32_ptr(Int(ones.unsafe_ptr()))
    _ = residual_dropout_device[True](ctx, f32_ptr(Int(outputs.dvalues.unsafe_ptr())),
                                      f32_ptr(Int(outputs.dresidual.unsafe_ptr())), gp, gp,
                                      RD_N, RD_OFFSET, p, RD_SEED_LO, RD_SEED_HI, RD_STREAM)
    check_outputs(unit, outputs, values)
    _ = residual^
    _ = ones^


def _cpu_cases() raises:
    check_cases("training_residual_dropout_mask", RD_MASK_FIXTURES, run_host_fixture)


def _gpu_cases() raises:
    var failed = String("")
    var count = 0
    for unit in range(RD_MASK_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_device_fixture(unit)
        except error:
            count += 1
            failed += "\n      training_residual_dropout_mask row " + String(unit) + ": " + String(error)
    if count > 0:
        raise Error("training_residual_dropout_mask: " + String(count) + " of " + String(RD_MASK_FIXTURES) + " cases failed" + failed)


def test_cases() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        _cpu_cases()
    else:
        _gpu_cases()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
