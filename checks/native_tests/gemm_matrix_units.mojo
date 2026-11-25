# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The matrix-units route of `fast` (its default on NVIDIA and AMD) stays inside its arithmetic bound, and is off
everywhere else.

One training step's three products, each run twice on the device: through the neural dispatch with the vendor
route open (gemm/neural_dispatch.mojo::identical_gemm_into, which the matrix-units route of a fast-rows build
takes), and on the FP32 kernel itself (gemm/checks/gemm_identical.mojo::identical_gemm_into with the vendor
route closed, which the route never reaches):

    logits  Z[m x n]  = X[m x k] . W[n x k]^T      OP_NT
    dX      [m x k]   = dZ[m x n] . W[n x k]        OP_NN
    dW      [n x k]   = dZ[m x n]^T . X[m x k]      OP_TN

plus a neighbouring non-square shape, a product under the size rule and a bias column sum.

IN A BUILD WITH THE ROUTE (a `fast` build on NVIDIA or AMD), for a
product the size rule takes:
  1. every cell is within `matrix_units_cell_bound(sum_k |a_k b_k|, k)` of the FP32 cell: both operands of
     every term truncated to TF32 (2 * 2^-10 + 2^-20 of the absolute sum) plus the FP32 fold allowance;
  2. the root-mean-square error over the product is at most 1.25 * 2^-10 of sqrt(sum (a_k b_k)^2). Truncating
     both operands with independent uniform residues gives 1.08 * 2^-10 (a mean shift of 2^-10 of the cell and
     2^-10 / sqrt(6) of scatter); rounding to nearest gives 0.41 * 2^-10. An operand cut to 8 significant bits
     (BF16) gives at least 1.6 * 2^-10 and fails, as does a dropped or doubled term. This is the test that
     stops a silent precision collapse: the per-cell bound alone is a worst case and would admit one;
  3. at least one cell differs in bits from FP32: the route was taken. A build that has the route and
     routes nothing fails here.
  A product under the size rule and the bias sum must be bit-equal to FP32: they are not routed.
IN EVERY OTHER IDENTICAL-KERNEL BUILD (every identical build, and a fast build with the `_OFF` define) all six
pairs must be bit-equal: without the route `allow_vendor` selects nothing. (An old non-identical build sends the open route to the unbounded vendor product; nothing is asserted
there.)

The reference sums run on the host in Float64, one row of C per unit across the host's cores. One owned GPU
context submits the cases in order (a device queue is serial by nature). Not an identity run.
"""
from std.testing import TestSuite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from checks.native_tests.lowbit_oracles import make_values
from max.gpu.host import DeviceContext
from gemm.contract import OP_NN, OP_NT, OP_TN
from gemm.neural_dispatch import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.checks.gemm_identical import identical_gemm_into as fp32_gemm_into
from gemm.checks.gemm_identical import identical_gemm_workspace_max_floats as fp32_workspace_max_floats
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.fast_matrix_units import (
    GEMM_FAST_MATRIX_UNITS, matrix_units_takes, matrix_units_workspace_floats, matrix_units_cell_bound,
)
# TOMBSTONE: MOJOLEARN_GEMM_FAST_BF16_UNITS (A/B abf2: median 0.96, mlp-clf/taxi infer 1.86; dominated by the TF32
# route) deleted 2026-11-20 by lane/fx-verdicts6 with its bound and RMS tolerance here; code recoverable at 06d1e8ff34.
# Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_BF16_UNITS.patch
from gemm.checks.gemm_lowbit_check import _upload_f32, _download_f32, _poisoned
from checks.native_tests.testsuite_cases import require


comptime MATRIX_UNITS_FIXTURES = 6
#: test 2's tolerance: 1.25 * 2^-10, squared
comptime MATRIX_UNITS_RMS_REL_SQ = (Float64(1.25) / Float64(1024.0)) * (Float64(1.25) / Float64(1024.0))
#: the route compiled in
comptime UNITS_ARM = GEMM_FAST_MATRIX_UNITS


def fixture_shape(fixture_id: Int) raises -> Tuple[Int, Int, Int, Int]:
    """(m, n, k, op) in the dispatch's own arguments."""
    if fixture_id == 0:
        return (512, 512, 512, OP_NT)       # logits
    if fixture_id == 1:
        return (512, 512, 512, OP_NN)       # dX
    if fixture_id == 2:
        return (512, 512, 512, OP_TN)       # dW
    if fixture_id == 3:
        return (1024, 192, 768, OP_NT)      # a neighbouring shape, not square
    if fixture_id == 4:
        return (128, 128, 64, OP_NT)        # under the size rule: FP32
    if fixture_id == 5:
        return (1, 256, 4096, OP_NN)        # a bias column sum: FP32
    raise Error("unknown matrix-units fixture")


def fixture_name(fixture_id: Int) -> String:
    if fixture_id == 0:
        return String("logits NT 512x512x512")
    if fixture_id == 1:
        return String("dX NN 512x512x512")
    if fixture_id == 2:
        return String("dW TN 512x512x512")
    if fixture_id == 3:
        return String("neighbour NT 1024x192x768")
    if fixture_id == 4:
        return String("under the rule NT 128x128x64")
    return String("bias sum NN 1x256x4096")


def compare_rows(
    arm: List[Float32], ref_c: List[Float32], a: List[Float32], b: List[Float32],
    m: Int, n: Int, k: Int, op: Int,
) -> Tuple[Int, Int, Float64, Float64]:
    """Per row of C, in parallel: (cells over the per-cell bound, cells whose bits differ,
    sum of squared differences, sum over cells and terms of (a b)^2)."""
    var over = List[Int](length=m, fill=0)
    var moved = List[Int](length=m, fill=0)
    var diff_sq = List[Float64](length=m, fill=0.0)
    var term_sq = List[Float64](length=m, fill=0.0)
    var op_ptr = over.unsafe_ptr()
    var mp = moved.unsafe_ptr()
    var dp = diff_sq.unsafe_ptr()
    var tp = term_sq.unsafe_ptr()
    var cp = arm.unsafe_ptr()
    var rp = ref_c.unsafe_ptr()
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    def task(i: Int) {imm op_ptr, imm mp, imm dp, imm tp, imm cp, imm rp, imm ap, imm bp, imm m, imm n, imm k, imm op}:
        var row_over = 0
        var row_moved = 0
        var row_diff = Float64(0.0)
        var row_term = Float64(0.0)
        for j in range(n):
            var abs_sum = Float64(0.0)
            # the reference sums are read only by the arm's tests; a build without the arm compares bits alone
            comptime if GEMM_FAST_MATRIX_UNITS:
                for p in range(k):
                    var av = Float64(0.0)
                    var bv = Float64(0.0)
                    if op == OP_NN:
                        av = Float64(ap.unsafe_load(i * k + p))
                        bv = Float64(bp.unsafe_load(p * n + j))
                    elif op == OP_NT:
                        av = Float64(ap.unsafe_load(i * k + p))
                        bv = Float64(bp.unsafe_load(j * k + p))
                    else:
                        av = Float64(ap.unsafe_load(p * m + i))
                        bv = Float64(bp.unsafe_load(p * n + j))
                    var term = av * bv
                    if term < 0.0:
                        term = -term
                    abs_sum += term
                    row_term += term * term
            var got = cp.unsafe_load(i * n + j)
            var want = rp.unsafe_load(i * n + j)
            if bitcast[DType.uint32](got) != bitcast[DType.uint32](want):
                row_moved += 1
            var diff = Float64(got) - Float64(want)
            if diff < 0.0:
                diff = -diff
            comptime if UNITS_ARM:
                # `not (diff <= bound)` so a NaN difference counts as over the bound
                if not (diff <= matrix_units_cell_bound(abs_sum, k)):
                    row_over += 1
            row_diff += diff * diff
        op_ptr.unsafe_store(i, row_over)
        mp.unsafe_store(i, row_moved)
        dp.unsafe_store(i, row_diff)
        tp.unsafe_store(i, row_term)
    host_parallelize(task, m)
    var total_over = 0
    var total_moved = 0
    var total_diff = Float64(0.0)
    var total_term = Float64(0.0)
    for i in range(m):
        total_over += over[i]
        total_moved += moved[i]
        total_diff += diff_sq[i]
        total_term += term_sq[i]
    _ = arm
    _ = ref_c
    _ = a
    _ = b
    return (total_over, total_moved, total_diff, total_term)


def check_fixture(ctx: DeviceContext, fixture_id: Int) raises:
    var shape = fixture_shape(fixture_id)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var op = shape[3]
    var name = fixture_name(fixture_id)
    # A holds m k floats and B holds n k floats in all three addressings; the values are make_values' (signed,
    # magnitudes 2^-4 to 2^3, twelve mantissa bits, so a TF32 operand drops two of them).
    var a = make_values(m, k, 31 + fixture_id)
    var b = make_values(n, k, 47 + fixture_id)
    var da = _upload_f32(ctx, a)
    var db = _upload_f32(ctx, b)
    var dc_arm = _poisoned(ctx, m * n)
    var dc_ref = _poisoned(ctx, m * n)
    var need = identical_gemm_workspace_max_floats(m, n, k)
    need = max(need, matrix_units_workspace_floats(m, n, k, op))
    var ws_arm = ctx.enqueue_create_buffer[DType.float32](need)
    var ws_ref = ctx.enqueue_create_buffer[DType.float32](max(1, fp32_workspace_max_floats(m, n, k)))
    identical_gemm_into[True](ctx, dc_arm, da, db, ws_arm, m, n, k, op)
    ctx.synchronize()
    fp32_gemm_into[False](ctx, dc_ref, da, db, ws_ref, m, n, k, op)
    ctx.synchronize()
    var arm = _download_f32(ctx, dc_arm, m * n, name + " (vendor route open)")
    var ref_c = _download_f32(ctx, dc_ref, m * n, name + " (FP32)")
    var r = compare_rows(arm, ref_c, a, b, m, n, k, op)
    var routed = matrix_units_takes(m, n, k)
    comptime if UNITS_ARM:
        if routed:
            require(r[0] == 0, name + ": " + String(r[0]) + " cells outside the matrix unit's operand bound")
            require(
                r[2] <= MATRIX_UNITS_RMS_REL_SQ * r[3],
                name + ": squared error " + String(r[2]) + " over the arm's squared operand tolerance of the squared terms " + String(r[3]),
            )
            require(r[1] > 0, name + ": no cell differs from FP32, the matrix-units route was not taken")
        else:
            require(r[1] == 0, name + ": " + String(r[1]) + " cells moved on a product the size rule leaves on FP32")
    else:
        require(not routed, name + ": the size rule answered True in a build without the arm")
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            require(r[1] == 0, name + ": " + String(r[1]) + " cells differ between allow_vendor True and False without the arm")
    _ = da
    _ = db
    _ = dc_arm
    _ = dc_ref
    _ = ws_arm
    _ = ws_ref


# GPU column only: the body opens the vendor route (`identical_gemm_into[True]`), which instantiates the MAX
# library's GPU matmul; the CPU column never builds this file (2026-11-19: in the CPU binary it failed the build).
def test_fixtures() raises:
    var ctx = DeviceContext()
    var failed = String("")
    for fixture_id in range(MATRIX_UNITS_FIXTURES):  # driver-loop: over the six fixtures, one in-order device context
        try:
            check_fixture(ctx, fixture_id)
        except problem:
            failed += "\n      gemm_matrix_units row " + String(fixture_id) + ": " + String(problem)
    if failed.byte_length() > 0:
        raise Error("gemm_matrix_units: cases failed" + failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
