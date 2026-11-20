# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The quality check of the classical family routes of the matrix-units row (lane fx-fast-nvidia-routes,
2026-11-19): `-D MOJOLEARN_DECOMP_FAST_MATRIX_UNITS` (decomposition/gram_matrix_units_route.mojo) and
`-D MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS` (neighbors/distance_matrix_units_route.mojo).

Three fixtures, each a product one route takes, compared with the exact FP32 kernel on the same operands:
  0  the PCA / truncated-SVD projection   Z[2048 x 128] = X[2048 x 512] . V[128 x 512]^T
  1  the Gram past the split-K width      Z[256 x 256]  = X[4096 x 256]^T . X
  2  the k-NN distance product            D[256 x 2048] = Q[256 x 256] . Y[2048 x 256]^T
Under a route's define (a fast-rows build):
  - every cell is inside `matrix_units_cell_bound` (operands truncated to 11 significant bits; AMD's fp32 MFMA
    keeps full operands and sits well inside it),
  - the RMS error is at most 1.25 * 2^-10 of the root of the summed squared terms (an 8-bit operand class,
    BF16, would be at least 1.6: a silent precision collapse fails),
  - on NVIDIA at least one cell differs from FP32 (the route was taken),
  - fixture 2 also holds the neighbour property: the index the route's distances rank first is, by the exact
    distances, within four cell bounds of the true nearest (two dots, each off by at most twice its bound).
Without the define the route answers False, nothing else runs, and the fixture passes on that answer.

GPU column only (the routes instantiate the MAX GPU matmul). Reference sums in Float64 on the host, one row of the
output per unit across all cores (`host_parallelize`).
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from checks.native_tests.int8_linalg import make_values
from max.gpu.host import DeviceContext
from gemm.contract import OP_NT, OP_TN
from gemm.checks.gemm_identical import identical_gemm_into as fp32_gemm_into
from gemm.checks.gemm_identical import identical_gemm_workspace_max_floats as fp32_workspace_max_floats
from gemm.fast_matrix_units import matrix_units_cell_bound
from gemm.checks.gemm_lowbit_check import _upload_f32, _download_f32, _poisoned
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_NVIDIA
from decomposition.gram_matrix_units_route import (
    DECOMP_MATRIX_UNITS, decomp_gram_units_into, decomp_project_units_into,
)
from neighbors.distance_matrix_units_route import KNN_DISTANCE_MATRIX_UNITS, knn_distance_units_product
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime FAMILY_UNITS_FIXTURES = 3
#: (1.25 * 2^-10)^2
comptime FAMILY_UNITS_RMS_REL_SQ = (Float64(1.25) / Float64(1024.0)) * (Float64(1.25) / Float64(1024.0))


def compare_family_rows(
    arm: List[Float32], ref_c: List[Float32], a: List[Float32], b: List[Float32],
    m: Int, n: Int, k: Int, op: Int, nearest: Bool,
) -> Tuple[Int, Int, Float64, Float64, Int]:
    """Per row of C, in parallel: (cells over the per-cell bound, cells whose bits differ, sum of squared
    differences, sum over cells and terms of (a b)^2, rows that fail the neighbour property). OP_NT reads
    a[m x k], b[n x k]; OP_TN reads a[k x m], b[k x n]. `nearest`: rank the row's cells as distances
    |b_j|^2 - 2 c_ij and test the neighbour property of the module docstring."""
    var over = List[Int](length=m, fill=0)
    var moved = List[Int](length=m, fill=0)
    var far = List[Int](length=m, fill=0)
    var diff_sq = List[Float64](length=m, fill=0.0)
    var term_sq = List[Float64](length=m, fill=0.0)
    var op_ptr = over.unsafe_ptr()
    var mp = moved.unsafe_ptr()
    var fp = far.unsafe_ptr()
    var dp = diff_sq.unsafe_ptr()
    var tp = term_sq.unsafe_ptr()
    var cp = arm.unsafe_ptr()
    var rp = ref_c.unsafe_ptr()
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    def task(i: Int) {imm op_ptr, imm mp, imm fp, imm dp, imm tp, imm cp, imm rp, imm ap, imm bp, imm m, imm n, imm k, imm op, imm nearest}:
        var row_over = 0
        var row_moved = 0
        var row_diff = Float64(0.0)
        var row_term = Float64(0.0)
        var max_bound = Float64(0.0)
        var arm_best = Float64(0.0)
        var ref_best = Float64(0.0)
        var ref_at_arm = Float64(0.0)
        for j in range(n):
            var abs_sum = Float64(0.0)
            var norm_b = Float64(0.0)
            for p in range(k):
                var av = Float64(0.0)
                var bv = Float64(0.0)
                if op == OP_NT:
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
                norm_b += bv * bv
            var got = cp.unsafe_load(i * n + j)
            var want = rp.unsafe_load(i * n + j)
            if bitcast[DType.uint32](got) != bitcast[DType.uint32](want):
                row_moved += 1
            var diff = Float64(got) - Float64(want)
            if diff < 0.0:
                diff = -diff
            var bound = matrix_units_cell_bound(abs_sum, k)
            # `not (diff <= bound)` so a NaN difference counts as over the bound
            if not (diff <= bound):
                row_over += 1
            if bound > max_bound:
                max_bound = bound
            row_diff += diff * diff
            if nearest:
                var arm_score = norm_b - Float64(2.0) * Float64(got)
                var ref_score = norm_b - Float64(2.0) * Float64(want)
                if j == 0 or ref_score < ref_best:
                    ref_best = ref_score
                if j == 0 or arm_score < arm_best:
                    arm_best = arm_score
                    ref_at_arm = ref_score
        var row_far = 0
        if nearest:
            if not (ref_at_arm - ref_best <= Float64(4.0) * max_bound):
                row_far = 1
        op_ptr.unsafe_store(i, row_over)
        mp.unsafe_store(i, row_moved)
        fp.unsafe_store(i, row_far)
        dp.unsafe_store(i, row_diff)
        tp.unsafe_store(i, row_term)
    host_parallelize(task, m)
    var total_over = 0
    var total_moved = 0
    var total_far = 0
    var total_diff = Float64(0.0)
    var total_term = Float64(0.0)
    for i in range(m):
        total_over += over[i]
        total_moved += moved[i]
        total_far += far[i]
        total_diff += diff_sq[i]
        total_term += term_sq[i]
    _ = arm
    _ = ref_c
    _ = a
    _ = b
    return (total_over, total_moved, total_diff, total_term, total_far)


def _judge(name: String, r: Tuple[Int, Int, Float64, Float64, Int]) raises:
    require(r[0] == 0, name + ": " + String(r[0]) + " cells outside the matrix-unit operand bound")
    require(
        r[2] <= FAMILY_UNITS_RMS_REL_SQ * r[3],
        name + ": squared error " + String(r[2]) + " over (1.25 * 2^-10)^2 of the squared terms " + String(r[3]),
    )
    require(r[4] == 0, name + ": " + String(r[4]) + " rows whose first-ranked neighbour is outside four cell bounds of the nearest")
    comptime if TARGET_COLUMN == COLUMN_NVIDIA:
        require(r[1] > 0, name + ": no cell differs from FP32, the matrix-units route was not taken")


def check_projection(ctx: DeviceContext) raises:
    var m = 2048
    var n = 128
    var k = 512
    var name = String("decomp projection 2048 x 128 x 512")
    var a = make_values(m, k, 131)
    var b = make_values(n, k, 147)
    var da = _upload_f32(ctx, a)
    var db = _upload_f32(ctx, b)
    var dc_arm = _poisoned(ctx, m * n)
    var took = decomp_project_units_into(ctx, dc_arm, da, db, m, n, k)
    ctx.synchronize()
    comptime if DECOMP_MATRIX_UNITS:
        require(took, name + ": the route declined a product its size rule takes")
        var dc_ref = _poisoned(ctx, m * n)
        var ws_ref = ctx.enqueue_create_buffer[DType.float32](max(1, fp32_workspace_max_floats(m, n, k)))
        fp32_gemm_into[False](ctx, dc_ref, da, db, ws_ref, m, n, k, OP_NT)
        ctx.synchronize()
        var arm = _download_f32(ctx, dc_arm, m * n, name + " (route)")
        var ref_c = _download_f32(ctx, dc_ref, m * n, name + " (FP32)")
        _judge(name, compare_family_rows(arm, ref_c, a, b, m, n, k, OP_NT, False))
        _ = dc_ref
        _ = ws_ref
    else:
        require(not took, name + ": the route took a product in a build without its define")
    _ = da
    _ = db
    _ = dc_arm


def check_gram(ctx: DeviceContext) raises:
    var m = 256
    var k = 4096
    var name = String("decomp Gram 256 x 256 x 4096")
    var x = make_values(k, m, 163)
    var dx = _upload_f32(ctx, x)
    var dc_arm = _poisoned(ctx, m * m)
    var took = decomp_gram_units_into(ctx, dc_arm, dx, m, k)
    ctx.synchronize()
    comptime if DECOMP_MATRIX_UNITS:
        require(took, name + ": the route declined a Gram past the split-K width")
        # the exact kernel takes two operand buffers; a second upload of the same values is its B
        var dx2 = _upload_f32(ctx, x)
        var dc_ref = _poisoned(ctx, m * m)
        var ws_ref = ctx.enqueue_create_buffer[DType.float32](max(1, fp32_workspace_max_floats(m, m, k)))
        fp32_gemm_into[False](ctx, dc_ref, dx, dx2, ws_ref, m, m, k, OP_TN)
        ctx.synchronize()
        var arm = _download_f32(ctx, dc_arm, m * m, name + " (route)")
        var ref_c = _download_f32(ctx, dc_ref, m * m, name + " (FP32)")
        _judge(name, compare_family_rows(arm, ref_c, x, x, m, m, k, OP_TN, False))
        _ = dx2
        _ = dc_ref
        _ = ws_ref
    else:
        require(not took, name + ": the route took a Gram in a build without its define")
    _ = dx
    _ = dc_arm


def check_knn_product(ctx: DeviceContext) raises:
    var m = 256
    var n = 2048
    var k = 256
    var name = String("k-NN distance product 256 x 2048 x 256")
    comptime if KNN_DISTANCE_MATRIX_UNITS:
        var a = make_values(m, k, 179)
        var b = make_values(n, k, 191)
        var da = _upload_f32(ctx, a)
        var db = _upload_f32(ctx, b)
        var dc_arm = _poisoned(ctx, m * n)
        var dc_ref = _poisoned(ctx, m * n)
        var ws_ref = ctx.enqueue_create_buffer[DType.float32](max(1, fp32_workspace_max_floats(m, n, k)))
        knn_distance_units_product(ctx, dc_arm, da, db, m, n, k)
        ctx.synchronize()
        fp32_gemm_into[False](ctx, dc_ref, da, db, ws_ref, m, n, k, OP_NT)
        ctx.synchronize()
        var arm = _download_f32(ctx, dc_arm, m * n, name + " (route)")
        var ref_c = _download_f32(ctx, dc_ref, m * n, name + " (FP32)")
        _judge(name, compare_family_rows(arm, ref_c, a, b, m, n, k, OP_NT, True))
        _ = da
        _ = db
        _ = dc_arm
        _ = dc_ref
        _ = ws_ref
    else:
        require(not KNN_DISTANCE_MATRIX_UNITS, name + ": unreachable")


def _run_gpu(mut report: Report) raises:
    var ctx = DeviceContext()
    var cases = report.cases(FAMILY_UNITS_FIXTURES)
    # driver-loop: over the three fixtures, one in-order device context
    for fixture_id in range(FAMILY_UNITS_FIXTURES):
        try:
            if fixture_id == 0:
                check_projection(ctx)
            elif fixture_id == 1:
                check_gram(ctx)
            else:
                check_knn_product(ctx)
        except error:
            cases.fail(fixture_id, error)


def run_cases(column: Column, mut report: Report):
    # GPU column only: the routes instantiate the MAX library's GPU matmul (see gemm_matrix_units.mojo).
    try:
        comptime if CPU_COLUMN_BUILD:
            raise Error("family_matrix_units runs on the gpu column only")
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("family_matrix_units", cpu=False, gpu=True, serial="", run=run_cases)
