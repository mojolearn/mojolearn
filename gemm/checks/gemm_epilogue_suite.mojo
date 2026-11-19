# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native check of the GEMM store epilogue arm (lane fx-gemm-epi).

`-D MOJOLEARN_GEMM_FAST_EPI=1`. For every case and both kinds it compares,
word for word (bit patterns, so -0.0 and +0.0 differ):

  FUSED     `identical_gemm_epi_into[EPI]` into a destination holding PRIOR
  FOLLOWER  `identical_gemm_into` into a scratch, then the kernel the arm
            replaces (`bwd_add2_kernel`, operands in the kind's order)
  HOST      the follower's expression on the host from the downloaded plain
            product and PRIOR (`checks.numerics.ftz`, the host column)

and one three-product chain against `bwd_add3_kernel` ((q + k) + v).
PRIOR carries -0.0, subnormals and both signs; A carries a zero row (cells
that are exactly zero) and A and B a scaled row and column whose products
are subnormal (the packed kernel's window admission path).

The cases are sizes, not board rows: outputs that take leaf groups and the
fold, outputs past the group rule (all leaves in one launch, with and without
a fold) and ragged edges on both axes. A case the build's route declines is
printed as declined; the check is VACUOUS (raises) unless at least one group
case and one all-leaves case ran fused. On a build without a fused store
(no define, or a column without the packed body) it raises "not compiled":
it never reports a pass it did not earn.

One device, one stream: cases run one after another on the GPU because each
is a launch sequence on the single device under test; the host side only
fills fixtures and compares downloaded words (each a single pass, bounded by
the largest case, 6.3 M cells).
"""
from max.gpu.host import DeviceBuffer, DeviceContext

from autograd.kernels.transformer import BWD_TPB, bwd_add2_kernel, bwd_add3_kernel
from checks.fixture_rng import u16_row_f32
from checks.numerics import ftz
from checks.scaffold import CaseTally, bits, download, hex32, mode_name, upload
from std.memory import bitcast
from gemm.contract import OP_NN, OP_NT, OP_TN
from gemm.checks.gemm_epilogue import (
    GEMM_EPI_FUSES,
    identical_gemm_epi_into,
    identical_gemm_epi_takes,
)
from gemm.checks.gemm_identical import (
    GEMM_EPI_C_PLUS_CELL,
    GEMM_EPI_CELL_PLUS_C,
    gemm_default_ksplit_leaves,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)


def _operand(rows: Int, cols: Int, salt: Int, tiny_row: Int, tiny_col: Int, zero_row: Int) -> List[Float32]:
    """`rows x cols` row-major, cells in [-0.5, 0.5) (exact in Float32).
    Row `tiny_row` and column `tiny_col` (when >= 0) are scaled so their
    mutual products are subnormal; row `zero_row` (when >= 0) is +0.0."""
    var vals = List[Float32](capacity=rows * cols)
    for i in range(rows):
        for f in range(cols):
            var v = u16_row_f32(i, f, salt) - Float32(0.5)
            if i == tiny_row or f == tiny_col:
                v = v * Float32(1.0e-21)
            if i == zero_row:
                v = Float32(0.0)
            vals.append(v)
    return vals^


def _prior(count: Int, salt: Int) -> List[Float32]:
    """The destination's words before the join: both signs, and every 7th
    cell -0.0, every 11th a negative subnormal, every 13th a positive one."""
    var vals = List[Float32](capacity=count)
    for i in range(count):
        var v = (u16_row_f32(i, 3, salt) - Float32(0.5)) * Float32(8.0)
        if i % 7 == 0:
            v = bitcast[DType.float32](UInt32(0x80000000))
        elif i % 11 == 0:
            v = bitcast[DType.float32](UInt32(0x80000123))
        elif i % 13 == 0:
            v = bitcast[DType.float32](UInt32(0x00000456))
        vals.append(v)
    return vals^


def _add2(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    count: Int,
) raises:
    """The replaced follower launch, as `TransformerRules._add2` issues it."""
    ctx.enqueue_function[bwd_add2_kernel](
        dst.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(), Int32(count),
        grid_dim=((count + BWD_TPB - 1) // BWD_TPB, 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


def _mismatches(name: String, got: List[Float32], want: List[Float32], count: Int) -> Int:
    var bad = 0
    for i in range(count):
        if bits(got[i]) != bits(want[i]):
            if bad < 4:
                print("    ", name, "cell", i, "got", hex32(got[i]), "want", hex32(want[i]))
            bad += 1
    return bad


def _join_case[EPI: Int](
    ctx: DeviceContext, mut tally: CaseTally, m: Int, n: Int, k: Int, op: Int,
) raises -> Bool:
    """One product, one kind. True when the fused store ran."""
    var name = (
        String("epi ") + String(EPI) + " op " + String(op) + " " + String(m) + "x" + String(n) + "x" + String(k)
    )
    if not identical_gemm_epi_takes(m, n, k):
        print("  declined", name, "(the route has no fused store at this size)")
        return False
    var mn = m * n
    # A is m x k (k x m under OP_TN), B is k x n (n x k under OP_NT): m k and
    # n k words either way; the tiny/zero lines are planted by flat row.
    var a_rows = k if op == OP_TN else m
    var a_cols = m if op == OP_TN else k
    var b_rows = n if op == OP_NT else k
    var b_cols = k if op == OP_NT else n
    var a_vals = _operand(a_rows, a_cols, 11, 1, -1, 2)
    var b_vals = _operand(b_rows, b_cols, 23, -1, 1, -1)
    var p_vals = _prior(mn, 37)
    var da = upload(ctx, a_vals)
    var db = upload(ctx, b_vals)
    var d_prior = upload(ctx, p_vals)
    var d_fused = upload(ctx, p_vals)
    var d_plain = ctx.enqueue_create_buffer[DType.float32](mn)
    var d_follow = ctx.enqueue_create_buffer[DType.float32](mn)
    var ws = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m, n, k))
    ctx.synchronize()
    # FOLLOWER: the plain product, then the replaced kernel.
    identical_gemm_into(ctx, d_plain, da, db, ws, m, n, k, op)
    ctx.synchronize()
    comptime if EPI == GEMM_EPI_C_PLUS_CELL:
        _add2(ctx, d_follow, d_prior, d_plain, mn)
    else:
        _add2(ctx, d_follow, d_plain, d_prior, mn)
    ctx.synchronize()
    # FUSED: the join in the product's own store.
    identical_gemm_epi_into[EPI](ctx, d_fused, da, db, ws, m, n, k, op)
    ctx.synchronize()
    var plain = download(ctx, d_plain, mn)
    var follow = download(ctx, d_follow, mn)
    var fused = download(ctx, d_fused, mn)
    # HOST: the follower's expression in the host column.
    var host = List[Float32](capacity=mn)
    for i in range(mn):
        comptime if EPI == GEMM_EPI_C_PLUS_CELL:
            host.append(ftz(ftz(p_vals[i]) + ftz(plain[i])))
        else:
            host.append(ftz(ftz(plain[i]) + ftz(p_vals[i])))
    var bad = _mismatches(String("fused/follower"), fused, follow, mn)
    bad += _mismatches(String("fused/host"), fused, host, mn)
    tally.production(name, bad)
    # Every buffer a queued launch read stays named past the last wait.
    _ = da
    _ = db
    _ = d_prior
    _ = d_fused
    _ = d_plain
    _ = d_follow
    _ = ws
    return True


def _chain_case(ctx: DeviceContext, mut tally: CaseTally, m: Int, n: Int, k: Int) raises -> Bool:
    """S32's shape of use: three products into one destination, the first
    plain, the next two joined (prior + cell), against `bwd_add3_kernel`."""
    var name = String("chain (q + k) + v ") + String(m) + "x" + String(n) + "x" + String(k)
    if not identical_gemm_epi_takes(m, n, k):
        print("  declined", name, "(the route has no fused store at this size)")
        return False
    var mn = m * n
    var a_q = upload(ctx, _operand(m, k, 41, 1, -1, 2))
    var a_k = upload(ctx, _operand(m, k, 43, 3, -1, -1))
    var a_v = upload(ctx, _operand(m, k, 47, -1, -1, 4))
    var b_q = upload(ctx, _operand(k, n, 53, -1, 1, -1))
    var b_k = upload(ctx, _operand(k, n, 59, -1, 2, -1))
    var b_v = upload(ctx, _operand(k, n, 61, -1, -1, -1))
    var t_q = ctx.enqueue_create_buffer[DType.float32](mn)
    var t_k = ctx.enqueue_create_buffer[DType.float32](mn)
    var t_v = ctx.enqueue_create_buffer[DType.float32](mn)
    var d_follow = ctx.enqueue_create_buffer[DType.float32](mn)
    var d_fused = ctx.enqueue_create_buffer[DType.float32](mn)
    var ws = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m, n, k))
    ctx.synchronize()
    identical_gemm_into(ctx, t_q, a_q, b_q, ws, m, n, k, OP_NN)
    ctx.synchronize()
    identical_gemm_into(ctx, t_k, a_k, b_k, ws, m, n, k, OP_NN)
    ctx.synchronize()
    identical_gemm_into(ctx, t_v, a_v, b_v, ws, m, n, k, OP_NN)
    ctx.synchronize()
    ctx.enqueue_function[bwd_add3_kernel](
        d_follow.unsafe_ptr(), t_q.unsafe_ptr(), t_k.unsafe_ptr(), t_v.unsafe_ptr(), Int32(mn),
        grid_dim=((mn + BWD_TPB - 1) // BWD_TPB, 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.synchronize()
    identical_gemm_into(ctx, d_fused, a_q, b_q, ws, m, n, k, OP_NN)
    ctx.synchronize()
    identical_gemm_epi_into[GEMM_EPI_C_PLUS_CELL](ctx, d_fused, a_k, b_k, ws, m, n, k, OP_NN)
    ctx.synchronize()
    identical_gemm_epi_into[GEMM_EPI_C_PLUS_CELL](ctx, d_fused, a_v, b_v, ws, m, n, k, OP_NN)
    ctx.synchronize()
    var follow = download(ctx, d_follow, mn)
    var fused = download(ctx, d_fused, mn)
    tally.production(name, _mismatches(String("chain fused/add3"), fused, follow, mn))
    _ = a_q
    _ = a_k
    _ = a_v
    _ = b_q
    _ = b_k
    _ = b_v
    _ = t_q
    _ = t_k
    _ = t_v
    _ = d_follow
    _ = d_fused
    _ = ws
    return True


def run_checks() raises:
    print("== gemm/checks/gemm_epilogue_suite.mojo [" + mode_name() + "] ==")
    comptime if not GEMM_EPI_FUSES:
        raise Error(
            "gemm_epilogue_check: NOT COMPILED. This build has no fused GEMM store"
            " (needs -D MOJOLEARN_GEMM_FAST_EPI=1, identical mode and a column whose"
            " shipped GEMM body is the packed kernel); nothing was checked."
        )
    else:
        var tally = CaseTally()
        var group_ran = 0
        var leaves_ran = 0
        # (m, n, k): leaf groups + fold (3 and 6 leaves, ragged edges), then
        # all leaves in one launch: one leaf, two leaves past the group rule.
        var shapes: List[Int] = [
            1536, 320, 320,
            1000, 300, 700,
            2048, 384, 1024,
            520, 510, 96,
            4096, 256, 128,
            3000, 2100, 200,
        ]
        with DeviceContext() as ctx:
            # driver-loop: over the check's cases
            for s in range(len(shapes) // 3):
                var m = shapes[3 * s]
                var n = shapes[3 * s + 1]
                var k = shapes[3 * s + 2]
                var grouped = gemm_default_ksplit_leaves(m, n, k) > 0
                var ran = 0
                if _join_case[GEMM_EPI_C_PLUS_CELL](ctx, tally, m, n, k, OP_NN):
                    ran += 1
                if _join_case[GEMM_EPI_CELL_PLUS_C](ctx, tally, m, n, k, OP_NN):
                    ran += 1
                if _join_case[GEMM_EPI_C_PLUS_CELL](ctx, tally, m, n, k, OP_NT):
                    ran += 1
                if _join_case[GEMM_EPI_CELL_PLUS_C](ctx, tally, m, n, k, OP_TN):
                    ran += 1
                if grouped:
                    group_ran += ran
                else:
                    leaves_ran += ran
            if _chain_case(ctx, tally, 1536, 320, 320):
                group_ran += 1
            if _chain_case(ctx, tally, 520, 510, 96):
                leaves_ran += 1
        print("  fused cases: leaf groups + fold", group_ran, " all leaves", leaves_ran)
        if group_ran == 0 or leaves_ran == 0:
            raise Error(
                "gemm_epilogue_check: VACUOUS, the fused store ran on "
                + String(group_ran) + " group cases and " + String(leaves_ran)
                + " all-leaves cases; both forms must run"
            )
        tally.finish(String("gemm_epilogue_check"))
