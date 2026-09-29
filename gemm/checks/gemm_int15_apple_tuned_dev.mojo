# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int15i64.v1` on Apple's FLOAT matrix unit with
the fragments loaded STRAIGHT FROM FLOAT32 PLANES IN DEVICE MEMORY: no
threadgroup staging, no barrier in the `k` loop.

Lane lane/lowbit-apple-tuned, 2026-09-29, job 3. The arithmetic is
`gemm/checks/gemm_int15_apple_tuned.mojo`'s, form by form, with its proofs
(the header there): FOUR (four products, carried every 512 steps of `k`)
and TWO with the DEFERRED carry (the left operand whole, two products, the
float accumulators carried after every step of the unit into Int32 runs,
the runs cut in 16-bit halves every 64 steps of the unit). What moves is
where a fragment comes from, which is SCHEDULING (contract clause L-9): the
same integers reach the unit in the same float32 values.

THE PLANES. The codes' pieces (lane/lowbit-int15's int8 planes, `ah`, `al`,
`bh`, `bl`) are converted once into float32 planes, each value the integer
it names (exact: every magnitude is below 2^24):

    left   STEP-MAJOR, `kp x mp`: element (row i, step p) at `p * mp + i`;
           under TWO one plane, the code whole `ah * 128 + al`; under FOUR
           two, `ah` and `al`
    right  ROW-MAJOR, `np x kp`: element (column j, step p) at `j * kp + p`;
           two planes, `bh` and `bl`

`mp` and `np` are `m` and `n` rounded up to 64, `kp` is `k` rounded up to
8, and every padded element is ZERO (the padding rule of L-9: a zero code
adds nothing). Both orientations are then the one simdgroup load the
repository already uses on threadgroup memory, `M[r][c] = p[c * stride +
r]`, from device memory. At inference the right operand's planes are made
once per weight; the left operand's are made every call, and the clock
times that conversion inside the complete call.

THE DEVICE LOAD. Nothing else in the repository loads a fragment from
device memory. `_dev_load_t` is spelled as `gemm/checks/
gemm_apple_devload_probe1.mojo` spells it; the job runs that probe (and
spelling 2) first.

THE SABOTAGE ARMS are the tuned kernel's defines:
`-D MOJOLEARN_INT15_APPLE_TUNED_CHUNK_SABOTAGE=1` (FOUR's accumulators run
the whole of `k`, TWO's two steps of the unit) and
`-D MOJOLEARN_LOWBIT_SABOTAGE=1` (every stored cell flipped).

`-D MOJOLEARN_TUNED_NO_DEV=1` builds none of this (no variant, nothing
instantiated): the job's fallback when no spelling of the device load
compiles.
"""

from std.ffi import external_call
from std.gpu import block_idx, thread_idx
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, column_name
from checks.numerics_int15 import dequant_int15_pinned, int15_recombine
from core.apple_air import simdgroup_load_legacy_air
from gemm.checks.gemm_identical import _AMMA_M64, _AMMA_V2, _amma_mma
from gemm.checks.gemm_int15_apple_tuned import (
    INT15_TUNED2D_FLUSH_STEPS,
    INT15_TUNED2_CHUNK_STEPS,
    INT15_TUNED2_STEP_MAX,
    INT15_TUNED4_CHUNK_STEPS,
    INT15_TUNED4_STEP_MAX,
    INT15_TUNED_CHUNK_SABOTAGE,
    INT15_TUNED_EXACT_BOUND,
    INT15_TUNED_FORM_FOUR,
    INT15_TUNED_FORM_TWO,
    INT15_TUNED_VALUE_SABOTAGE,
    _flush_deferred,
)
from gemm.host.gemm_int15_oracle import INT15_MAX_K
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: Whether the device-fragment plans are built at all.
comptime INT15_TUNED_DEV_BUILT = (
    TARGET_COLUMN == COLUMN_APPLE and not is_defined["MOJOLEARN_TUNED_NO_DEV"]()
)

#: The padding of a plane's rows (both operands) and of its steps.
comptime DEV_ROW_PAD = 64
comptime DEV_STEP_PAD = 8

#: THE DEVICE-FRAGMENT VARIANTS.
comptime DEV_F2D_T32 = 0
comptime DEV_F2D_S32 = 1
comptime DEV_F2D_T64 = 2
comptime DEV_F4_T32 = 3
comptime DEV_F4_S32 = 4
comptime DEV_F2D_S16 = 5
comptime TUNED_DEV_COUNT = 6 if INT15_TUNED_DEV_BUILT else 0


def int15_apple_tuned_dev_variant_name(variant: Int) -> String:
    """`dev.<form>.<tile>`: t32 a 32 x 32 tile of 2 x 2 simdgroups with
    2 x 2 fragments (128 threads); t64 64 x 64, 2 x 2 simdgroups with 4 x 4
    fragments; s32 ONE simdgroup (32 threads) with 4 x 4 fragments, a
    32 x 32 tile; s16 one simdgroup with 2 x 2 fragments, 16 x 16."""
    if variant == DEV_F2D_T32:
        return String("dev.f2d.t32")
    if variant == DEV_F2D_S32:
        return String("dev.f2d.s32")
    if variant == DEV_F2D_T64:
        return String("dev.f2d.t64")
    if variant == DEV_F4_T32:
        return String("dev.f4.t32")
    if variant == DEV_F4_S32:
        return String("dev.f4.s32")
    if variant == DEV_F2D_S16:
        return String("dev.f2d.s16")
    return String("dev.unknown")


def _pad(x: Int, p: Int) -> Int:
    return ((x + p - 1) // p) * p


@always_inline
def _dev_load_t(p: MutPointer[Float32, MutAnyOrigin], stride: Int) -> _AMMA_M64:
    """Fragment M[r][c] = p[c * stride + r], from DEVICE memory."""
    var q = p
    comptime if simdgroup_load_legacy_air():
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", _AMMA_M64](
            q, Int64(stride), _AMMA_V2(0, 0), True
        )
    else:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", _AMMA_M64](
            q, _AMMA_V2(Int64(stride), 8), _AMMA_V2(Int64(stride), 1), _AMMA_V2(0, 0)
        )


def int15_dev_planes_left_kernel(
    d0: MutPointer[Float32, MutAnyOrigin],
    d1: MutPointer[Float32, MutAnyOrigin],
    qh: MutPointer[Int8, MutAnyOrigin],
    ql: MutPointer[Int8, MutAnyOrigin],
    rows_in: Int32,
    k_in: Int32,
    mp_in: Int32,
    kp_in: Int32,
    whole_in: Int32,
):
    """The left operand's planes, step-major and padded: one thread per
    element of `kp x mp`, the row fastest (the stores are consecutive)."""
    var rows = Int(rows_in)
    var k = Int(k_in)
    var mp = Int(mp_in)
    var kp = Int(kp_in)
    var t = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if t >= mp * kp:
        return
    var p = t // mp
    var i = t % mp
    var h = Float32(0)
    var l = Float32(0)
    if i < rows and p < k:
        h = qh.unsafe_load(i * k + p).cast[DType.float32]()
        l = ql.unsafe_load(i * k + p).cast[DType.float32]()
    if whole_in != Int32(0):
        d0.unsafe_store(t, h * Float32(128) + l)
    else:
        d0.unsafe_store(t, h)
        d1.unsafe_store(t, l)


def int15_dev_planes_right_kernel(
    d0: MutPointer[Float32, MutAnyOrigin],
    d1: MutPointer[Float32, MutAnyOrigin],
    qh: MutPointer[Int8, MutAnyOrigin],
    ql: MutPointer[Int8, MutAnyOrigin],
    rows_in: Int32,
    k_in: Int32,
    np_in: Int32,
    kp_in: Int32,
):
    """The right operand's planes, row-major and padded: one thread per
    element of `np x kp`, the step fastest."""
    var rows = Int(rows_in)
    var k = Int(k_in)
    var np_ = Int(np_in)
    var kp = Int(kp_in)
    var t = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if t >= np_ * kp:
        return
    var j = t // kp
    var p = t % kp
    var h = Float32(0)
    var l = Float32(0)
    if j < rows and p < k:
        h = qh.unsafe_load(j * k + p).cast[DType.float32]()
        l = ql.unsafe_load(j * k + p).cast[DType.float32]()
    d0.unsafe_store(t, h)
    d1.unsafe_store(t, l)


struct Int15DevPlanes(Movable):
    """The float32 planes of one product's two operands, grown on demand:
    the left code whole (TWO), the left pieces (FOUR), the right pieces."""

    var aw: DeviceBuffer[DType.float32]
    var a0: DeviceBuffer[DType.float32]
    var a1: DeviceBuffer[DType.float32]
    var b0: DeviceBuffer[DType.float32]
    var b1: DeviceBuffer[DType.float32]
    var a_cap: Int
    var b_cap: Int

    def __init__(out self, ctx: DeviceContext) raises:
        self.aw = ctx.enqueue_create_buffer[DType.float32](1)
        self.a0 = ctx.enqueue_create_buffer[DType.float32](1)
        self.a1 = ctx.enqueue_create_buffer[DType.float32](1)
        self.b0 = ctx.enqueue_create_buffer[DType.float32](1)
        self.b1 = ctx.enqueue_create_buffer[DType.float32](1)
        self.a_cap = 1
        self.b_cap = 1

    def ensure(mut self, ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
        var ka = _pad(m, DEV_ROW_PAD) * _pad(k, DEV_STEP_PAD)
        var kb = _pad(n, DEV_ROW_PAD) * _pad(k, DEV_STEP_PAD)
        if ka > self.a_cap:
            self.aw = ctx.enqueue_create_buffer[DType.float32](ka)
            self.a0 = ctx.enqueue_create_buffer[DType.float32](ka)
            self.a1 = ctx.enqueue_create_buffer[DType.float32](ka)
            self.a_cap = ka
        if kb > self.b_cap:
            self.b0 = ctx.enqueue_create_buffer[DType.float32](kb)
            self.b1 = ctx.enqueue_create_buffer[DType.float32](kb)
            self.b_cap = kb


def dev_variant_whole_left(variant: Int) -> Bool:
    """Whether the variant's left plane is the code whole (form TWO)."""
    return variant != DEV_F4_T32 and variant != DEV_F4_S32


def int15_dev_planes_left(
    ctx: DeviceContext,
    mut pl: Int15DevPlanes,
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    m: Int,
    k: Int,
    whole: Bool,
) raises:
    """Enqueue the left operand's planes; the caller waits."""
    pl.ensure(ctx, m, 1, k)
    var mp = _pad(m, DEV_ROW_PAD)
    var kp = _pad(k, DEV_STEP_PAD)
    var d0 = pl.aw.unsafe_ptr() if whole else pl.a0.unsafe_ptr()
    ctx.enqueue_function[int15_dev_planes_left_kernel](
        d0, pl.a1.unsafe_ptr(), ah.unsafe_ptr(), al.unsafe_ptr(),
        Int32(m), Int32(k), Int32(mp), Int32(kp), Int32(1 if whole else 0),
        grid_dim=((mp * kp + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )


def int15_dev_planes_right(
    ctx: DeviceContext,
    mut pl: Int15DevPlanes,
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    n: Int,
    k: Int,
) raises:
    """Enqueue the right operand's planes; the caller waits."""
    pl.ensure(ctx, 1, n, k)
    var np_ = _pad(n, DEV_ROW_PAD)
    var kp = _pad(k, DEV_STEP_PAD)
    ctx.enqueue_function[int15_dev_planes_right_kernel](
        pl.b0.unsafe_ptr(), pl.b1.unsafe_ptr(), bh.unsafe_ptr(), bl.unsafe_ptr(),
        Int32(n), Int32(k), Int32(np_), Int32(kp),
        grid_dim=((np_ * kp + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )


def identical_gemm_int15_apple_dev_kernel[
    SGM: Int, SGN: Int, FM: Int, FN: Int, FORM: Int
](
    c: MutPointer[Float32, MutAnyOrigin],
    a0: MutPointer[Float32, MutAnyOrigin],
    a1: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    b0: MutPointer[Float32, MutAnyOrigin],
    b1: MutPointer[Float32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    mp_in: Int32,
    kp_in: Int32,
    tile_row0_in: Int32,
):
    """OP_NT on the float32 planes, one `BM x BN` tile a block, `FM x FN`
    fragments a simdgroup, every fragment loaded from device memory. TWO
    (deferred carry) or FOUR, the arithmetic and the epilogue of the tuned
    kernel."""
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime NF = FM * FN
    comptime NC = 2 * NF
    comptime TWO = FORM == INT15_TUNED_FORM_TWO
    comptime assert FORM == INT15_TUNED_FORM_TWO or FORM == INT15_TUNED_FORM_FOUR, "TWO or FOUR"
    comptime assert DEV_ROW_PAD % BM == 0 and DEV_ROW_PAD % BN == 0, "a tile never passes the padding"
    comptime assert INT15_TUNED2_STEP_MAX * INT15_TUNED2_CHUNK_STEPS < INT15_TUNED_EXACT_BOUND, "TWO's step is exact"
    comptime assert INT15_TUNED4_STEP_MAX * INT15_TUNED4_CHUNK_STEPS < INT15_TUNED_EXACT_BOUND, "FOUR's chunk is exact"
    comptime CHUNK_UNITS = INT15_TUNED4_CHUNK_STEPS // 8
    var m = Int(m_in)
    var n = Int(n_in)
    var mp = Int(mp_in)
    var kp = Int(kp_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var nbn = (n + BN - 1) // BN
    var bid = Int(block_idx.x)
    var m0 = (Int(tile_row0_in) + bid // nbn) * BM
    var n0 = (bid % nbn) * BN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var hh_total = InlineArray[Int32, NC](fill=Int32(0))
    var mid_total = InlineArray[Int32, NC](fill=Int32(0))
    var ll_total = InlineArray[Int32, NC](fill=Int32(0))
    var run_l = InlineArray[Int32, NC](fill=Int32(0))
    var hh_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var mid_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var ll_acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var units = kp // 8
    for u in range(units):
        var k0 = u * 8
        var a0f = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
        var a1f = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
        var b0f = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
        var b1f = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
        comptime for fm in range(FM):
            var off = k0 * mp + m0 + (sgm * FM + fm) * 8
            a0f[fm] = _dev_load_t(a0 + off, mp)
            comptime if not TWO:
                a1f[fm] = _dev_load_t(a1 + off, mp)
        comptime for fq in range(FN):
            var off = (n0 + (sgn * FN + fq) * 8) * kp + k0
            b0f[fq] = _dev_load_t(b0 + off, kp)
            b1f[fq] = _dev_load_t(b1 + off, kp)
        comptime for fm in range(FM):
            comptime for fq in range(FN):
                comptime f = fm * FN + fq
                hh_acc[f] = _amma_mma(a0f[fm], b0f[fq], hh_acc[f])
                comptime if TWO:
                    ll_acc[f] = _amma_mma(a0f[fm], b1f[fq], ll_acc[f])
                else:
                    mid_acc[f] = _amma_mma(a0f[fm], b1f[fq], mid_acc[f])
                    mid_acc[f] = _amma_mma(a1f[fm], b0f[fq], mid_acc[f])
                    ll_acc[f] = _amma_mma(a1f[fm], b1f[fq], ll_acc[f])
        var last = u + 1 == units
        comptime if TWO:
            var step_end = True
            comptime if INT15_TUNED_CHUNK_SABOTAGE:
                step_end = ((u + 1) % 2) == 0 or last
            if step_end:
                comptime for f in range(NF):
                    comptime for e in range(2):
                        mid_total[2 * f + e] += hh_acc[f][e].cast[DType.int32]()
                        run_l[2 * f + e] += ll_acc[f][e].cast[DType.int32]()
                    hh_acc[f] = _AMMA_M64(0)
                    ll_acc[f] = _AMMA_M64(0)
            if (u + 1) % INT15_TUNED2D_FLUSH_STEPS == 0:
                _flush_deferred[NC](hh_total, ll_total, mid_total, run_l)
        else:
            var chunk_end = last
            comptime if not INT15_TUNED_CHUNK_SABOTAGE:
                chunk_end = chunk_end or ((u + 1) % CHUNK_UNITS) == 0
            if chunk_end:
                comptime for f in range(NF):
                    comptime for e in range(2):
                        hh_total[2 * f + e] += hh_acc[f][e].cast[DType.int32]()
                        mid_total[2 * f + e] += mid_acc[f][e].cast[DType.int32]()
                        ll_total[2 * f + e] += ll_acc[f][e].cast[DType.int32]()
                    hh_acc[f] = _AMMA_M64(0)
                    mid_acc[f] = _AMMA_M64(0)
                    ll_acc[f] = _AMMA_M64(0)
    comptime if TWO:
        _flush_deferred[NC](hh_total, ll_total, mid_total, run_l)
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var gi = m0 + (sgm * FM + fm) * 8 + frow
                var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                if gi < m and gj < n:
                    var at_cell = 2 * (fm * FN + fq) + e
                    var total = Int64(0)
                    comptime if TWO:
                        total = (Int64(hh_total[at_cell]) << Int64(16)) + Int64(ll_total[at_cell])
                    else:
                        total = int15_recombine(hh_total[at_cell], mid_total[at_cell], ll_total[at_cell])
                    var cell = dequant_int15_pinned(
                        total, Int(ea.unsafe_load(gi)) + Int(eb.unsafe_load(gj))
                    )
                    comptime if INT15_TUNED_VALUE_SABOTAGE:
                        cell = gemm_oracle_sabotage_value_flip(cell)
                    c.unsafe_store(gi * n + gj, cell)


def _launch_dev[
    SGM: Int, SGN: Int, FM: Int, FN: Int, FORM: Int
](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut pl: Int15DevPlanes,
    mut ea: DeviceBuffer[DType.int32],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    slice_macs: Int,
) raises:
    """In slices of whole rows of tiles, a wait between two slices."""
    comptime kern = identical_gemm_int15_apple_dev_kernel[SGM, SGN, FM, FN, FORM]
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    var mp = _pad(m, DEV_ROW_PAD)
    var kp = _pad(k, DEV_STEP_PAD)
    var left0 = pl.aw.unsafe_ptr() if FORM == INT15_TUNED_FORM_TWO else pl.a0.unsafe_ptr()
    var nbm = (m + BM - 1) // BM
    var nbn = (n + BN - 1) // BN
    var rows_per = slice_macs // (BM * n * k)
    if rows_per < 1:
        rows_per = 1
    var r0 = 0
    while r0 < nbm:
        var rows = nbm - r0
        if rows > rows_per:
            rows = rows_per
        ctx.enqueue_function[kern](
            c.unsafe_ptr(), left0, pl.a1.unsafe_ptr(), ea.unsafe_ptr(),
            pl.b0.unsafe_ptr(), pl.b1.unsafe_ptr(), eb.unsafe_ptr(),
            Int32(m), Int32(n), Int32(mp), Int32(kp), Int32(r0),
            grid_dim=(rows * nbn, 1, 1),
            block_dim=(SGM * SGN * 32, 1, 1),
        )
        r0 += rows
        if r0 < nbm:
            ctx.synchronize()


def identical_gemm_int15_apple_dev_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut pl: Int15DevPlanes,
    mut ea: DeviceBuffer[DType.int32],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    variant: Int,
    slice_macs: Int,
) raises:
    """The NAMED device-fragment variant on planes already made (by
    `int15_dev_planes_left` with the variant's `dev_variant_whole_left`,
    and `int15_dev_planes_right`). The caller waits."""
    comptime if not INT15_TUNED_DEV_BUILT:
        raise Error("identical_gemm_int15_apple_dev: not built (column " + column_name(TARGET_COLUMN) + " or MOJOLEARN_TUNED_NO_DEV)")
    else:
        if m <= 0 or n <= 0 or k <= 0 or k > INT15_MAX_K:
            raise Error("identical_gemm_int15_apple_dev: m, n, k positive and k at most " + String(INT15_MAX_K))
        if variant == DEV_F2D_T32:
            _launch_dev[2, 2, 2, 2, INT15_TUNED_FORM_TWO](ctx, c, pl, ea, eb, m, n, k, slice_macs)
        elif variant == DEV_F2D_S32:
            _launch_dev[1, 1, 4, 4, INT15_TUNED_FORM_TWO](ctx, c, pl, ea, eb, m, n, k, slice_macs)
        elif variant == DEV_F2D_T64:
            _launch_dev[2, 2, 4, 4, INT15_TUNED_FORM_TWO](ctx, c, pl, ea, eb, m, n, k, slice_macs)
        elif variant == DEV_F4_T32:
            _launch_dev[2, 2, 2, 2, INT15_TUNED_FORM_FOUR](ctx, c, pl, ea, eb, m, n, k, slice_macs)
        elif variant == DEV_F4_S32:
            _launch_dev[1, 1, 4, 4, INT15_TUNED_FORM_FOUR](ctx, c, pl, ea, eb, m, n, k, slice_macs)
        elif variant == DEV_F2D_S16:
            _launch_dev[1, 1, 2, 2, INT15_TUNED_FORM_TWO](ctx, c, pl, ea, eb, m, n, k, slice_macs)
        else:
            raise Error("identical_gemm_int15_apple_dev: no variant " + String(variant))


def identical_gemm_int15_apple_dev_from_pieces_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    variant: Int,
    slice_macs: Int,
) raises:
    """The gate's entry: the int8 planes in, the float32 planes made here,
    then the variant. The caller waits."""
    var pl = Int15DevPlanes(ctx)
    pl.ensure(ctx, m, n, k)
    int15_dev_planes_left(ctx, pl, ah, al, m, k, dev_variant_whole_left(variant))
    int15_dev_planes_right(ctx, pl, bh, bl, n, k)
    identical_gemm_int15_apple_dev_into(ctx, c, pl, ea, eb, m, n, k, variant, slice_macs)
    ctx.synchronize()
    _ = pl^
