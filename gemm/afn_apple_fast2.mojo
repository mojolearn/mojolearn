# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-neural-w2-gemm2 (2026-10-03): the second Apple FAST GEMM
round. A second simdgroup-matrix kernel, `afn2_gemm_kernel`, reached from
`gemm/afn_apple_fast.mojo::_afn_dispatch` (one comptime branch) for every
fp32, fp32 x bf16-bits and bf16-bits x bf16-bits product the wave-1 kernel
would run at its square tile without a split.

Every symbol here is compiled only under `AFN_GEMM2_APPLE` (the FAST tier on
an Apple GPU build, never the CPU column) AND one `MOJOLEARN_AFN_GEMM2_*`
define; IDENTICAL never instantiates a line of this file (the dispatch
branch reads `AFN_GEMM2_ON`, False there).

Products are exact on the matrix unit (fp32 x fp32, bf16 bits widened by
`bits << 16`, exact) and the accumulate is f32; only the fold order moves.
No approximation, no subsampling.

THE DEFINES (each default OFF; `MOJOLEARN_AFN_GEMM2_ALL` turns on all four):

  MOJOLEARN_AFN_GEMM2_BIGTILE  128x64 block tiles: 8 simdgroups (4x2), each
      owning a 32x32 corner (4x4 fragments, 16 accumulators), when the
      128x64 grid still covers the cores twice (`2 x AFN_GEMM_CORES`
      tiles); else the 64x64 tile. Each staged A word feeds 8 fragments'
      worth of columns instead of 4: half the B staging per output cell.
  MOJOLEARN_AFN_GEMM2_DBUF     two threadgroup pages always: the window
      depth `KB` is the deepest (multiple of 8, whole 4-wide slots) whose
      TWO pages fit Apple's 32 KB, so window w + 1 stages into the other
      page while w multiplies, one barrier per window (the wave-1 kernel at
      its default KB = 32 has one page and two barriers per window).
  MOJOLEARN_AFN_GEMM2_DIRECT_B B fragments are built straight from device
      memory in each lane's two registers (the per-lane fragment layout
      `(frow, fcol + e)` that the FAST k-NN kernel already builds its query
      fragments with, neighbors/impl/detail/fast_mma_knn.mojo:204-215): no
      threadgroup page for B, so the A panel takes the whole 32 KB (KB up to
      64 at 64x64, 48 at 128x64; with DBUF, the deepest two A pages).
  MOJOLEARN_AFN_GEMM2_SWIZZLE  block-index swizzle: consecutive blocks walk
      `AFN_GEMM2_SWZ_G` (8) tile rows column by column, so the blocks
      resident together share B columns and A rows in the L2. Pure
      scheduling: every cell's products and order are unchanged.

Shared-memory pages carry a fits gate; kernel arguments are fixed-width;
pointers cross only `@always_inline` callees.
"""

from std.ffi import external_call
from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import (
    COLUMN_APPLE,
    TARGET_COLUMN,
    column_shared_limit,
    lib_smem_page_fits_for,
    lib_smem_pages_for,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.apple_air import simdgroup_load_legacy_air
from gemm.contract import OP_NN, OP_NT, OP_TN


# ===========================================================================
# The define table
# ===========================================================================

#: The tier and the vendor every candidate is behind.
comptime AFN_GEMM2_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and TARGET_COLUMN == COLUMN_APPLE
)
comptime AFN_GEMM2_ALL = AFN_GEMM2_APPLE and is_defined["MOJOLEARN_AFN_GEMM2_ALL"]()
comptime AFN_GEMM2_BIGTILE = AFN_GEMM2_ALL or (
    AFN_GEMM2_APPLE and is_defined["MOJOLEARN_AFN_GEMM2_BIGTILE"]()
)
comptime AFN_GEMM2_DBUF = AFN_GEMM2_ALL or (
    AFN_GEMM2_APPLE and is_defined["MOJOLEARN_AFN_GEMM2_DBUF"]()
)
comptime AFN_GEMM2_DIRECT_B = AFN_GEMM2_ALL or (
    AFN_GEMM2_APPLE and is_defined["MOJOLEARN_AFN_GEMM2_DIRECT_B"]()
)
comptime AFN_GEMM2_SWIZZLE = AFN_GEMM2_ALL or (
    AFN_GEMM2_APPLE and is_defined["MOJOLEARN_AFN_GEMM2_SWIZZLE"]()
)
#: Any candidate on: the wave-1 entry points route through this kernel (and
#: `afn_apple_fast.mojo` turns its SIMDGROUP and BF16_MMA entry points on).
comptime AFN_GEMM2_ON = (
    AFN_GEMM2_BIGTILE or AFN_GEMM2_DBUF or AFN_GEMM2_DIRECT_B or AFN_GEMM2_SWIZZLE
)

#: GPU cores the grid should cover twice over (the wave-1 define; M3 Ultra 80).
comptime AFN_GEMM2_CORES = get_defined_int["MOJOLEARN_AFN_GEMM_CORES", 80]()
#: Swizzle group: tile rows walked together.
comptime AFN_GEMM2_SWZ_G = get_defined_int["MOJOLEARN_AFN_GEMM2_SWZ_G", 8]()
#: The swizzle the kernels are built with (1 = row-major block order).
comptime AFN_GEMM2_SWZ = AFN_GEMM2_SWZ_G if AFN_GEMM2_SWIZZLE else 1

comptime _M64 = SIMD[DType.float32, 64]
comptime _V2 = SIMD[DType.int64, 2]
comptime _SPtr = UnsafePointer[Float32, MutUntrackedOrigin, address_space = AddressSpace.SHARED]


# ===========================================================================
# The window depth (host, comptime)
# ===========================================================================


def afn2_kb(bm: Int, bn: Int, nt: Int, dbuf: Bool, direct_b: Bool) -> Int:
    """The window depth for a `bm x bn` tile of `nt` threads, evaluated at
    comptime: the deepest multiple of 8 (at most 32 with a staged B page, 64
    without one) whose page(s) fit Apple's threadgroup memory, two pages when
    `dbuf`, and whose operand words split into whole 4-wide slots per thread
    (`_afn2_gload`'s assertion)."""
    var limit = column_shared_limit(COLUMN_APPLE)
    var cap = 64 if direct_b else 32
    var kb = cap
    while kb >= 8:
        var page = kb * (bm + 4)
        if not direct_b:
            page += bn * (kb + 4)
        var need = page * 4
        if dbuf:
            need = 2 * need
        var slots_ok = (bm * kb) % (4 * nt) == 0
        if not direct_b:
            slots_ok = slots_ok and (bn * kb) % (4 * nt) == 0
        if need <= limit and slots_ok:
            return kb
        kb -= 8
    return 8


# ===========================================================================
# The matrix unit
# ===========================================================================


@always_inline
def _afn2_load_t(p: _SPtr, stride: Int) -> _M64:
    """Fragment M[r][c] = p[c * stride + r] (the transposed load; the AIR
    signature depends on the target, `core/apple_air.mojo`)."""
    comptime if simdgroup_load_legacy_air():
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, Int64(stride), _V2(0, 0), True
        )
    else:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, _V2(Int64(stride), 8), _V2(Int64(stride), 1), _V2(0, 0)
        )


@always_inline
def _afn2_mma(a: _M64, b: _M64, c: _M64) -> _M64:
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        _M64,
    ](a, b, c)


@always_inline
def _afn2_widen[T: DType](v: Scalar[T]) -> Float32:
    """The operand word as float32: bf16 bits are the top half of a float32
    (exact, contract L-1); float32 is itself."""
    comptime if T == DType.uint16:
        return bitcast[DType.float32](v.cast[DType.uint32]() << UInt32(16))
    else:
        return v.cast[DType.float32]()


@always_inline
def _afn2_gload[
    ROWS: Int, KB: Int, NT: Int, T: DType
](
    src: MutPointer[Scalar[T], MutAnyOrigin],
    outer_stride: Int,
    k_stride: Int,
    base_outer: Int,
    outer_limit: Int,
    k0: Int,
    chunk: Int,
    tid: Int,
    ofast: Bool,
) -> SIMD[DType.float32, 4 * ((ROWS * KB) // (4 * NT))]:
    """One window's operand words for this thread, 4 per slot, widened (the
    wave-1 `_afn_gload`). `ofast` (the outer index has stride 1): slot =
    (p, 4 consecutive outer); else slot = (outer, 4 consecutive p). Words
    outside `outer_limit` or past `chunk` read +0.0."""
    comptime SL = (ROWS * KB) // (4 * NT)
    comptime assert SL * 4 * NT == ROWS * KB, "_afn2_gload: whole slots"
    comptime assert ROWS % 4 == 0 and KB % 4 == 0, "_afn2_gload: 4-wide slots"
    var r = SIMD[DType.float32, 4 * SL](0.0)
    comptime for sl in range(SL):
        var s = sl * NT + tid
        if ofast:
            var pp = s // (ROWS // 4)
            var o4 = (s % (ROWS // 4)) * 4
            if pp < chunk:
                var go = base_outer + o4
                var off = go + (k0 + pp) * k_stride
                if go + 3 < outer_limit:
                    var v = src.unsafe_load[width=4](off)
                    comptime for q in range(4):
                        r[4 * sl + q] = _afn2_widen[T](v[q])
                else:
                    comptime for q in range(4):
                        if go + q < outer_limit:
                            r[4 * sl + q] = _afn2_widen[T](src.unsafe_load(off + q))
        else:
            var o = s // (KB // 4)
            var p4 = (s % (KB // 4)) * 4
            var go = base_outer + o
            if go < outer_limit:
                var off = go * outer_stride + (k0 + p4) * k_stride
                if k_stride == 1 and p4 + 3 < chunk:
                    var v = src.unsafe_load[width=4](off)
                    comptime for q in range(4):
                        r[4 * sl + q] = _afn2_widen[T](v[q])
                else:
                    comptime for q in range(4):
                        if p4 + q < chunk:
                            r[4 * sl + q] = _afn2_widen[T](src.unsafe_load(off + q * k_stride))
    return r


@always_inline
def _afn2_stage[
    ROWS: Int, KB: Int, NT: Int, PMAJOR: Bool, ST: Int
](
    dst: _SPtr,
    r: SIMD[DType.float32, 4 * ((ROWS * KB) // (4 * NT))],
    tid: Int,
    ofast: Bool,
):
    """Store one thread's slots (the wave-1 `_afn_stage`). `PMAJOR`: element
    (outer o, p) at `dst[p * ST + o]` (A); else at `dst[o * ST + p]` (B)."""
    comptime SL = (ROWS * KB) // (4 * NT)
    comptime for sl in range(SL):
        var s = sl * NT + tid
        var v = SIMD[DType.float32, 4](0.0)
        comptime for q in range(4):
            v[q] = r[4 * sl + q]
        if ofast:
            var pp = s // (ROWS // 4)
            var o4 = (s % (ROWS // 4)) * 4
            comptime if PMAJOR:
                (dst + pp * ST + o4).store[alignment=16](v)
            else:
                comptime for q in range(4):
                    dst[(o4 + q) * ST + pp] = v[q]
        else:
            var o = s // (KB // 4)
            var p4 = (s % (KB // 4)) * 4
            comptime if PMAJOR:
                comptime for q in range(4):
                    dst[(p4 + q) * ST + o] = v[q]
            else:
                (dst + o * ST + p4).store[alignment=16](v)


@always_inline
def _afn2_bfrag[
    BT: DType
](
    b: MutPointer[Scalar[BT], MutAnyOrigin],
    b_sp: Int,
    b_sj: Int,
    p: Int,
    ke: Int,
    j0: Int,
    n: Int,
) -> _M64:
    """DIRECT_B: this lane's two cells `(frow, fcol + e)` of one B fragment,
    read from device memory: `p` is the step of row `frow`, `j0` the column
    of `fcol`. Cells past `ke` or `n` are +0.0 (their A partner is a staged
    zero at the same step, or the column is never stored)."""
    var v = _M64(0)
    if p < ke:
        comptime for e in range(2):
            if j0 + e < n:
                v[e] = _afn2_widen[BT](b.unsafe_load(p * b_sp + (j0 + e) * b_sj))
    return v


# ===========================================================================
# The kernel
# ===========================================================================


def afn2_gemm_kernel[
    SGM: Int, SGN: Int, FM: Int, FN: Int, KB: Int,
    AT: DType, BT: DType, DIRECT_B: Bool, SWZ: Int,
](
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Scalar[AT], MutAnyOrigin],
    b: MutPointer[Scalar[BT], MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
):
    """`C[m x n] = A_eff[m x k] . B_eff[k x n]` on the matrix unit.
    `A_eff[i, p] = a[i * a_si + p * a_sp]`, `B_eff[p, j] = b[p * b_sp + j *
    b_sj]`. One block per `BM x BN` tile (`SWZ` > 1: grouped block order),
    `SGM x SGN` simdgroups of `FM x FN` fragments, the whole of `k` in one
    f32 accumulator per fragment cell. `DIRECT_B`: B fragments come from
    device memory per lane, only A is staged."""
    comptime NSG = SGM * SGN
    comptime NT = NSG * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    comptime ASZ = KB * AST
    comptime BSZ = 0 if DIRECT_B else BN * BST
    comptime BALLOC = 4 if DIRECT_B else BSZ
    comptime PAGE_BYTES = (ASZ + BSZ) * 4
    comptime NPG = lib_smem_pages_for[COLUMN_APPLE, PAGE_BYTES]()
    comptime RBW = 4 if DIRECT_B else 4 * ((BN * KB) // (4 * NT))
    comptime assert lib_smem_page_fits_for[COLUMN_APPLE, PAGE_BYTES](), (
        "afn2_gemm_kernel: one staged page must fit Apple's threadgroup memory"
    )
    comptime assert lib_smem_page_fits_for[COLUMN_APPLE, NPG * (ASZ + BALLOC) * 4](), (
        "afn2_gemm_kernel: the staged pages must fit Apple's threadgroup memory"
    )
    comptime assert KB % 8 == 0, "afn2_gemm_kernel: whole 8-step fragments"
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var nbm = (m + BM - 1) // BM
    var nbn = (n + BN - 1) // BN
    var bid = Int(block_idx.x)
    var tm: Int
    var tn: Int
    comptime if SWZ > 1:
        # Grouped order: SWZ tile rows, column by column; the last group
        # holds the remaining rows.
        var width = SWZ * nbn
        var first = (bid // width) * SWZ
        var gsz = min(nbm - first, SWZ)
        var r = bid % width
        tm = first + r % gsz
        tn = r // gsz
    else:
        tm = bid // nbn
        tn = bid % nbn
    var m0 = tm * BM
    var n0 = tn * BN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[NPG * ASZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[NPG * BALLOC, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_M64, NF](fill=_M64(0))
    var a_ofast = a_si == 1 and a_sp != 1
    var b_ofast = b_sj == 1 and b_sp != 1
    var ke = k
    var windows = (k + KB - 1) // KB
    # The next window's staged words are in registers while this one
    # multiplies.
    var ra = _afn2_gload[BM, KB, NT, AT](a, a_si, a_sp, m0, m, 0, min(KB, k), tid, a_ofast)
    var rb = SIMD[DType.float32, RBW](0.0)
    comptime if not DIRECT_B:
        rb = rebind[SIMD[DType.float32, RBW]](
            _afn2_gload[BN, KB, NT, BT](b, b_sj, b_sp, n0, n, 0, min(KB, k), tid, b_ofast)
        )
    comptime if NPG == 2:
        # Two pages: window w multiplies from one while w + 1 stages into
        # the other, one barrier per window.
        _afn2_stage[BM, KB, NT, True, AST](at, ra, tid, a_ofast)
        comptime if not DIRECT_B:
            _afn2_stage[BN, KB, NT, False, BST](
                bt, rebind[SIMD[DType.float32, 4 * ((BN * KB) // (4 * NT))]](rb), tid, b_ofast
            )
        barrier()
        if windows > 1:
            ra = _afn2_gload[BM, KB, NT, AT](a, a_si, a_sp, m0, m, KB, min(KB, k - KB), tid, a_ofast)
            comptime if not DIRECT_B:
                rb = rebind[SIMD[DType.float32, RBW]](
                    _afn2_gload[BN, KB, NT, BT](b, b_sj, b_sp, n0, n, KB, min(KB, k - KB), tid, b_ofast)
                )
    for w in range(windows):
        var k0 = w * KB
        var cur = w % NPG
        var atc = at + cur * ASZ
        var btc = bt + cur * BALLOC
        comptime if NPG == 1:
            _afn2_stage[BM, KB, NT, True, AST](at, ra, tid, a_ofast)
            comptime if not DIRECT_B:
                _afn2_stage[BN, KB, NT, False, BST](
                    bt, rebind[SIMD[DType.float32, 4 * ((BN * KB) // (4 * NT))]](rb), tid, b_ofast
                )
            barrier()
            if w + 1 < windows:
                var k1 = k0 + KB
                ra = _afn2_gload[BM, KB, NT, AT](a, a_si, a_sp, m0, m, k1, min(KB, k - k1), tid, a_ofast)
                comptime if not DIRECT_B:
                    rb = rebind[SIMD[DType.float32, RBW]](
                        _afn2_gload[BN, KB, NT, BT](b, b_sj, b_sp, n0, n, k1, min(KB, k - k1), tid, b_ofast)
                    )
        comptime for p8 in range(KB // 8):
            var af = InlineArray[_M64, FM](fill=_M64(0))
            var bf = InlineArray[_M64, FN](fill=_M64(0))
            comptime for fm in range(FM):
                af[fm] = _afn2_load_t(atc + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
            comptime if DIRECT_B:
                var p = k0 + 8 * p8 + frow
                comptime for fq in range(FN):
                    bf[fq] = _afn2_bfrag[BT](
                        b, b_sp, b_sj, p, ke, n0 + (sgn * FN + fq) * 8 + fcol, n
                    )
            else:
                comptime for fq in range(FN):
                    bf[fq] = _afn2_load_t(btc + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
            comptime for fm in range(FM):
                comptime for fq in range(FN):
                    acc[fm * FN + fq] = _afn2_mma(af[fm], bf[fq], acc[fm * FN + fq])
        comptime if NPG == 2:
            if w + 1 < windows:
                var nxt = (w + 1) % NPG
                _afn2_stage[BM, KB, NT, True, AST](at + nxt * ASZ, ra, tid, a_ofast)
                comptime if not DIRECT_B:
                    _afn2_stage[BN, KB, NT, False, BST](
                        bt + nxt * BALLOC,
                        rebind[SIMD[DType.float32, 4 * ((BN * KB) // (4 * NT))]](rb),
                        tid,
                        b_ofast,
                    )
                if w + 2 < windows:
                    var k2 = k0 + 2 * KB
                    ra = _afn2_gload[BM, KB, NT, AT](a, a_si, a_sp, m0, m, k2, min(KB, k - k2), tid, a_ofast)
                    comptime if not DIRECT_B:
                        rb = rebind[SIMD[DType.float32, RBW]](
                            _afn2_gload[BN, KB, NT, BT](b, b_sj, b_sp, n0, n, k2, min(KB, k - k2), tid, b_ofast)
                        )
        barrier()
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var gi = m0 + (sgm * FM + fm) * 8 + frow
                var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                if gi < m and gj < n:
                    c.unsafe_store(gi * n + gj, acc[fm * FN + fq][e])


# ===========================================================================
# The host side
# ===========================================================================


def _afn2_launch[
    SGM: Int, SGN: Int, FM: Int, FN: Int, AT: DType, BT: DType
](
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Scalar[AT], MutAnyOrigin],
    b: MutPointer[Scalar[BT], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_si: Int,
    a_sp: Int,
    b_sp: Int,
    b_sj: Int,
) raises:
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime NT = SGM * SGN * 32
    comptime KB = afn2_kb(BM, BN, NT, AFN_GEMM2_DBUF, AFN_GEMM2_DIRECT_B)
    comptime kern = afn2_gemm_kernel[
        SGM, SGN, FM, FN, KB, AT, BT, AFN_GEMM2_DIRECT_B, AFN_GEMM2_SWZ
    ]
    var tiles = ((m + BM - 1) // BM) * ((n + BN - 1) // BN)
    ctx.enqueue_function[kern](
        c, a, b,
        Int32(m), Int32(n), Int32(k),
        Int32(a_si), Int32(a_sp), Int32(b_sp), Int32(b_sj),
        grid_dim=(tiles, 1, 1),
        block_dim=(NT, 1, 1),
    )


def afn2_gemm_dispatch[
    AT: DType, BT: DType
](
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Scalar[AT], MutAnyOrigin],
    b: MutPointer[Scalar[BT], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`C = op(A) . op(B)` on `afn2_gemm_kernel`, asynchronous, every buffer
    the caller's. True when served; False with the defines off or for an op
    or shape this route does not serve (the wave-1 route takes it)."""
    comptime if not AFN_GEMM2_ON:
        return False
    else:
        if m <= 0 or n <= 0 or k <= 0:
            return False
        if op != OP_NN and op != OP_NT and op != OP_TN:
            return False
        # `gemm_operand_strides`' table (wave-1 `_afn_strides`).
        var a_si = k
        var a_sp = 1
        if op == OP_TN:
            a_si = 1
            a_sp = m
        var b_sp = n
        var b_sj = 1
        if op == OP_NT:
            b_sp = 1
            b_sj = k
        comptime if AFN_GEMM2_BIGTILE:
            var big = ((m + 127) // 128) * ((n + 63) // 64)
            if big >= 2 * AFN_GEMM2_CORES:
                _afn2_launch[4, 2, 4, 4, AT, BT](
                    ctx, c, a, b, m, n, k, a_si, a_sp, b_sp, b_sj
                )
                return True
        _afn2_launch[2, 2, 4, 4, AT, BT](ctx, c, a, b, m, n, k, a_si, a_sp, b_sp, b_sj)
        return True
