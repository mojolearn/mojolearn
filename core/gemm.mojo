# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The matrix product."""

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.gram_splitk import (
    GRAM_MAX_CELLS_PER_THREAD,
    GRAM_MAX_COLS,
    GRAM_TPB,
    gemm_tn_splitk_into,
    gram_splitk_applies,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)
from checks.rtf_seam import rtf_mul_add
from gemm.checks.gemm_identical import (
    APPLE_MMA,
    GEMM_ADMIT_EXP_SUM,
    _AMMA_M64,
    _admit_warp_min,
    _amma_gload,
    _amma_load_t,
    _amma_mma,
    _amma_stage,
)
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from core.gram_multi_gpu import pinned_gemm_nt_gram_kernel, parallel_gram_outputs

from gemm.checks.gemm_identical import (
    identical_gemm,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NT, OP_TN




def pinned_gemm_nt_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """`z[m x n] = x[m x k] .

    ML-Sys Note (2026-09-18): 
    Attempts to vectorize this kernel using `SIMD[DType.float32, 4]` loads and
    FMAs resulted in a 15+ minute compiler hang on the NVPTX backend (NVIDIA A100/A6000). 
    Because GPUs lack native 128-bit SIMD ALUs akin to CPU AVX, the LLVM backend 
    attempts pathological register unrolling to emulate the vectors, breaking the build.
    Bitwise determinism must be achieved via strictly scalar exact-order accumulation.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell % n
    var acc = Float32(0.0)
    for p in range(k):
        acc = rtf_mul_add(
            ftz(x.unsafe_load(i * k + p)),
            ftz(y.unsafe_load(j * k + p)),
            acc,
        )
    z.unsafe_store(cell, ftz(Float32(0.0) + ftz(acc)))





#: EVERY GPU since lane/gap-classical-nv (2026-10-02), the name kept:
#: NVIDIA and AMD ran `pinned_gemm_nt_kernel`, one thread per cell with
#: untiled global loads (Ridge's U = A V at 1M x 220 x 220, PCA's
#: transform). The tiled kernel is plain shared memory and barriers; the
#: simdgroup-matrix arm (`APPLE_GEMM_NT_MMA`) stays Apple-only via APPLE_MMA.
comptime APPLE_GEMM_NT_TILED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_APPLE_GEMM_NT_TILED_OFF"]()
)
"""IDENTICAL on Apple: `gemm_nt` stages x and y tiles in threadgroup memory
and gives each thread RM x RN cells. Every cell is still ONE chain,
`acc = rtf_mul_add(ftz(x[i, p]), ftz(y[j, p]), acc)` for p = 0 .. k-1 in
order, closed by `ftz(0 + ftz(acc))`: the pinned kernel's bits exactly.
Only which thread owns a chain and where its operands are read from move."""

comptime GNT_KT = 16


def apple_gemm_nt_tiled_kernel[TM: Int, TN: Int, RM: Int, RN: Int](
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    comptime TX = TN // RN
    comptime THREADS = (TM // RM) * TX
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var ty_ = tid // TX
    var tx_ = tid - ty_ * TX
    var i0 = Int(block_idx.x) * TM
    var j0 = Int(block_idx.y) * TN
    var xs = stack_allocation[
        TM * GNT_KT, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var ys = stack_allocation[
        TN * GNT_KT, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var acc = InlineArray[Float32, RM * RN](fill=Float32(0.0))
    var p0 = 0
    while p0 < k:
        var kt = min(GNT_KT, k - p0)
        var t = tid
        while t < TM * GNT_KT:
            var r = t // GNT_KT
            var c = t - r * GNT_KT
            var gi = i0 + r
            xs[t] = ftz(x[gi * k + p0 + c]) if (gi < m and c < kt) else Float32(0.0)
            t += THREADS
        t = tid
        while t < TN * GNT_KT:
            var r = t // GNT_KT
            var c = t - r * GNT_KT
            var gj = j0 + r
            ys[t] = ftz(y[gj * k + p0 + c]) if (gj < n and c < kt) else Float32(0.0)
            t += THREADS
        barrier()
        for pp in range(kt):
            var a = InlineArray[Float32, RM](fill=Float32(0.0))
            var b = InlineArray[Float32, RN](fill=Float32(0.0))
            comptime for rr in range(RM):
                a[rr] = xs[(ty_ * RM + rr) * GNT_KT + pp]
            comptime for cc in range(RN):
                b[cc] = ys[(tx_ * RN + cc) * GNT_KT + pp]
            comptime for rr in range(RM):
                comptime for cc in range(RN):
                    acc[rr * RN + cc] = rtf_mul_add(a[rr], b[cc], acc[rr * RN + cc])
        barrier()
        p0 += GNT_KT
    comptime for rr in range(RM):
        comptime for cc in range(RN):
            var i = i0 + ty_ * RM + rr
            var j = j0 + tx_ * RN + cc
            if i < m and j < n:
                z[i * n + j] = ftz(Float32(0.0) + ftz(acc[rr * RN + cc]))


comptime APPLE_GEMM_NT_MMA = (
    APPLE_GEMM_NT_TILED
    and APPLE_MMA
    and not is_defined["MOJOLEARN_APPLE_GEMM_NT_MMA_OFF"]()
)
"""IDENTICAL on Apple (lane/apple-identical-neural, 2026-09-26): `gemm_nt`
for m, n >= 64 on the simdgroup matrix unit. Every cell is still the one
chain `acc = rtf_mul_add(ftz(x[i, p]), ftz(y[j, p]), acc)`, p ascending,
closed by `ftz(0 + ftz(acc))`. A 16-step window runs as matrix steps when
the GEMM's window admission holds (flushed operand exponent fields
`Ea + Eb >= 174` over the window's staged words, and every earlier window
of the chain admitted, so every entering accumulator is a multiple of
2^-126): then no step result can be subnormal and the bare FMA, which the
M4's matrix chain reproduces bit for bit, IS `rtf_mul_add`. Any other
window, every later one, and a ragged tail run `rtf_mul_add` cell by cell.
`-D MOJOLEARN_APPLE_GEMM_NT_MMA_OFF` reverts to the tiled kernel."""


def apple_gemm_nt_mma_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    comptime SGM = 2
    comptime SGN = 2
    comptime FM = 4
    comptime FN = 4
    comptime KB = 16
    comptime NSG = SGM * SGN
    comptime NT = NSG * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var m0 = Int(block_idx.x) * BM
    var n0 = Int(block_idx.y) * BN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[KB * AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wmin = stack_allocation[2 * NSG, Scalar[DType.uint32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var exact_ok = True
    var windows = (k + KB - 1) // KB
    var ra = _amma_gload[BM, KB, NT](x, k, 1, m0, m, 0, min(KB, k), tid, False)
    var rb = _amma_gload[BN, KB, NT](y, k, 1, n0, n, 0, min(KB, k), tid, False)
    for w in range(windows):
        var k0 = w * KB
        var chunk = min(KB, k - k0)
        var ea = _amma_stage[BM, KB, NT, True, AST](at, ra, tid, False)
        var eb = _amma_stage[BN, KB, NT, False, BST](bt, rb, tid, False)
        ea = _admit_warp_min(ea)
        eb = _admit_warp_min(eb)
        if lane == 0:
            wmin[sg] = ea
            wmin[NSG + sg] = eb
        barrier()
        if w + 1 < windows:
            var k1 = k0 + KB
            ra = _amma_gload[BM, KB, NT](x, k, 1, m0, m, k1, min(KB, k - k1), tid, False)
            rb = _amma_gload[BN, KB, NT](y, k, 1, n0, n, k1, min(KB, k - k1), tid, False)
        var bea = UInt32(0xFF)
        var beb = UInt32(0xFF)
        comptime for q in range(NSG):
            bea = min(bea, wmin[q])
            beb = min(beb, wmin[NSG + q])
        var admitted = exact_ok and chunk == KB and (bea + beb) >= UInt32(GEMM_ADMIT_EXP_SUM)
        if not admitted:
            exact_ok = False
        if admitted:
            comptime for p8 in range(KB // 8):
                var af = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
                var bf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
                comptime for fm in range(FM):
                    af[fm] = _amma_load_t(at + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
                comptime for fq in range(FN):
                    bf[fq] = _amma_load_t(bt + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                comptime for fm in range(FM):
                    comptime for fq in range(FN):
                        acc[fm * FN + fq] = _amma_mma(af[fm], bf[fq], acc[fm * FN + fq])
        else:
            for p in range(chunk):
                comptime for fm in range(FM):
                    var av = at[p * AST + (sgm * FM + fm) * 8 + frow]
                    comptime for fq in range(FN):
                        comptime for e in range(2):
                            var bv = bt[((sgn * FN + fq) * 8 + fcol + e) * BST + p]
                            acc[fm * FN + fq][e] = rtf_mul_add(av, bv, acc[fm * FN + fq][e])
        barrier()
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var i = m0 + (sgm * FM + fm) * 8 + frow
                var j = n0 + (sgn * FN + fq) * 8 + fcol + e
                if i < m and j < n:
                    comptime if is_defined["MOJOLEARN_APPLE_GEMM_NT_MMA_SABOTAGE"]():
                        z[i * n + j] = ftz(Float32(0.0) + ftz(acc[fm * FN + fq][e])) * Float32(1.0000001)
                    else:
                        z[i * n + j] = ftz(Float32(0.0) + ftz(acc[fm * FN + fq][e]))


def pinned_gemv_n_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    k_in: Int32,
):
    """`z[m] = x[m x k] ."""
    var m = Int(m_in)
    var k = Int(k_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m:
        return
    var acc = Float32(0.0)
    for p in range(k):
        acc = rtf_mul_add(
            ftz(x.unsafe_load(i * k + p)), ftz(y.unsafe_load(p)), acc
        )
    z.unsafe_store(i, ftz(Float32(0.0) + ftz(acc)))


comptime GEMM_IDENT_SWAP_537 = is_defined["MOJOLEARN_537_GEMM_IDENT_SWAP"]()


comptime PINNED_GEMM_TPB = 256


comptime GEMM_VECLEN = 4
comptime GEMM_KBLK = 32
comptime GEMM_ACC_ROWS_PER_TH = 4
comptime GEMM_ACC_COLS_PER_TH = 4
comptime GEMM_ACC_TH_ROWS = 16
comptime GEMM_ACC_TH_COLS = 16

comptime GEMM_THREADS = GEMM_ACC_TH_ROWS * GEMM_ACC_TH_COLS
comptime GEMM_MBLK = GEMM_ACC_ROWS_PER_TH * GEMM_ACC_TH_ROWS
comptime GEMM_NBLK = GEMM_ACC_COLS_PER_TH * GEMM_ACC_TH_COLS
comptime GEMM_SMEM_STRIDE = GEMM_KBLK + GEMM_VECLEN
comptime GEMM_SMEM_PAGE_X = GEMM_SMEM_STRIDE * GEMM_MBLK
comptime GEMM_SMEM_PAGE_Y = GEMM_SMEM_STRIDE * GEMM_NBLK


def gemm_nt(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[m x k] ."""
    if n == 1:
        gemv_n(ctx, z, x, y, m, k)
        return
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        comptime if GEMM_IDENT_SWAP_537:
            identical_gemm(ctx, z, x, y, m, n, k, OP_NT)
            return
        comptime if APPLE_GEMM_NT_TILED:
            if n <= 4:
                pass
            elif n <= 16:
                ctx.enqueue_function[apple_gemm_nt_tiled_kernel[256, 16, 4, 4]](
                    z.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(),
                    Int32(m), Int32(n), Int32(k),
                    grid_dim=((m + 255) // 256, (n + 15) // 16, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                var mma = False
                comptime if APPLE_GEMM_NT_MMA:
                    if m >= 64 and n >= 64:
                        mma = True
                        ctx.enqueue_function[apple_gemm_nt_mma_kernel](
                            z.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(),
                            Int32(m), Int32(n), Int32(k),
                            grid_dim=((m + 63) // 64, (n + 63) // 64, 1),
                            block_dim=(128, 1, 1),
                        )
                if not mma:
                    ctx.enqueue_function[apple_gemm_nt_tiled_kernel[64, 64, 4, 4]](
                        z.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(),
                        Int32(m), Int32(n), Int32(k),
                        grid_dim=((m + 63) // 64, (n + 63) // 64, 1),
                        block_dim=(256, 1, 1),
                    )
            if n > 4:
                return
        ctx.enqueue_function[pinned_gemm_nt_kernel](
            z.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=((m * n + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        return
    var tz = TileTensor(z, row_major(m, n))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(n, k))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)


def gemm_nt_gram(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    xt: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = xt[m x k] ."""
    if n == 1:
        raise Error(
            "gemm_nt_gram: n == 1 is not a Gram shape this entry serves."
            " gemm_nt's gemv route takes two mut buffers and cannot be"
            " handed one buffer twice; a 1 x 1 Gram is a dot product and"
            " belongs somewhere else."
        )
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var xtm = xt
        if m == n and parallel_gram_outputs[False](ctx,z,xtm,m,k):
            return
        ctx.enqueue_function[pinned_gemm_nt_gram_kernel](
            z.unsafe_ptr(),
            xtm.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(0),
            Int32(0),  # full original extent; do not narrow m*n to Int32
            grid_dim=((m * n + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        return
    var tz = TileTensor(z, row_major(m, n))
    var tx = TileTensor(xt, row_major(m, k))
    var ty = TileTensor(xt, row_major(n, k))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)


def gemm_tn(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32],
    mut xt2: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[k x m]^T .

    IDENTICAL: the split-K Gram kernel where it applies (`m == n <= 128`,
    the shipped OLS/PCA shapes, bits unchanged), and past its capacity the
    v1 profile's OP_TN arm (`identical_gemm_into`, one operand handed in
    twice) instead of the refusal that stood here until 2026-09-09. Both
    arms are pinned; they are different profiles, so a shape that crosses
    128 features changes profile, not vendor-dependence.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if not gram_splitk_applies(m, n, k):
            if m != n:
                raise Error(
                    "gemm_tn: NUMERIC_IDENTICAL refuses "
                    + String(m)
                    + " x "
                    + String(n)
                    + " x "
                    + String(k)
                    + ": this entry is the Gram case x^T . x with one"
                    + " operand and one width, so m must equal n."
                    + " IDENTITY_PATHS row 27."
                )
            gemm_tn_identical_v1(ctx, z, x, xt2, m, k)
            return
    if gram_splitk_applies(m, n, k):
        gemm_tn_splitk_into(ctx, z, x, xt, m, k)
        return
    gemm_tn_via_transpose(ctx, z, x, xt, xt2, m, n, k)


def gemm_tn_identical_v1(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut scratch: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`z[m x m] = x[k x m]^T . x[k x m]` on profile
    mojolearn.identical.gemm.fp32.v1 (OP_TN). `scratch` is `gemm_tn`'s
    `xt2` alias buffer (at least `k * m` floats) and serves as v1's
    workspace when it covers the plan; every plan v1 picks past the
    split-K capacity is a fused tile plan needing no workspace at all, so
    the allocate-and-wait branch below is the guard, not the path.
    """
    if parallel_gram_outputs[True](ctx,z,x,m,k):
        return
    var x2 = x
    var need = identical_gemm_workspace_max_floats(m, m, k)
    if need <= k * m:
        identical_gemm_into(ctx, z, x, x2, scratch, m, m, k, OP_TN)
        return
    var ws = ctx.enqueue_create_buffer[DType.float32](need)
    ctx.synchronize()
    identical_gemm_into(ctx, z, x, x2, ws, m, m, k, OP_TN)
    ctx.synchronize()
    _ = ws^


def gemm_tn_via_transpose(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32],
    mut xt2: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[k x m]^T ."""
    from core.column_stats import (
        CUDA_MAX_GRID_YZ,
        TRANSPOSE_TILE,
        transpose_kernel,
    )

    if m != n:
        raise Error(
            "gemm_tn_via_transpose: m="
            + String(m)
            + " n="
            + String(n)
            + ". This entry is the GRAM case: one operand, one width. Its"
            " transpose is built at width m, and reading that block at width"
            " n runs off it. A genuine two-operand TN needs a second"
            " transpose at width n, which this does not do."
        )

    ctx.enqueue_function[transpose_kernel](
        xt.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(k),
        Int32(m),
        grid_dim=(
            (m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            min((k + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, CUDA_MAX_GRID_YZ),
            1,
        ),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    gemm_nt_gram(ctx, z, xt, m, n, k)
    _ = xt2



from linalg.gemv import gemv_gpu


comptime GEMV_FAST_PINNED_MAX_K = 64


def gemv_n(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`z[m] = x[m x k] ."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        ctx.enqueue_function[pinned_gemv_n_kernel](
            z.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            Int32(m),
            Int32(k),
            grid_dim=((m + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        return
    # FAST on Apple, k <= GEMV_FAST_PINNED_MAX_K: the one-thread-per-row
    # kernel beats the vendor gemv on narrow matrices.
    # `-D MOJOLEARN_GEMV_FAST_VENDOR` keeps the vendor gemv.
    comptime if (
        has_apple_gpu_accelerator()
        and not is_defined["MOJOLEARN_GEMV_FAST_VENDOR"]()
    ):
        if k <= GEMV_FAST_PINNED_MAX_K:
            ctx.enqueue_function[pinned_gemv_n_kernel](
                z.unsafe_ptr(),
                x.unsafe_ptr(),
                y.unsafe_ptr(),
                Int32(m),
                Int32(k),
                grid_dim=((m + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
                block_dim=(PINNED_GEMM_TPB, 1, 1),
            )
            return
    var tz = TileTensor(z, row_major(m, Int(1)))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(k, Int(1)))
    gemv_gpu[transpose_b=False](tz, tx, ty, ctx)
