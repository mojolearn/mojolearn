# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The epsilon neighborhood, FUSED: no distance matrix is ever written.

PORT OF `raft/spatial/knn/detail/epsilon_neighborhood.cuh` at RAFT `661a3b8`
(`EpsUnexpL2SqNeighborhood`, `epsUnexpL2SqNeighKernel`,
`epsUnexpL2SqNeighborhood`), built on their
`raft/linalg/detail/contractions.cuh` policy. Do not improve.

This is what `cuml/cpp/src/dbscan/vertexdeg/algo.cuh:229` actually calls for
the brute-force arm:

    raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood<value_t,
      index_t>(data.adj, data.vd, data.x + start_vertex_id * k, data.x,
               n, m, k, eps2, stream);

WHY THIS FILE EXISTS: THEIRS IS FUSED AND OURS WAS NOT
------------------------------------------------------
`EpsUnexpL2SqNeighborhood` is a `Contractions_NT` tile kernel. It accumulates
`acc[i][j]` in REGISTERS (`epsilon_neighborhood.cuh:41`), tests
`acc[i][j] <= eps` in its `epilog()` (`:106`), writes the boolean `adj` and
reduces the vertex degrees with `logicalWarpReduce` + `blockReduce` +
atomics, ALL IN THE SAME KERNEL (`:137-160`). **It never materializes a float
distance matrix.**

What this repository did instead was `gemm_nt` into an `m x N` float32
`dist` buffer, then `expand_distances_kernel` over that buffer, then a third
kernel that read it back to threshold it. Per batch that is one `m*N` float
write, one `m*N` float read-modify-write, and one `m*N` float read that
upstream does not perform at all: 16 bytes of traffic per pair against
their 1. The measured consequence is in `bench/results/FIRST_RUN_2026-08-19.md`
and the scaling run after it -- 0.021x of scikit-learn at 100,000 rows, on a
problem where the arithmetic is 8 multiply-adds per pair.

THEIR L2 IS UNEXPANDED AND OURS WAS EXPANDED, WHICH IS AN ARITHMETIC CHANGE
---------------------------------------------------------------------------
`accumulate()` (`:118-135`) is

    auto diff = this->regx[i][v] - this->regy[j][v];
    acc[i][j] += diff * diff;

computed straight from the coordinates. The old path used the expanded
identity `||x||^2 + ||y||^2 - 2 x.y` so that the cross term could go through
a GEMM. That is not a different route to the same number: when the norms
dominate the distances, the subtraction cancels and float32 loses the
answer. This tree has already paid for that once -- `PORTING.md 21`, and the
comment in `dbscan_check.mojo` that keeps the fixture coordinates small on
purpose. Unexpanded needs no norms, so `row_norm_kernel`, `x_norm` and
`xn_alias` leave the DBSCAN path entirely.

THEIR POLICY
------------
`epsUnexpL2SqNeighImpl` instantiates `raft::linalg::Policy4x4<DataT, VecLen>`
(`:192`), which for float is `KernelPolicy<float, VecLen, 32, 4, 4, 16, 16>`
(`raft/linalg/contractions.cuh:119`): Kblk 32, a 4x4 output block per thread,
256 threads, a 64x64 output tile.

Their `Veclen` selection at the bottom of the file (`:230-237`) picks 4 for
float whenever `4 * k` is a multiple of 16. Ours loads one float at a time,
which is the same gap `core/gemm.mojo` records for the GEMM; `Veclen` still
sets `SmemStride` here, so their bank-conflict padding is kept.

DEVIATION BLOCK 30: THE ROW REDUCTION IS A BUTTERFLY, NOT A ROTATE
------------------------------------------------------------------
THEIRS: `raft::logicalWarpReduce<P::AccThCols>(sums[i], raft::add_op())`
(`:150`) reduces across the 16 threads that share an output row using
`shfl_xor` inside a width-16 logical warp.
OURS: `shuffle_xor` at offsets 1, 2, 4, 8. Mojo's `shuffle_idx` has no width
argument (`PORTING.md`, the `simt_kernel.mojo` note), but `shuffle_xor` needs
none: XOR with an offset below `AccThCols` only ever flips the low lane bits,
so it stays inside the same aligned 16-lane group their logical warp is, and
every lane of the group ends holding the group's sum.
REASON: not a performance choice and not a numerical one. The reducer is
integer addition, which is associative and commutative, so no reduction shape
can change the result. `AccThCols = 16` is a power of two, `acccolid =
tid % 16`, and the lane width is 32, so the 16 threads sharing a row are
contiguous and aligned and never straddle a warp boundary. Change any of
those three and this silently reduces the wrong set.

DEVIATION BLOCK 31: `vd` IS ZEROED BY `enqueue_memset`, NOT `cudaMemsetAsync`
-----------------------------------------------------------------------------
THEIRS: `epsUnexpL2SqNeighborhood` opens with
`cudaMemsetAsync(vd, 0, (m + 1) * sizeof(IdxT), stream)` (`:228`).
OURS: the caller does it with `ctx.enqueue_memset` on the sub-buffer.
REASON: same operation, and MAX has no in-kernel counterpart to hoist it
into. It is called out because the kernel ACCUMULATES into `vd` and produces
garbage without it -- the old unfused kernel ASSIGNED, so this is a new
precondition, not an inherited one.
"""

from original.numerics import ftz_simd, identical_mul_add_simd
from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


# `typedef typename raft::linalg::Policy4x4<DataT, VecLen>::Policy Policy;`
# (`epsilon_neighborhood.cuh:192`), which for float is
# `KernelPolicy<float, VecLen, 32, 4, 4, 16, 16>`
# (`raft/linalg/contractions.cuh:119`). The derived members are their
# arithmetic verbatim (`contractions.cuh:60-101`).
#
# These are restated here rather than imported from `core/gemm.mojo`, whose
# copy of the same policy belongs to the matmul wrapper and is another lane's
# to change. A kernel whose correctness depends on `AccThCols` being 16 (the
# `shuffle_xor` group width, deviation 30) must not read that 16 out of a file
# that is free to retune it.
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

# `Policy::Nthreads`, the launch's `block_dim.x`. Named so callers do not have
# to know the policy's internals.
comptime EPS_THREADS = GEMM_THREADS
comptime EPS_MBLK = GEMM_MBLK
comptime EPS_NBLK = GEMM_NBLK


def eps_unexp_l2_sq_neigh_kernel(
    adj: MutPointer[UInt8, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    eps_in: Float32,
):
    """`epsUnexpL2SqNeighKernel`: `prolog(); loop(); epilog();`.

    `adj` is `m x n` row major booleans, `vd` is `m + 1` where `vd[m]` is the
    total edge count of the batch -- their layout, and `runner.cuh:281` reads
    exactly that last element back to size the CSR.

    `x` is the BATCH (`m` rows), `y` is the whole dataset (`n` rows), both
    `. x k` row major, and `eps_in` is ALREADY SQUARED: their `launcher`
    squares it once on the host (`vertexdeg/algo.cuh:225`) and passes `eps2`.

    Launch `grid_dim = (ceil(m / EPS_MBLK), ceil(n / EPS_NBLK), 1)` and
    `block_dim = (EPS_THREADS, 1, 1)`, matching `epsUnexpL2SqNeighImpl:193`
    -- grid.x tiles the ROWS and grid.y tiles the COLUMNS, which is the
    opposite of `gemm_nt_kernel`'s convention and is theirs.

    **`vd` must be zeroed for `m + 1` elements first.** See deviation 31.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)

    var tid = Int(thread_idx.x)
    # `Contractions_NT`'s ctor: accrowid = threadIdx.x / P::AccThCols,
    # acccolid = threadIdx.x % P::AccThCols (`contractions.cuh:99-100`).
    var accrowid = tid // GEMM_ACC_TH_COLS
    var acccolid = tid % GEMM_ACC_TH_COLS

    var m0 = Int(block_idx.x) * GEMM_MBLK
    var n0 = Int(block_idx.y) * GEMM_NBLK

    var sx = stack_allocation[
        GEMM_SMEM_PAGE_X,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sy = stack_allocation[
        GEMM_SMEM_PAGE_Y,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # `DataT acc[P::AccRowsPerTh][P::AccColsPerTh]`, in REGISTERS
    # (`epsilon_neighborhood.cuh:41`). SIMD values and not
    # `stack_allocation`: without an address space that is thread-local
    # MEMORY, which is what made the first register-tiled GEMM here slower
    # than the naive one (`PORTING.md 26`).
    var acc0 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
    var acc1 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
    var acc2 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
    var acc3 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)

    # --- `loop()`: ldgXY, stsXY, accumulate over each k-block ------------
    # Single-buffered. Theirs double-buffers (`switch_write_buffer` /
    # `switch_read_buffer`), which two pages at this policy would cost 32 KB
    # of threadgroup memory -- exactly Metal's ceiling. That gap is recorded
    # once, in `core/gemm.mojo`, and is the same gap here.
    var kt = 0
    while kt < k:
        var e = tid
        while e < GEMM_MBLK * GEMM_KBLK:
            var r = e // GEMM_KBLK
            var c = e % GEMM_KBLK
            var v = Float32(0.0)
            if m0 + r < m and kt + c < k:
                v = x.unsafe_load((m0 + r) * k + kt + c)
            sx[r * GEMM_SMEM_STRIDE + c] = v
            e += EPS_THREADS
        e = tid
        while e < GEMM_NBLK * GEMM_KBLK:
            var r2 = e // GEMM_KBLK
            var c2 = e % GEMM_KBLK
            var v2 = Float32(0.0)
            if n0 + r2 < n and kt + c2 < k:
                v2 = y.unsafe_load((n0 + r2) * k + kt + c2)
            sy[r2 * GEMM_SMEM_STRIDE + c2] = v2
            e += EPS_THREADS
        barrier()

        # `ldsX`/`ldsY` are STRIDED, not blocked: row `accrowid + i *
        # AccThRows` and column `acccolid + j * AccThCols`
        # (`contractions.cuh:285` and `:305`). That is what makes the `adj`
        # writes below land on consecutive addresses for consecutive
        # threads.
        for kk in range(GEMM_KBLK):
            var regy = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
            for j in range(GEMM_ACC_COLS_PER_TH):
                regy[j] = sy[
                    (acccolid + j * GEMM_ACC_TH_COLS) * GEMM_SMEM_STRIDE + kk
                ]
            regy = ftz_simd[GEMM_ACC_COLS_PER_TH](regy)
            # `auto diff = regx - regy; acc += diff * diff;` -- UNEXPANDED.
            # `auto diff = regx - regy; acc += diff * diff;` -- UNEXPANDED,
            # and under IDENTICAL every step of it is pinned. This is the
            # ONLY float arithmetic in DBSCAN: everything downstream is an
            # Int32 degree, a bool adjacency and a min over labels, so ONE
            # bit here is the difference between a point being a neighbour
            # and not, and from there between two clusters and one.
            #
            # `acc += diff * diff` is a multiply-add and therefore
            # IDENTITY_PATHS row 9's shape (one rounding under IDENTICAL,
            # the codegen's choice under FAST). `diff` is a SUBTRACTION OF
            # NEARBY VALUES, which is where a denormal actually appears in
            # this port -- two points a hair apart in one feature -- so
            # row 10's flush is not decorative here either.
            var d0 = ftz_simd[GEMM_ACC_COLS_PER_TH](
                sx[
                    (accrowid + 0 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk
                ]
                - regy
            )
            acc0 = ftz_simd[GEMM_ACC_COLS_PER_TH](
                identical_mul_add_simd[GEMM_ACC_COLS_PER_TH](d0, d0, acc0)
            )
            var d1 = ftz_simd[GEMM_ACC_COLS_PER_TH](
                sx[
                    (accrowid + 1 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk
                ]
                - regy
            )
            acc1 = ftz_simd[GEMM_ACC_COLS_PER_TH](
                identical_mul_add_simd[GEMM_ACC_COLS_PER_TH](d1, d1, acc1)
            )
            var d2 = ftz_simd[GEMM_ACC_COLS_PER_TH](
                sx[
                    (accrowid + 2 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk
                ]
                - regy
            )
            acc2 = ftz_simd[GEMM_ACC_COLS_PER_TH](
                identical_mul_add_simd[GEMM_ACC_COLS_PER_TH](d2, d2, acc2)
            )
            var d3 = ftz_simd[GEMM_ACC_COLS_PER_TH](
                sx[
                    (accrowid + 3 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk
                ]
                - regy
            )
            acc3 = ftz_simd[GEMM_ACC_COLS_PER_TH](
                identical_mul_add_simd[GEMM_ACC_COLS_PER_TH](d3, d3, acc3)
            )
        barrier()
        kt += GEMM_KBLK

    # --- `epilog()` (`epsilon_neighborhood.cuh:93-116`) -------------------
    # The radius test happens HERE, on the accumulator, before it is
    # discarded. Nothing float ever reaches global memory.
    var startx = m0 + accrowid
    var starty = n0 + acccolid

    var sums = SIMD[DType.int32, GEMM_ACC_ROWS_PER_TH](0)
    for i in range(GEMM_ACC_ROWS_PER_TH):
        var xid = startx + i * GEMM_ACC_TH_ROWS
        var s = Int32(0)
        for j in range(GEMM_ACC_COLS_PER_TH):
            var yid = starty + j * GEMM_ACC_TH_COLS
            var a = acc0[j]
            if i == 1:
                a = acc1[j]
            elif i == 2:
                a = acc2[j]
            elif i == 3:
                a = acc3[j]
            var is_neigh = a <= eps_in
            if xid < m and yid < n:
                adj.unsafe_store(
                    xid * n + yid, UInt8(1) if is_neigh else UInt8(0)
                )
                if is_neigh:
                    s += Int32(1)
        sums[i] = s

    # --- `updateVertexDegree()` (`:137-160`) ------------------------------
    barrier()  # their `__syncthreads()` before the reductions
    var cidx = m0 + accrowid
    var total_sum = Int32(0)
    for i in range(GEMM_ACC_ROWS_PER_TH):
        var cid = cidx + i * GEMM_ACC_TH_ROWS
        # `logicalWarpReduce<P::AccThCols>` -- see deviation 30. Every
        # thread reaches every shuffle: the bound is a comptime constant and
        # nothing here is conditional, which is required because a lane that
        # skips a full-mask shuffle hangs the lanes that reach it.
        var s2 = sums[i]
        var off = 1
        while off < GEMM_ACC_TH_COLS:
            s2 += shuffle_xor(s2, UInt32(off))
            off *= 2
        if acccolid == 0 and cid < m:
            _ = Atomic.fetch_add(vd.unsafe_offset(cid), s2)
            total_sum += s2
        barrier()  # theirs, "for safe smem reuse"

    # `totalSum = raft::blockReduce<IdxT>(totalSum, smem);` then
    # `if (threadIdx.x == 0) atomicUpdate(this->m, totalSum);` -- `vd[m]` is
    # the batch's total edge count and `runner.cuh:281` reads it back.
    var block_total = block_sum[block_size=EPS_THREADS](total_sum)
    if tid == 0:
        _ = Atomic.fetch_add(vd.unsafe_offset(m), block_total)


def eps_unexp_l2_sq_neighborhood(
    ctx: DeviceContext,
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    start_vertex_id: Int,
    m: Int,
    n: Int,
    k: Int,
    eps: Float32,
) raises:
    """`epsUnexpL2SqNeighborhood` (`epsilon_neighborhood.cuh:218`).

    Zero `vd`, then launch. `x` is the whole dataset and the FIRST operand is
    the batch slice `x + start_vertex_id * k`, which is theirs
    (`vertexdeg/algo.cuh:230`); the second operand is the whole thing.

    Both operands are the same buffer here, which the old expanded path could
    not do: `gemm_nt` takes `DeviceBuffer` arguments and Mojo refuses one
    buffer as two mutable kernel arguments (`PORTING.md 24`), so the runner
    carried an `x_alias` COPY of the whole dataset and a second copy of its
    norms. `enqueue_function` takes raw pointers, so the fused kernel simply
    reads `x` twice and both copies are gone.

    `eps` is ALREADY SQUARED. Their `launcher` squares it (`algo.cuh:225`)
    and their doc comment on this function says so: "should be passed as
    squared as we compute L2-squared distance in this method".
    """
    var xb = x.create_sub_buffer[DType.float32](
        start_vertex_id * k, m * k
    )
    # `cudaMemsetAsync(vd, 0, (m + 1) * sizeof(IdxT), stream)` -- the kernel
    # ACCUMULATES into vd. See deviation 31.
    ctx.enqueue_memset(vd, Int32(0))
    ctx.synchronize()
    ctx.enqueue_function[eps_unexp_l2_sq_neigh_kernel](
        adj.unsafe_ptr(),
        vd.unsafe_ptr(),
        xb.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        eps,
        grid_dim=(
            (m + EPS_MBLK - 1) // EPS_MBLK,
            (n + EPS_NBLK - 1) // EPS_NBLK,
            1,
        ),
        block_dim=(EPS_THREADS, 1, 1),
    )
