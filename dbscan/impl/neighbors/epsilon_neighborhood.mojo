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

DEVIATION BLOCK 27: THE L1 ARM, AND WHY IT COULD NOT BE A BRANCH ON `eps`
-------------------------------------------------------------------------
ORIGINAL WORK. There is no upstream eps-neighborhood kernel to be faithful
to here: `epsilon_neighborhood.cuh` carries the squared-L2 formulation and
nothing else, and cuML's DBSCAN surface never offers Manhattan. So this arm
credits nothing and no `DERIVATION_MAP.tsv` row points anywhere for it. What
IS borrowed, and is cited, is the per-pair arithmetic: RAFT's
`raft/distance/detail/distance_ops/l1.cuh:49`

    DI void core(AccT& acc, DataT& x, DataT& y) const { acc += raft::abs(x - y); };

with `use_norms = false` (`:36`) and an EMPTY epilog (`:51-59`). This
lane already spells that op once, in
`kde/impl/distance/distance_ops.mojo::l1_core`, as
`ftz(acc + abs(ftz(x - y)))`, and this file spells it the same way.

**THE THRESHOLD IS NOT SQUARED ON THIS ARM, AND THAT IS THE WHOLE
STRUCTURAL DIFFERENCE.** The L2 arm never takes a square root: it
accumulates `sum (x-y)^2` and compares against `eps * eps`, squared once on
the host (`vertexdeg/algo.cuh:225`), because squaring one threshold is
cheaper than rooting a million distances and, being monotone on
non-negatives, decides exactly the same neighborhoods. **L1 has no squared
form.** `sum |x-y|` is already the distance, `sqrt` never enters, and there
is no monotone rewriting of the comparison that lets the L1 accumulator
meet an `eps^2`. Comparing an L1 sum against `eps * eps` is not a slower
answer, it is a DIFFERENT and wrong radius (at eps 0.5 it would shrink the
neighborhood to 0.25; at eps 2 it would double it).

So the threshold argument of this kernel is no longer "eps squared". It is
`thresh`, and its meaning is chosen by the `metric` parameter:

    metric = DBSCAN_METRIC_L2   thresh = eps * eps   (theirs, algo.cuh:225)
    metric = DBSCAN_METRIC_L1   thresh = eps

`vertex_deg_run` is the only host site that computes it and it branches on
the same comptime parameter, so the two cannot drift apart. A runtime
`metric` argument was rejected for exactly that reason: it would put a
branch inside the innermost accumulate loop AND make the host's squaring a
separate decision from the kernel's arithmetic.

WHAT THE L1 ARM DOES NOT CHANGE: the tile geometry, the shared-memory
staging, the boundary guards, the `adj` write, the Int32 degree count and
the row reduction below are all shared, ONE copy, compiled twice. The only
comptime-varying statement in the whole kernel is the accumulate helper
`_eps_acc`, which is the port of `core()` and nothing else.

NUMERICS. `acc += abs(diff)` is a PLAIN ADD, not a multiply-add, so
IDENTITY_PATHS row 9 does not apply to it and `identical_mul_add` must NOT
be wrapped around it (that would be a different function, `fma(1, |d|, acc)`,
and would round differently). Row 10 does apply: `diff` is a subtraction of
nearby values, which is where a denormal appears, so both the difference
and the running sum go through `ftz`. The accumulation ORDER is unchanged
from the L2 arm -- the same `kk` walk over the same `Policy4x4` K-blocks --
so the fold is the same pure function of `(k, KBLK)` on every vendor.

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

**AT A LANE WIDTH OF 64 (AMD CDNA), AND WHAT THE L1 ARM DID NOT DO TO IT.**
The L1 arm added above changes the ACCUMULATOR and the THRESHOLD COMPARE
and touches nothing below the `epilog()` line: `sums[i]` is still an Int32
count of neighbors and this butterfly is byte-for-byte the code it was, so
DEVIATION 30's constraint is inherited unchanged rather than re-argued. For
the record of what that constraint is worth at 64, because the tree-wide
claim that "nothing indexes by hardware lane" was found FALSE and deleted
on 2026-09-01 and this file is one of the reasons:

  - The ARITHMETIC survives. 16 divides 64, `acccolid = tid % 16` still
    numbers the 16 threads of an output row consecutively, and a
    `shuffle_xor` at an offset of 1, 2, 4 or 8 only ever flips bits below
    the 16-lane boundary, so the exchange stays inside one aligned 16-lane
    group of the 64-lane wavefront exactly as it stays inside one aligned
    group of a 32-lane warp. The 256-thread block is 4 wavefronts instead
    of 8 warps and every lane reaches every shuffle unconditionally, so
    there is no partial-mask hang either.
  - The REDUCER is `Int32` addition, which is associative and commutative,
    so even a mis-shaped fold could not change the VALUE, only the set
    reduced; and the set is what the paragraph above pins.
  - WHAT IS STILL OWED IS A MEASUREMENT, NOT AN ARGUMENT. No MI325X leg has
    ever run this kernel, so the sentence that opens this block -- correct
    only at a hardware lane width of 32 -- stands as the CERTIFIED claim and
    the 64-lane reasoning above is reasoning. Do not promote it without an
    AMD row.

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

from checks.numerics import ftz_simd, identical_mul_add_simd
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
#: THE METRIC OF THE EPS NEIGHBORHOOD, as a comptime parameter of the
#: kernel. See DEVIATION 27. `DBSCAN_METRIC_L2` is the ported arm
#: (`epsilon_neighborhood.cuh`, unexpanded squared L2, threshold squared on
#: the host); `DBSCAN_METRIC_L1` is original work whose per-pair `core()` is
#: RAFT's `l1.cuh:49` and whose threshold is NOT squared.
comptime DBSCAN_METRIC_L2 = 0
comptime DBSCAN_METRIC_L1 = 1


def dbscan_metric_name(metric: Int) -> String:
    """The metric's public spelling, for messages and for the checks."""
    if metric == DBSCAN_METRIC_L1:
        return String("manhattan")
    return String("euclidean")


def dbscan_metric_threshold(metric: Int, eps: Float64) -> Float32:
    """The number the kernel compares its accumulator against.

    L2: `eps2 = data.eps * data.eps` (`vertexdeg/algo.cuh:225`), because the
    accumulator is a SQUARED distance.
    L1: `eps` itself, because `sum |x-y|` IS the distance and there is no
    squared form of it. DEVIATION 27; getting this wrong is silent and
    changes the radius rather than the speed.

    ONE function, called by the host launcher and by the checks, so the
    fixture in a gate cannot compute the threshold a different way from the
    code it is gating.
    """
    if metric == DBSCAN_METRIC_L1:
        return Float32(eps)
    return Float32(eps * eps)


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


@always_inline
def _eps_acc[
    metric: Int, w: Int
](
    acc: SIMD[DType.float32, w],
    xv: Float32,
    regy: SIMD[DType.float32, w],
) -> SIMD[DType.float32, w]:
    """ONE `core()` op, chosen at compile time. DEVIATION 27.

    L2 (`epsilon_neighborhood.cuh:118-135`, and `l2_unexp.cuh:62-63` is the
    same two lines):  `diff = x - y; acc += diff * diff`, a multiply-add, so
    IDENTITY_PATHS row 9 pins it to `identical_mul_add`.

    L1 (`raft/distance/detail/distance_ops/l1.cuh:49`):
    `acc += raft::abs(x - y)`, a PLAIN add. Row 9 does not apply and
    `identical_mul_add` must not be wrapped around it. Row 10 does: both the
    difference and the running sum go through `ftz`, because a difference of
    nearby coordinates is exactly where a denormal appears in this port.

    This is the ONLY statement in the kernel below that varies with the
    metric. Everything else -- the tile, the staging, the guards, the `adj`
    write, the Int32 degree, the row reduction -- is one copy compiled
    twice.
    """
    # `xv - regy` broadcasts the scalar, which is the expression the L2 arm
    # carried before this helper existed; the shape of the subtraction is
    # unchanged and no L2 bit moves.
    var diff = ftz_simd[w](xv - regy)
    comptime if metric == DBSCAN_METRIC_L1:
        return ftz_simd[w](acc + abs(diff))
    else:
        return ftz_simd[w](identical_mul_add_simd[w](diff, diff, acc))


def eps_unexp_neigh_kernel[
    metric: Int
](
    adj: MutPointer[UInt8, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    thresh_in: Float32,
):
    """`epsUnexpL2SqNeighKernel`: `prolog(); loop(); epilog();`.

    `metric` is `DBSCAN_METRIC_L2` (the ported arm) or `DBSCAN_METRIC_L1`
    (DEVIATION 27, original work). It was called
    `eps_unexp_l2_sq_neigh_kernel` while L2 was the only arm.

    `adj` is `m x n` row major booleans, `vd` is `m + 1` where `vd[m]` is the
    total edge count of the batch -- their layout, and `runner.cuh:281` reads
    exactly that last element back to size the CSR.

    `x` is the BATCH (`m` rows), `y` is the whole dataset (`n` rows), both
    `. x k` row major.

    **`thresh_in` IS NOT ALWAYS `eps` AND IS NOT ALWAYS `eps` SQUARED.** On
    the L2 arm it is ALREADY SQUARED, because their `launcher` squares it
    once on the host (`vertexdeg/algo.cuh:225`) and the accumulator is a
    squared distance. On the L1 arm it is `eps` itself, because an L1 sum
    has no squared form. `dbscan_metric_threshold` is the one function that
    decides which, and the host launcher and the checks both call it rather
    than spelling the arithmetic twice. See DEVIATION 27.

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
            # `auto diff = regx - regy; acc += diff * diff;` -- UNEXPANDED,
            # and under IDENTICAL every step of it is pinned. This is the
            # ONLY float arithmetic in DBSCAN: everything downstream is an
            # Int32 degree, a bool adjacency and a min over labels, so ONE
            # bit here is the difference between a point being a neighbour
            # and not, and from there between two clusters and one. (The
            # WEIGHTED core-point test added later is a second float site,
            # and it is in `vertexdeg/algo.mojo` behind its own pinned fold;
            # this is still the only float arithmetic on the unweighted
            # path.)
            #
            # On the L2 arm `acc += diff * diff` is a multiply-add and
            # therefore IDENTITY_PATHS row 9's shape (one rounding under
            # IDENTICAL, the codegen's choice under FAST). On the L1 arm it
            # is a plain add and row 9 does not apply. `diff` is a
            # SUBTRACTION OF NEARBY VALUES on both arms, which is where a
            # denormal actually appears in this port -- two points a hair
            # apart in one feature -- so row 10's flush is not decorative
            # here either. `_eps_acc` is where both arms live; see
            # DEVIATION 27.
            acc0 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](
                acc0,
                sx[(accrowid + 0 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk],
                regy,
            )
            acc1 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](
                acc1,
                sx[(accrowid + 1 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk],
                regy,
            )
            acc2 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](
                acc2,
                sx[(accrowid + 2 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk],
                regy,
            )
            acc3 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](
                acc3,
                sx[(accrowid + 3 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk],
                regy,
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
            # THE RADIUS TEST. On the L2 arm `a` is a squared distance and
            # `thresh_in` is `eps * eps`; on the L1 arm `a` is the L1
            # distance and `thresh_in` is `eps`. One compare, two meanings,
            # and `dbscan_metric_threshold` is the only place that decides
            # which. DEVIATION 27.
            var is_neigh = a <= thresh_in
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


def eps_unexp_neighborhood[
    metric: Int
](
    ctx: DeviceContext,
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    start_vertex_id: Int,
    m: Int,
    n: Int,
    k: Int,
    thresh: Float32,
) raises:
    """`epsUnexpL2SqNeighborhood` (`epsilon_neighborhood.cuh:218`).

    `metric` picks the arm; `thresh` MUST come from
    `dbscan_metric_threshold(metric, eps)` and never from a hand-written
    `eps * eps`. It was called `eps_unexp_l2_sq_neighborhood` while L2 was
    the only arm.

    Zero `vd`, then launch. `x` is the whole dataset and the FIRST operand is
    the batch slice `x + start_vertex_id * k`, which is theirs
    (`vertexdeg/algo.cuh:230`); the second operand is the whole thing.

    Both operands are the same buffer here, which the old expanded path could
    not do: `gemm_nt` takes `DeviceBuffer` arguments and Mojo refuses one
    buffer as two mutable kernel arguments (`PORTING.md 24`), so the runner
    carried an `x_alias` COPY of the whole dataset and a second copy of its
    norms. `enqueue_function` takes raw pointers, so the fused kernel simply
    reads `x` twice and both copies are gone.

    ON THE L2 ARM `thresh` IS ALREADY SQUARED. Their `launcher` squares it
    (`algo.cuh:225`) and their doc comment on this function says so: "should
    be passed as squared as we compute L2-squared distance in this method".
    ON THE L1 ARM IT IS NOT, because an L1 sum has no squared form; see
    DEVIATION 27 at the top of this file.
    """
    var xb = x.create_sub_buffer[DType.float32](
        start_vertex_id * k, m * k
    )
    # `cudaMemsetAsync(vd, 0, (m + 1) * sizeof(IdxT), stream)` -- the kernel
    # ACCUMULATES into vd. See deviation 31.
    ctx.enqueue_memset(vd, Int32(0))
    ctx.synchronize()
    ctx.enqueue_function[eps_unexp_neigh_kernel[metric]](
        adj.unsafe_ptr(),
        vd.unsafe_ptr(),
        xb.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        thresh,
        grid_dim=(
            (m + EPS_MBLK - 1) // EPS_MBLK,
            (n + EPS_NBLK - 1) // EPS_NBLK,
            1,
        ),
        block_dim=(EPS_THREADS, 1, 1),
    )
