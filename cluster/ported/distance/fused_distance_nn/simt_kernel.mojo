"""Distance and argmin FUSED, so the distance matrix is never written.

PORT OF `cuvs/src/distance/detail/fused_distance_nn/simt_kernel.cuh` at cuVS
`94c2819`, built on their `linalg/contractions.cuh` policy and the
`PairwiseDistances` loop structure of
`raft/distance/detail/pairwise_distance_base.cuh`. Partial.
Do not improve.

WHY THIS EXISTS AND THE UNFUSED PATH WAS NOT ENOUGH
---------------------------------------------------
`unfused_distance_nn.mojo` computes the whole `m x n` distance tile into
memory and then reduces it. This kernel never writes it. Each thread holds a
running `(value, key)` minimum for each of the `AccRowsPerTh` rows it owns,
in REGISTERS, and updates it as each column tile's accumulator is completed:

    KVPair val[P::AccRowsPerTh];        // simt_kernel.cuh:78

That is the design, not a tuning of it. At k-means' shape, 200,000 rows and
16 clusters, the unfused path writes 3.2 million floats to global memory and
reads them straight back, every single Lloyd iteration, to extract one
minimum per row. This kernel does the same arithmetic and moves none of it.

**I ported the unfused arm first and justified it by their own selector
preferring unfused on Blackwell, which is true and was not the point.** The
fused arm is what runs on the hardware most people have, and its CUTLASS
version is unportable but THIS one is not: `simt_kernel.cuh` is the SIMT
path, plain CUDA cores, no tensor-core instructions anywhere.

THE POLICY IS A PARAMETER AND THE SELECTION IS THEIRS
------------------------------------------------------
The kernel is parameterized on `[veclen, kblk, tr, tc]`, which is
`KernelPolicy<float, _veclen, _kblk, 4, 4, _tr, _tc>`
(`raft/linalg/contractions.cuh:63-107`; `AccRowsPerTh = AccColsPerTh = 4` in
both policies their float dispatch instantiates, so those two are fixed
here). The host picks the instantiation with THEIR selection computation,
transcribed in `fused_veclen_for` and `fused_is_skinny` below from
`cuvs/src/distance/fused_distance_nn-inl.cuh:102-233`:

- veclen 4 when `4k % 16 == 0` and both base pointers are 16-byte aligned
  (`:110`), else 2 on the 8-byte test (`:158`), else 1 (`:210`). The
  divisibility test is what makes a `veclen`-wide load never straddle `k`,
  so the vector arm needs no per-load tail handling and the scalar arm IS
  the tail handling.
- `Policy4x4Skinny` (Kblk=8, 8x8 threads, `contractions.cuh:183-196`) when
  `k < 32` (`is_skinny`, `-inl.cuh:105`), `Policy4x4` (Kblk=32, 16x16
  threads, `contractions.cuh:160-166`) otherwise.

An earlier version of this file was a single instantiation that read one
float at a time where their `ldg` reads `Veclen` (`raft::TxN_t`, used by
`ldgX`/`ldgY` at `raft/linalg/detail/contractions.cuh:189-259`), and owned
rows/columns in BLOCKED runs (`tr * AccRowsPerTh + i`) where theirs are
STRIDED (`accrowid + i * P::AccThRows`, `acccolid + j * P::AccThCols`,
`contractions.cuh:100-102` and the lds/epilog indexing everywhere). Both of
those were port errors, not decisions, and both are now theirs.

THE LOOP STRUCTURE IS `PairwiseDistances::run()`, SINGLE-BUFFERED
------------------------------------------------------------------
The m axis and the n axis both grid-stride exactly as `run()` does
(`pairwise_distance_base.cuh:131-186`), the norms are staged through shared
memory (`load_norms`, `:243-274`), and the epilogue is
`l2_exp_distance_op::epilog` (`distance_ops/l2_exp.cuh:117-146`) including
its self-neighbor round-off guard. What is NOT theirs is the buffering:
their `P::SmemSize` is TWO pages (`2 * SmemPage`, `contractions.cuh:104`,
36,864 bytes at Policy4x4<float> plus the norm rows) and Apple caps a
threadgroup at 32,768, so this kernel runs one page with a second barrier
per k-tile where their double buffer needs one. PORTING.md 42. At the
shipped k-means shapes `k <= Kblk`, so their main loop body runs zero times
and nothing is lost; the deviation is priced for large k in the entry.

THE CROSS-THREAD MERGE IS THEIRS. THE CROSS-BLOCK ONE STILL IS NOT
-----------------------------------------------------------------------
`rowEpilog_lambda` does two separate things and only the second one is
unportable. Keeping them apart is the whole content of this section.

1. **The intra-warp merge, `simt_kernel.cuh:119-130`.** The `P::AccThCols`
   threads that share a row hold partial `(value, key)` minima, and they are
   combined with `raft::shfl` on the key AND on the value. **That is
   ported.** It used to be a shared-memory transpose plus a serial 16-way
   scan, justified by a claim that Mojo has no lane primitives. **The claim
   was false** (`PORTING.md` 2): `std.gpu.primitives.warp` has the shuffles,
   and `block.min` was never the answer here because it reduces VALUES ONLY
   and this reduction has to carry the key that achieved the minimum.

   One detail is not literal. Their `raft::shfl(x, lid + j, P::AccThCols)`
   ROTATES within a width-`AccThCols` subgroup, relying on CUDA's `width`
   argument to apply the modulo; their own comment at `simt_kernel.cuh:123`
   says so ("the shfl op applies the modulo internally"). Mojo's
   `shuffle_idx` has no width parameter, so the same coverage is obtained
   with `shuffle_xor`, which Modular's own GPU guide names as THE reduction
   primitive. XOR with an offset below
   `AccThCols` flips only the low lane bits, so it stays inside the same
   aligned `AccThCols` group their rotate stays inside, and both leave EVERY
   lane of the group holding the group's winner. The results are identical,
   not merely equivalent, because the reducer is a min over a TOTAL order and
   is therefore associative, commutative and idempotent: no reduction shape
   can change the answer.

   The total order is the one part of this that is OURS. Their fused
   comparator `KVPMinReduceImpl`
   (`detail/fused_distance_nn/helper_structs.cuh:39-44`) compares the VALUE
   only and keeps the shuffled partner on a tie, so their fused answer is
   shuffle-shape dependent. This tree uses the tie-break their UNFUSED arm
   gets, which is `raft::argmin_op`
   (`raft/core/operators.hpp:198-205`: `(b.value < a.value) || ((a.value ==
   b.value) && (b.key < a.key))`), reached from
   `kmeans_common.cuh:474-489` where `minClusterAndDistanceCompute`'s
   non-fused branch hands `raft::argmin_op{}` to `coalescedReduction`. It is
   used in both arms here so the two can be diffed against each other.
   (This paragraph used to attribute the tie-break to
   `MinAndDistanceReduceOpImpl` at `helper_structs.cuh:39-62`. That struct is
   at `:47-97` and it is VALUE-ONLY too -- `if (other.value < out->value)`,
   `:52` and `:59` -- so it was never the source of a key tie-break.) That
   is load-bearing rather than tidiness (`PORTING.md` 14).

   Two things this relies on, both true for BOTH policies: `tc`
   (`P::AccThCols`) is a power of two no larger than the lane width, and
   `acccolid = tid % tc`, so the threads sharing a row are CONTIGUOUS and
   aligned and never straddle a warp boundary. Change either and the
   shuffle silently reduces the wrong set. The `comptime assert` in the
   kernel pins the first.

2. **The cross-block merge, `updateReducedVal` (`simt_kernel.cuh:25-47`).**
   Still replaced, and for a reason that has nothing to do with Mojo. It
   serializes per-row global updates with a mutex array and `atomicCAS`;
   their own comment calls it a workaround for pre-Volta hang behavior and
   says a 64-bit `atomicCAS` would eliminate it. The launcher instead PINS
   `grid.x = 1` -- the rest of the grid shape is their
   `launchConfigGenerator`, see `min_cluster_distance_compute.mojo` -- so
   every row is owned by exactly one block and there is nothing to
   serialize. Their kernel grid-strides both axes and therefore needs the
   mutex. This is the `replaced` row in `PORTED_MAP.tsv`, and it costs
   parallelism when `n` is large and `m` is small, which is the opposite of
   every shape in this repository.

   What survives from `updateReducedVal` is its LANE STRUCTURE: the lanes it
   lets write are `lid == j * P::AccThCols`, the first lane of each group,
   each writing its own `P::AccRowsPerTh` rows. That is `acccolid == 0`
   below.

One more deliberate difference: when `sqrt` is requested, their epilog takes
it per accumulator CELL before the min reduce (`l2_exp.cuh:137-145`); this
kernel takes it once per ROW at the final write. `sqrt` is monotone and
injective on `[0, inf)`, so the argmin, the tie set, and the written value
are all bit-identical, and `AccRowsPerTh * AccColsPerTh` sqrts per thread
per tile become `AccRowsPerTh` per row. PORTING.md 43.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation


comptime FUSED_MAX = Float32(3.4028234663852886e38)

#: `get_clamp_precision<float>()`, `distance_ops/l2_exp.cuh:36`: the
#: round-off tolerance of the self-neighbor guard in the epilog.
comptime FUSED_CLAMP_PRECISION = Float32(1.0e-6)

# `Policy4x4<float, _veclen>`: KernelPolicy<float, v, 32, 4, 4, 16, 16>
# (`raft/linalg/contractions.cuh:160-166`).
comptime FUSED_NORMAL_KBLK = 32
comptime FUSED_NORMAL_TR = 16
comptime FUSED_NORMAL_TC = 16

# `Policy4x4Skinny<float, _veclen>`: KernelPolicy<float, v, 8, 4, 4, 8, 8>
# (`raft/linalg/contractions.cuh:183-196`), "faster for fusedL2NN on skinny
# matrices, i.e., matrices with a small k dimension" (`:177-181`).
comptime FUSED_SKINNY_KBLK = 8
comptime FUSED_SKINNY_TR = 8
comptime FUSED_SKINNY_TC = 8


def fused_is_skinny(k: Int) -> Bool:
    """`bool is_skinny = k < 32;`, `fused_distance_nn-inl.cuh:105`, with
    their own comment: at k below 32 the Policy4x4 tiles waste work on
    padding, so a Kblk=8 policy with 8x8 threads is used instead."""
    return k < 32


def fused_veclen_for(k: Int, x_addr: Int, y_addr: Int) -> Int:
    """Their veclen selection, `fused_distance_nn-inl.cuh:107-110` (16-byte
    arm), `:158` (8-byte arm), `:210` (scalar arm), for float32.

        size_t bytes = sizeof(DataT) * k;
        if (16 % sizeof(DataT) == 0 && bytes % 16 == 0 && px % 16 == 0
                                                       && py % 16 == 0) -> 4
        else if (8 % ... && bytes % 8 == 0 && px % 8 == 0 && py % 8 == 0) -> 2
        else -> 1

    `x_addr`/`y_addr` are the BYTE addresses of the two base pointers, their
    `reinterpret_cast<uintptr_t>`. The `bytes % 16` term is the load-safety
    term: with `k` a multiple of the veclen, a vector load that starts in
    bounds ends in bounds, which is why the kernel's `ldg` needs no tail arm.
    The launcher and the checks call THIS function, so the arm the bench
    takes is the arm the check pins.
    """
    var bytes = 4 * k
    if bytes % 16 == 0 and x_addr % 16 == 0 and y_addr % 16 == 0:
        return 4
    if bytes % 8 == 0 and x_addr % 8 == 0 and y_addr % 8 == 0:
        return 2
    return 1


def fused_smem_bytes(skinny: Bool, veclen: Int) -> Int:
    """The kernel's ACTUAL threadgroup footprint: one X page, one Y page
    (`SmemStride * (Mblk + Nblk)` floats, single-buffered -- PORTING.md 42),
    plus the two norm rows (`(Mblk + Nblk)` floats, their
    `shared_mem_size()` addend at `l2_exp.cuh:98-101`). Fed to the occupancy
    term of `launch_config_generator`, which is what their
    `launchConfigGenerator<P>(m, n, shmemSize, kernel)` call does with
    `shmemSize` at `fused_l2_nn.cuh:135-136`."""
    var kblk = FUSED_SKINNY_KBLK if skinny else FUSED_NORMAL_KBLK
    var mblk = 4 * (FUSED_SKINNY_TR if skinny else FUSED_NORMAL_TR)
    var nblk = 4 * (FUSED_SKINNY_TC if skinny else FUSED_NORMAL_TC)
    var stride = kblk + veclen
    return (mblk + nblk) * stride * 4 + (mblk + nblk) * 4


def fused_distance_nn_kernel[
    veclen: Int, kblk: Int, tr: Int, tc: Int
](
    out_key: MutPointer[UInt32, MutAnyOrigin],
    out_value: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    xn: MutPointer[Float32, MutAnyOrigin],
    yn: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    is_sqrt_in: Int32,
):
    """`fusedDistanceNNkernel` with the L2-expanded op and a min reduce, at
    `KernelPolicy<float, veclen, kblk, 4, 4, tr, tc>`.

    Launch `block_dim = (tr * tc, 1, 1)` and a grid from
    `launch_config_generator` with `grid_dim.x` PINNED to 1 (module
    docstring, part 2): both axes grid-stride as `PairwiseDistances::run()`
    does, but a `grid_dim.x > 1` launch would race the per-row writes that
    replaced their mutex. `k % veclen == 0` is the caller's contract, from
    `fused_veclen_for`.
    """
    # Their Policy enums, `raft/linalg/contractions.cuh:63-107`.
    comptime rpt = 4  # AccRowsPerTh, both float policies
    comptime cpt = 4  # AccColsPerTh, both float policies
    comptime nthreads = tr * tc
    comptime mblk = rpt * tr
    comptime nblk = cpt * tc
    comptime ldg_th_row = kblk // veclen
    comptime ldg_per_th_x = (mblk * ldg_th_row) // nthreads
    comptime ldg_rows_x = mblk // ldg_per_th_x
    comptime ldg_per_th_y = (nblk * ldg_th_row) // nthreads
    comptime ldg_rows_y = nblk // ldg_per_th_y
    comptime smem_stride = kblk + veclen  # padding, not a rounding
    comptime page_x = smem_stride * mblk
    comptime page_y = smem_stride * nblk

    comptime assert kblk % veclen == 0, "Kblk must be a Veclen multiple"
    comptime assert (mblk * ldg_th_row) % nthreads == 0, "LdgPerThX frac"
    comptime assert (nblk * ldg_th_row) % nthreads == 0, "LdgPerThY frac"
    comptime assert (tc & (tc - 1)) == 0 and tc <= 32, (
        "the rowEpilog shuffle needs AccThCols to be a power of two no"
        " larger than the lane width"
    )

    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)

    # `Contractions_NT` thread ids, `contractions.cuh:96-102`: the LDG
    # assignment (srowid/scolid) and the accumulation assignment
    # (accrowid/acccolid) are different partitions of the same threads.
    var srowid = tid // ldg_th_row
    var scolid = (tid % ldg_th_row) * veclen
    var accrowid = tid // tc
    var acccolid = tid % tc

    var sx = stack_allocation[
        page_x,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sy = stack_allocation[
        page_y,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    # `load_norms`' rows, their `&smem[P::SmemSize]` region
    # (`pairwise_distance_base.cuh:245-246`), separate allocations here.
    var s_xnorm = stack_allocation[
        mblk,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_ynorm = stack_allocation[
        nblk,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # `run()`'s grid offsets and strides, `pairwise_distance_base.cuh:121-124`.
    var grid_offset_m = Int(block_idx.y) * mblk
    var grid_stride_m = Int(grid_dim.y) * mblk
    var grid_offset_n = Int(block_idx.x) * nblk
    var grid_stride_n = Int(grid_dim.x) * nblk

    var tile_m = grid_offset_m
    while tile_m < m:
        # `KVPair val[P::AccRowsPerTh]`, in registers, reset per row tile
        # (`simt_kernel.cuh:79-82` init, `:144-147` reset). Lane `i` of the
        # SIMD is row `accrowid + i * tr` of the tile.
        var val = SIMD[DType.float32, rpt](FUSED_MAX)
        var key = SIMD[DType.uint32, rpt](0xFFFFFFFF)

        var tile_n = grid_offset_n
        while tile_n < n:
            var acc0 = SIMD[DType.float32, cpt](0.0)
            var acc1 = SIMD[DType.float32, cpt](0.0)
            var acc2 = SIMD[DType.float32, cpt](0.0)
            var acc3 = SIMD[DType.float32, cpt](0.0)

            var kt = 0
            while kt < k:
                # --- ldgX + stsX (`contractions.cuh:189-216, 262-269`):
                # `LdgPerThX` vector loads of `Veclen` floats, zero-filled
                # out of bounds exactly where theirs is. `koff < k` never
                # splits a vector: `k % veclen == 0` by selection.
                var koff = kt + scolid
                comptime for li in range(ldg_per_th_x):
                    var xrow = tile_m + srowid + li * ldg_rows_x
                    var vx = SIMD[DType.float32, veclen](0.0)
                    if koff < k and xrow < m:
                        vx = x.unsafe_load[width=veclen](xrow * k + koff)
                    sx.unsafe_store(
                        (srowid + li * ldg_rows_x) * smem_stride + scolid, vx
                    )
                # --- ldgY + stsY (`contractions.cuh:225-259, 271-278`).
                comptime for li in range(ldg_per_th_y):
                    var yrow = tile_n + srowid + li * ldg_rows_y
                    var vy = SIMD[DType.float32, veclen](0.0)
                    if koff < k and yrow < n:
                        vy = y.unsafe_load[width=veclen](yrow * k + koff)
                    sy.unsafe_store(
                        (srowid + li * ldg_rows_y) * smem_stride + scolid, vy
                    )
                barrier()

                # --- accumulate (`pairwise_distance_base.cuh:207-236`):
                # `ldsXY` a Veclen chunk per accumulator row/col
                # (`contractions.cuh:281-299`), then the register-tile FMA of
                # `accumulate_reg_tile` with `l2_exp_distance_op::core`
                # (`l2_exp.cuh:103-109`, `acc += x * y`). The `v` loop is
                # ascending inside ascending chunks, so every accumulator
                # cell sums its k terms in the same order at every veclen.
                comptime for kc in range(kblk // veclen):
                    var rx0 = sx.unsafe_load[width=veclen](
                        (accrowid + 0 * tr) * smem_stride + kc * veclen
                    )
                    var rx1 = sx.unsafe_load[width=veclen](
                        (accrowid + 1 * tr) * smem_stride + kc * veclen
                    )
                    var rx2 = sx.unsafe_load[width=veclen](
                        (accrowid + 2 * tr) * smem_stride + kc * veclen
                    )
                    var rx3 = sx.unsafe_load[width=veclen](
                        (accrowid + 3 * tr) * smem_stride + kc * veclen
                    )
                    var ry0 = sy.unsafe_load[width=veclen](
                        (acccolid + 0 * tc) * smem_stride + kc * veclen
                    )
                    var ry1 = sy.unsafe_load[width=veclen](
                        (acccolid + 1 * tc) * smem_stride + kc * veclen
                    )
                    var ry2 = sy.unsafe_load[width=veclen](
                        (acccolid + 2 * tc) * smem_stride + kc * veclen
                    )
                    var ry3 = sy.unsafe_load[width=veclen](
                        (acccolid + 3 * tc) * smem_stride + kc * veclen
                    )
                    comptime for v in range(veclen):
                        var yv = SIMD[DType.float32, cpt](
                            ry0[v], ry1[v], ry2[v], ry3[v]
                        )
                        acc0 += SIMD[DType.float32, cpt](rx0[v]) * yv
                        acc1 += SIMD[DType.float32, cpt](rx1[v]) * yv
                        acc2 += SIMD[DType.float32, cpt](rx2[v]) * yv
                        acc3 += SIMD[DType.float32, cpt](rx3[v]) * yv
                # Second barrier per k-tile: the single-buffer deviation
                # (PORTING.md 42). Their double buffer writes the NEXT page
                # while this one is read and needs one sync; one page cannot.
                barrier()
                kt += kblk

            # --- load_norms (`pairwise_distance_base.cuh:243-274`): both
            # norm rows staged through shared memory, the X row only on the
            # first column tile of this pass, exactly their guard.
            if tile_n == grid_offset_n:
                var i = tid
                while i < mblk:
                    var idx = tile_m + i
                    # Explicit `if`, not a conditional expression: the load
                    # must not be reachable out of bounds (PORTING.md 19).
                    var nv = Float32(0.0)
                    if idx < m:
                        nv = xn.unsafe_load(idx)
                    s_xnorm[i] = nv
                    i += nthreads
            var i2 = tid
            while i2 < nblk:
                var idx2 = tile_n + i2
                var nv2 = Float32(0.0)
                if idx2 < n:
                    nv2 = yn.unsafe_load(idx2)
                s_ynorm[i2] = nv2
                i2 += nthreads
            barrier()

            var rxn = SIMD[DType.float32, rpt](0.0)
            var ryn = SIMD[DType.float32, rpt](0.0)
            comptime for i in range(rpt):
                # `regxn[i] = sxNorm[i * P::AccThRows + tid / P::AccThCols]`
                rxn[i] = s_xnorm[i * tr + accrowid]
            comptime for j in range(cpt):
                # `regyn[i] = syNorm[i * P::AccThCols + tid % P::AccThCols]`
                ryn[j] = s_ynorm[j * tc + acccolid]

            # --- THE EPILOGUE, in registers: `l2_exp_distance_op::epilog`
            # (`l2_exp.cuh:117-136`) then the intra-thread reduce of
            # `epilog_lambda` (`simt_kernel.cuh:101-117`), applied to this
            # column tile before it is discarded. Their epilog does not
            # guard the row axis and neither does this: an out-of-range row
            # accumulates zeros and is dropped at the write.
            comptime for i in range(rpt):
                var acc_row = acc0
                comptime if i == 1:
                    acc_row = acc1
                elif i == 2:
                    acc_row = acc2
                elif i == 3:
                    acc_row = acc3
                comptime for j in range(cpt):
                    # `tmpkey = acccolid + j * P::AccThCols + gridStrideX`
                    var col = tile_n + acccolid + j * tc
                    if col < n:
                        var d = rxn[i] + ryn[j] - Float32(2.0) * acc_row[j]
                        # `val * (val > 0) * !((val*val < eps) * (xn == yn))`
                        # -- the positivity clamp and the self-neighbor
                        # round-off guard, `l2_exp.cuh:127-135`.
                        if d <= Float32(0.0) or (
                            d * d < FUSED_CLAMP_PRECISION and rxn[i] == ryn[j]
                        ):
                            d = Float32(0.0)
                        # `raft::argmin_op`'s total order (module docstring,
                        # part 1): lowest value, then lowest key.
                        if d < val[i] or (d == val[i] and UInt32(col) < key[i]):
                            val[i] = d
                            key[i] = UInt32(col)
            tile_n += grid_stride_n

        # --- `rowEpilog_lambda`, `simt_kernel.cuh:119-130` -------------------
        # Merge the `tc` column-threads that share each row, one butterfly
        # pass per accumulator row, key and value shuffled together, with
        # the argmin_op total order (module docstring, part 1). Every
        # thread reaches every shuffle: the loop bound is a comptime
        # constant and nothing here is conditional. That is required,
        # because a lane that skips a full-mask shuffle hangs the lanes
        # that reach it.
        var lane_offset = 1
        while lane_offset < tc:
            comptime for i in range(rpt):
                var cur_v = val[i]
                var cur_k = key[i]
                var o_v = shuffle_xor(cur_v, UInt32(lane_offset))
                var o_k = shuffle_xor(cur_k, UInt32(lane_offset))
                if o_v < cur_v or (o_v == cur_v and o_k < cur_k):
                    val[i] = o_v
                    key[i] = o_k
            lane_offset *= 2

        # `updateReducedVal`'s lane structure without its mutex: the first
        # lane of each column group writes the `rpt` rows that group owns.
        # Every lane holds the winner after the butterfly, so
        # `acccolid == 0` is a choice of writer and not a reduction step.
        # The value is already non-negative: the epilog clamped it.
        if acccolid == 0:
            comptime for i in range(rpt):
                var row = tile_m + accrowid + i * tr
                if row < m:
                    var best_v = val[i]
                    if is_sqrt_in != 0:
                        best_v = sqrt(best_v)
                    out_value.unsafe_store(row, best_v)
                    out_key.unsafe_store(row, key[i])
        tile_m += grid_stride_m
