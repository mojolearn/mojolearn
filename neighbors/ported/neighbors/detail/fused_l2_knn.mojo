"""L2 k-NN with the distance matrix NEVER written. Their DISPATCHED DEFAULT.

PORT OF `cuvs/src/neighbors/detail/fused_l2_knn.cuh::fusedL2kNN` at cuVS
`94c2819`, built on `raft/linalg/contractions.cuh::Policy2x8` and
`raft/linalg/detail/contractions.cuh::Contractions_NT`, with their
`raft/neighbors/detail/faiss_select/Select.cuh::WarpSelect` as the selector.
Partial. Do not improve.

WHY THIS FILE EXISTS, WHICH IS THE MOST IMPORTANT THING IN IT
--------------------------------------------------------------
`knn_brute_force.mojo` says it is a port of `tiled_brute_force_knn`. It is,
and **`tiled_brute_force_knn` is not the function cuVS runs for our
benchmark.** `brute_force_knn_impl` dispatches at
`knn_brute_force.cuh:443-447`:

    if (k <= 64 && rowMajorQuery == rowMajorIndex && rowMajorQuery == true &&
        (metric == L2Unexpanded || L2SqrtUnexpanded ||
         L2Expanded  || L2SqrtExpanded)) {
      fusedL2Knn(...);
    } else {
      switch (metric) { case Haversine: ... default: tiled_brute_force_knn(...) }
    }

`bench/scaling_main.mojo` measures k=10, row major, L2, 32 features. Every
one of those four conditions holds, so **cuVS takes `fusedL2Knn` and this
repository had ported only the `else`.** We ported their fallback and
benchmarked it against scikit-learn as if it were their algorithm.

WHAT THE FALLBACK COSTS, IN BYTES
----------------------------------
`tiled_brute_force_knn` materializes a `tile_rows x n` float32 distance
matrix. At the scaling harness' 400,000 index points with
`scaling_main.mojo:46`'s `tile = 256`, that is a 409.6 MB matrix per tile,
written by the GEMM, read and rewritten by `expand_distances_kernel`, and
read a third time by the selector. Eight tiles of that is roughly 23 GB of
traffic to carry 51.2 GFLOP of arithmetic.

This kernel moves none of it. The `Mblk x Nblk` accumulator tile stays in
registers, the per-row top-k stays in REGISTERS, and the only global traffic
is reading the index once.

THE SELECTOR IS THEIRS, AND IT IS IN REGISTERS
-----------------------------------------------
`fused_l2_knn.cuh:218-222` declares

    typedef WarpSelect<AccT, uint32_t, Dir=false, Comparator<AccT>,
                       NumWarpQ, NumThreadQ, 32> myWarpSelect;

and `fusedL2ExpKnnImpl:743-771` instantiates the WHOLE KERNEL at exactly two
`(NumWarpQ, NumThreadQ)` pairs, dispatched on `numOfNN` from the host:

    numOfNN <= 32  ->  <32, 2>
    numOfNN <= 64  ->  <64, 3>
    otherwise      ->  ASSERT("num of nearest neighbors must be <= 64")

That is copied literally: `fused_l2_knn_kernel` is parameterized on the same
two integers and `fused_l2_knn` below picks between the same two
instantiations at the same thresholds. The queue is
`neighbors/ported/neighbors/detail/faiss_select/select.mojo`, a struct of
`SIMD` values with no shared memory and no block phase; it is what makes the
fusion pay, because a shared-memory or device-wide selector has to read its
input from memory and therefore forces the distance matrix to exist.

Each warp of the block is exactly one `Policy::AccThCols` = 32 columns wide
and owns exactly one value of `threadIdx.x / AccThCols`, so a warp owns the
`AccRowsPerTh` = 2 tile rows `starty` and `starty + AccThRows`, and holds one
`WarpSelect` for each -- their `heapArr1` / `heapArr2` at `:359-361`.
Every lane of a warp calls `add` the SAME number of times, `AccColsPerTh`
per column tile, with `{keyMax, identity}` for a column past `n`: that is
their `:456-469` verbatim, and it is also the call contract `WarpSelect`
requires, because `checkThreadQ` holds a warp vote and a merge full of
shuffles.

**DEVIATION BLOCK 1 - the queues live in registers for the WHOLE column
sweep, where theirs are spilled to shared memory and rebuilt every column
tile.** THEIRS: the `epilog_lambda` (`:341-484`) is called once per
`gridStrideX`, constructs fresh `heapArr1`/`heapArr2` each call, and carries
the state between calls through `shDumpKV`, a `Mblk x numOfNN` shared array
(`:352`, `loadWarpQShmem` `:59`, `storeWarpQShmem` `:77`). The first column
tile (`gridStrideX == blockIdx.x * Nblk`, the `else` arm at `:454-477`)
feeds every accumulator through `add` and then `reduce`s; every LATER column
tile takes the `:367-453` arm instead, which counts the candidates below
`warpKTop`, warp-prefix-sums them into `allWarpTopKs` (`:396`, `:427-429`),
and merges them with `updateSortedWarpQ` (`:147-185`).

OURS: `heap0`/`heap1` are constructed once, before the column loop, and
every column tile runs their `else` arm. `shDumpKV`, `allWarpTopKs`,
`loadWarpQShmem`, `storeWarpQShmem` and `updateSortedWarpQ` are all
unreached and unported.

REASON, and it is a hard language wall, not a preference:
`updateSortedWarpQ` is built on `__ballot_sync` and `__ffs` (`:160`, `:165`)
-- it needs to know WHICH lanes voted, and then the index of the first of
them. Mojo 1.0 exposes warp shuffles and warp reductions
(`std.gpu.primitives.warp`) but no ballot and no lane mask, so the value
`activeLanes` cannot be formed at all. The prefix-sum staging at `:405-412`
needs `__ballot_sync` as well. **Note that this is NOT their cross-block
code**: `updateSortedWarpQ` has exactly two call sites in the file, its
definition at `:147` and `:437` inside `epilog_lambda`, and `:437` runs on
every column tile after the first at any `gridDim.x`. Their cross-block
merge in `rowEpilog_lambda` uses plain `heapArr[i]->add` (`:284-300`) and
would have been expressible.

Both arms are exact selections over the same set of `(distance, column)`
pairs, so with distinct distances they return the identical sorted answer;
`neighbors/mojo_only/knn_check.mojo` asserts that per slot and in order
against a host Float64 oracle. What the deviation costs is their threshold
pre-count: theirs rejects a whole column tile's candidates with one compare
against `warpKTop` and only pays the merge for the survivors, while ours
pays one `add` -- a compare, plus a warp vote -- per accumulator element.
`addThreadQ` (`Select.cuh:383-397`) applies the SAME `warpKTop` threshold, so no
element enters the thread queue that theirs would have rejected; the cost is
the vote in `checkThreadQ`, which is 8 per row per column tile. Not measured
separately.

**DEVIATION BLOCK 2 - the cross-block merge is not ported.** Theirs
grid-strides BOTH axes and serializes the per-row merge across column blocks
with a mutex array, `atomicCAS`/`atomicExch` and `__threadfence`
(`:241-281`, `:313-338`). We launch `gridDim.x == 1` and grid-stride the
column axis INSIDE the block, so every row is owned by exactly one block and
there is nothing to serialize. **This is their own configuration, not an
invention**: their `rowEpilog_lambda` opens `if (gridDim.x == 1) { return; }`
(`:226`) and their final store is guarded by
`(gridStrideX + Nblk * gridDim.x) >= n && gridDim.x == 1` (`:479`), so at
`gridDim.x == 1` their kernel IS this kernel. Their
`launchConfigGenerator` picks `grid.x > 1` when `m` is small; we never do,
which costs parallelism when `m` is small and `n` is huge. Same deviation,
same reason, as `cluster/ported/distance/fused_distance_nn/simt_kernel.mojo`.
What blocks it is the mutex protocol itself and nothing else: its merge is
plain `heapArr[i]->add` (`:284-300`), which IS ported, so if a spin on
`atomicCAS` plus `__threadfence` is ever established as sound on Metal this
is reachable. That is an OPEN item, not a wall.

**DEVIATION BLOCK 3 - single-buffered shared pages.** Their
`Policy::SmemSize` is `2 * SmemPage` because `Contractions_NT` is DOUBLE
BUFFERED. Two pages at Policy2x8 is 36,992 bytes against Metal's 32 KB
threadgroup limit (`PORTING.md 1`), so this port is single-buffered exactly
as `core/gemm.mojo` is. With the selector now in registers the kernel's
shared footprint is one page, 18,496 bytes, and nothing else -- so the
ceiling that forces this is Apple's alone and the double buffer would fit on
NVIDIA's 48 KB and AMD's 64 KB. That is a `lib_smem_pages_for` row, not a
constant, and it is left OPEN; see the lane file.

**DEVIATION BLOCK 4 - `sqrt`.** Not a deviation, a copy, recorded because it
looks like one. `fusedL2Knn` hard-codes `constexpr bool sqrt = false`
(`fused_l2_knn.cuh:977-979`) with the comment that FAISS bfKNN only
supports the non-sqrt metric, and `brute_force_knn_impl:463-475` takes the
square root AFTERWARDS with `powf(fabsf(x), 0.5)` over the k outputs. So the
kernel is squared-distance only and `sqrt_postprocess_kernel` below is their
`raft::linalg::unaryOp`. Taking k square roots instead of m*n of them is the
point.

THE EPILOGUE IS THEIR l2_exp OP AND IT HAS A CLAUSE `core/` IS MISSING
-----------------------------------------------------------------------
`cuvs/src/distance/detail/distance_ops/l2_exp.cuh:132-134`:

    val = regxn[i] + regyn[j] - 2 * acc
    acc[i][j] = val * (val > 0) *
                !((val * val < get_clamp_precision<float,float>()) &&
                  (regxn[i] == regyn[j]))

with `get_clamp_precision<float, float>() == 1e-6` (`:30-39`, `case 4` at
`:35`). There are TWO
clauses. The first zeroes a negative, and that is the one
`core/expand_distances.mojo:40-41` has. The second zeroes a SMALL POSITIVE
whose two norms are equal, which is their fix for a point finding itself at
a round-off distance of 1e-4 rather than 0. `core/expand_distances.mojo`
does not have it, so the unfused path ranks a self-match behind a genuine
zero-distance duplicate and the fused path does not. This kernel copies both
clauses. See the lane file; `core/` is another lane's.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import max as warp_max
from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from neighbors.ported.neighbors.detail.faiss_select.select import WarpSelect


# `raft::linalg::Policy2x8<float, 1>::Policy`, which is
# `KernelPolicy<float, Veclen=1, Kblk=16, AccRowsPerTh=2, AccColsPerTh=8,
#               AccThRows=8, AccThCols=32>`
# (`contractions.cuh:203-206`). This is NOT `core/gemm.mojo`'s Policy4x4:
# `fusedL2ExpKnnImpl:724` selects Policy2x8 for the k-NN shape, whose block
# is 16 rows by 256 columns.
comptime FKNN_VECLEN = 1
comptime FKNN_KBLK = 16
comptime FKNN_ACC_ROWS_PER_TH = 2
comptime FKNN_ACC_COLS_PER_TH = 8
comptime FKNN_ACC_TH_ROWS = 8
comptime FKNN_ACC_TH_COLS = 32

comptime FKNN_THREADS = FKNN_ACC_TH_ROWS * FKNN_ACC_TH_COLS
comptime FKNN_MBLK = FKNN_ACC_ROWS_PER_TH * FKNN_ACC_TH_ROWS
comptime FKNN_NBLK = FKNN_ACC_COLS_PER_TH * FKNN_ACC_TH_COLS
comptime FKNN_SMEM_STRIDE = FKNN_KBLK + FKNN_VECLEN
comptime FKNN_SMEM_PAGE_X = FKNN_SMEM_STRIDE * FKNN_MBLK
comptime FKNN_SMEM_PAGE_Y = FKNN_SMEM_STRIDE * FKNN_NBLK

# `ASSERT(numOfNN <= 64, "fusedL2kNN: num of nearest neighbors must be <= 64")`
# (`fused_l2_knn.cuh:581`, `:770`), which is also the `k <= 64` in the
# dispatch at `knn_brute_force.cuh:443`. Their bound is `NumWarpQ = 64` being
# the largest warp queue they instantiate, and it is ours for the same
# reason and no other.
comptime FKNN_MAX_NN = 64

# `std::numeric_limits<float>::max()` and
# `std::numeric_limits<uint32_t>::max()`, their `identity` and `keyMax`
# at `fused_l2_knn.cuh:218-219`.
comptime FKNN_IDENTITY = Float32(3.4028234663852886e38)
comptime FKNN_KEY_MAX = UInt32(0xFFFFFFFF)

# `get_clamp_precision<float, float>()`, `l2_exp.cuh:35`.
comptime FKNN_CLAMP_PRECISION = Float32(1.0e-6)


@always_inline
def l2_exp_epilog(xnv: Float32, ynv: Float32, dot: Float32) -> Float32:
    """`l2_exp_distance_op::epilog`, `l2_exp.cuh:114-146`, the clamp at
    `:132-134`.

    BOTH clauses; see the module docstring for why the second one matters and
    where it is missing in this tree.
    """
    var val = xnv + ynv - Float32(2.0) * dot
    if val <= Float32(0.0):
        return Float32(0.0)
    if val * val < FKNN_CLAMP_PRECISION and xnv == ynv:
        return Float32(0.0)
    return val


def fused_l2_knn_kernel[
    num_warp_q: Int, num_thread_q: Int
](
    out_dists: MutPointer[Float32, MutAnyOrigin],
    out_inds: MutPointer[UInt32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    xn: MutPointer[Float32, MutAnyOrigin],
    yn: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    d_in: Int32,
    num_nn_in: Int32,
):
    """`fusedL2kNN<..., NumWarpQ, NumThreadQ, ...>` with
    `l2_exp_distance_op`, at `gridDim.x == 1`.

    `x` is the QUERIES (`m x d`) and `y` is the INDEX (`n x d`), which is
    their argument order at `fused_l2_knn.cuh:1003-1004`
    (`query` then `index`) and is the opposite of the name order in
    `fusedL2Knn`'s own signature. `xn` and `yn` are the SQUARED row norms.

    `d_in` is their `k`, the feature count. `num_nn_in` is their `numOfNN`,
    the neighbor count. Their kernel calls the feature count `k` and this
    port does not, because `k` means the neighbor count everywhere else in
    this tree and the collision has already cost one reading of their file.

    `num_warp_q` / `num_thread_q` are their `NumWarpQ` / `NumThreadQ`
    template arguments, and `num_nn_in` must be `<= num_warp_q`. The host
    picks between `<32, 2>` and `<64, 3>` exactly as `:765-771` does.

    Launch `grid_dim = (1, ceil(m / FKNN_MBLK), 1)`,
    `block_dim = (FKNN_THREADS, 1, 1)`. `grid_dim.x` must be 1; see
    DEVIATION BLOCK 2.

    Output is `m x num_nn`, SORTED ascending by distance, which is what
    their `WarpSelect::reduce()` produces for `Dir == false`.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var d = Int(d_in)
    var num_nn = Int(num_nn_in)

    var tid = Int(thread_idx.x)
    # `accrowid` and `acccolid`, `detail/contractions.cuh:99-100`. A warp is
    # one `tr` and all 32 `tc`, which is why the queues below are per-warp.
    var tr = tid // FKNN_ACC_TH_COLS
    var tc = tid % FKNN_ACC_TH_COLS
    var m0 = Int(block_idx.y) * FKNN_MBLK

    var sx = stack_allocation[
        FKNN_SMEM_PAGE_X,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sy = stack_allocation[
        FKNN_SMEM_PAGE_Y,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # `myWarpSelect heapArr1(identity, keyMax, numOfNN);`
    # `myWarpSelect heapArr2(identity, keyMax, numOfNN);`
    # `myWarpSelect* heapArr[] = {&heapArr1, &heapArr2};`, `:359-361`.
    # Theirs are re-declared inside `epilog_lambda` and therefore once per
    # column tile; ours are declared once, outside the column sweep. That is
    # DEVIATION BLOCK 1 and it is the whole point of the file.
    var heap0 = WarpSelect[num_warp_q, num_thread_q, False](
        FKNN_IDENTITY, FKNN_KEY_MAX, num_nn
    )
    var heap1 = WarpSelect[num_warp_q, num_thread_q, False](
        FKNN_IDENTITY, FKNN_KEY_MAX, num_nn
    )

    # `const IdxT starty = gridStrideY + (threadIdx.x / Policy::AccThCols);`
    # `:355`, then `gmemRowId = starty + i * Policy::AccThRows` `:457`.
    # Both are warp-uniform, which is what makes the `< m` guards below safe
    # to wrap around a queue method that votes.
    var row0 = m0 + tr
    var row1 = m0 + tr + FKNN_ACC_TH_ROWS
    var have0 = row0 < m
    var have1 = row1 < m

    var xn0 = Float32(0.0)
    if have0:
        xn0 = xn.unsafe_load(row0)
    var xn1 = Float32(0.0)
    if have1:
        xn1 = xn.unsafe_load(row1)

    # The grid-stride over the column axis, pulled inside the block.
    var n0 = 0
    while n0 < n:
        var acc0 = SIMD[DType.float32, FKNN_ACC_COLS_PER_TH](0.0)
        var acc1 = SIMD[DType.float32, FKNN_ACC_COLS_PER_TH](0.0)

        var kt = 0
        while kt < d:
            # `ldgXY` (`:150-165`) + `stsXY` (`:166-175`, and the `stsX`
            # at `:261` / `stsY` at `:270`), as a flat
            # strided sweep. `Veclen == 1` here, so there is no vector load
            # to lose.
            var ex = tid
            while ex < FKNN_MBLK * FKNN_KBLK:
                var r = ex // FKNN_KBLK
                var c = ex % FKNN_KBLK
                var v = Float32(0.0)
                if m0 + r < m and kt + c < d:
                    v = x.unsafe_load((m0 + r) * d + kt + c)
                sx[r * FKNN_SMEM_STRIDE + c] = v
                ex += FKNN_THREADS
            var ey = tid
            while ey < FKNN_NBLK * FKNN_KBLK:
                var r2 = ey // FKNN_KBLK
                var c2 = ey % FKNN_KBLK
                var v2 = Float32(0.0)
                if n0 + r2 < n and kt + c2 < d:
                    v2 = y.unsafe_load((n0 + r2) * d + kt + c2)
                sy[r2 * FKNN_SMEM_STRIDE + c2] = v2
                ey += FKNN_THREADS
            barrier()

            # `ldsXY`, `detail/contractions.cuh:176-180`, and the `ldsX`
            # (`:279-298`) / `ldsY` (`:299-317`) it calls. The row a thread
            # owns is `accrowid + i * AccThRows` and the column is
            # `acccolid + j * AccThCols`: STRIDED, not blocked. That mapping
            # is not cosmetic here, because their epilogue at
            # `fused_l2_knn.cuh:355-356` recomputes the same expressions to
            # decide which global row and column an accumulator belongs to.
            for kk in range(FKNN_KBLK):
                var regy = SIMD[DType.float32, FKNN_ACC_COLS_PER_TH](0.0)

                @parameter
                for j in range(FKNN_ACC_COLS_PER_TH):
                    regy[j] = sy[
                        (tc + j * FKNN_ACC_TH_COLS) * FKNN_SMEM_STRIDE + kk
                    ]
                acc0 += sx[(tr + 0) * FKNN_SMEM_STRIDE + kk] * regy
                acc1 += (
                    sx[(tr + FKNN_ACC_TH_ROWS) * FKNN_SMEM_STRIDE + kk] * regy
                )
            barrier()
            kt += FKNN_KBLK

        # --- `epilog_lambda`, `fused_l2_knn.cuh:341-484` -----------------
        # The `gridStrideX == blockIdx.x * Policy::Nblk` arm, `:454-477`,
        # which for us is EVERY column tile. DEVIATION BLOCK 1.
        #
        # `regyn`, the per-column norms their contraction hands the lambda
        # as `OutT* regyn` (`:346`).
        var regyn = SIMD[DType.float32, FKNN_ACC_COLS_PER_TH](0.0)

        @parameter
        for j in range(FKNN_ACC_COLS_PER_TH):
            var cj = n0 + tc + j * FKNN_ACC_TH_COLS
            if cj < n:
                regyn[j] = yn.unsafe_load(cj)

        # `for (int i = 0; i < Policy::AccRowsPerTh; ++i)` `:456`, unrolled
        # by hand because `acc0`/`acc1` are separate SIMD values. The
        # `if (gmemRowId < m)` guard is `:459` and is warp-uniform.
        if have0:
            @parameter
            for j in range(FKNN_ACC_COLS_PER_TH):
                # `Pair otherKV = {keyMax, identity};` `:463`, overwritten
                # only when `colId < ldd`. The out-of-range lane still calls
                # `add`, which is both their code and the call contract
                # `checkThreadQ`'s warp vote requires.
                var col = n0 + tc + j * FKNN_ACC_TH_COLS
                var key = FKNN_IDENTITY
                var val = FKNN_KEY_MAX
                if col < n:
                    key = l2_exp_epilog(xn0, regyn[j], acc0[j])
                    val = UInt32(col)
                heap0.add(key, val)
        if have1:
            @parameter
            for j in range(FKNN_ACC_COLS_PER_TH):
                var col1 = n0 + tc + j * FKNN_ACC_TH_COLS
                var key1 = FKNN_IDENTITY
                var val1 = FKNN_KEY_MAX
                if col1 < n:
                    key1 = l2_exp_epilog(xn1, regyn[j], acc1[j])
                    val1 = UInt32(col1)
                heap1.add(key1, val1)

        n0 += FKNN_NBLK

    # `bool needSort = (heapArr[i]->numVals > 0);`
    # `needSort = __any_sync(mask, needSort);`
    # `if (needSort) { heapArr[i]->reduce(); }`, `:471-473`. Theirs runs it
    # per column tile because the queue is about to be spilled; ours runs it
    # once, because the queue was never spilled.
    #
    # Then `storeWarpQGmem`, `:95-117`, reached through their
    # `gridDim.x == 1` final-iteration guard at `:479`. `writeOut` places
    # register `i` of lane `l` at output slot `i * 32 + l`, which is exactly
    # the `idx = j * warpSize + lid` of `:109`.
    if have0:
        var f0 = Int32(0)
        if heap0.num_vals > 0:
            f0 = Int32(1)
        if warp_max(f0) != Int32(0):
            heap0.reduce()
        heap0.write_out(
            out_dists.unsafe_offset(row0 * num_nn),
            out_inds.unsafe_offset(row0 * num_nn),
            num_nn,
        )
    if have1:
        var f1 = Int32(0)
        if heap1.num_vals > 0:
            f1 = Int32(1)
        if warp_max(f1) != Int32(0):
            heap1.reduce()
        heap1.write_out(
            out_dists.unsafe_offset(row1 * num_nn),
            out_inds.unsafe_offset(row1 * num_nn),
            num_nn,
        )


def sqrt_postprocess_kernel(
    out_dists: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
):
    """`raft::linalg::unaryOp` at `knn_brute_force.cuh:463-475`.

    Theirs is `powf(fabsf(input), p)` with `p = 0.5` for L2Sqrt. The
    absolute value is theirs and is not decoration: the kernel's clamp can
    only produce a non-negative, but their `LpUnexpanded` case shares this
    lambda. `n * k` elements, not `n * m`, which is the whole reason the
    square root is taken here and not in the kernel.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(count_in):
        return
    var v = out_dists.unsafe_load(idx)
    if v < Float32(0.0):
        v = -v
    out_dists.unsafe_store(idx, sqrt(v))


def fused_l2_knn(
    ctx: DeviceContext,
    mut queries: DeviceBuffer[DType.float32],
    mut query_norm: DeviceBuffer[DType.float32],
    mut index: DeviceBuffer[DType.float32],
    mut index_norm: DeviceBuffer[DType.float32],
    mut out_dist: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    n_queries: Int,
    n_index: Int,
    n_features: Int,
    k: Int,
    is_sqrt: Bool,
) raises:
    """`fusedL2Knn`, `fused_l2_knn.cuh:947-1076`, expanded-L2 arm only.

    Their validation block at `:963-975` is copied as raises. The norms are
    an argument here rather than computed inside, because
    `fusedL2ExpKnnImpl:776-800` only computes them when they were not passed
    and every caller in this tree already has them from `compute_norms`.

    `L2Unexpanded` / `L2SqrtUnexpanded` route to `fusedL2UnexpKnn` upstream
    and are NOT ported; see `neighbors/UNPORTED.tsv` in the lane file.
    """
    # `ASSERT(k > 0)`, `ASSERT(D > 0)`, `ASSERT(n_index_rows > 0)`,
    # `ASSERT(n_query_rows > 0)`, `fused_l2_knn.cuh:963-967`.
    if k <= 0:
        raise Error("l2Knn: k must be > 0")
    if n_features <= 0:
        raise Error("l2Knn: D must be > 0")
    if n_index <= 0:
        raise Error("l2Knn: n_index_rows must be > 0")
    if n_queries <= 0:
        raise Error("l2Knn: n_query_rows must be > 0")
    # `ASSERT(numOfNN <= 64, ...)`, `:581` and `:770`. Theirs is the `else`
    # of the same two-way instantiation choice made below.
    if k > FKNN_MAX_NN:
        raise Error("fusedL2kNN: num of nearest neighbors must be <= 64")
    # Not theirs. Their queue is padded with `identity` to `numOfNN` and a
    # row shorter than k comes back with FLT_MAX entries; ours would too,
    # but the caller has no way to read that as "missing" without the
    # sentinel contract, so refuse instead of returning a silent one.
    if k > n_index:
        raise Error(
            "fusedL2kNN: k exceeds the index size; the dispatch must send"
            " this to tiled_brute_force_knn, which fills the short rows"
        )

    var grid_y = (n_queries + FKNN_MBLK - 1) // FKNN_MBLK

    # `if (numOfNN <= 32) { ...Knn32RowMajor } else if (numOfNN <= 64)
    # { ...Knn64RowMajor }`, `fusedL2ExpKnnImpl:765-771`. Two whole-kernel
    # instantiations, chosen on the host, exactly as theirs.
    if k <= 32:
        ctx.enqueue_function[fused_l2_knn_kernel[32, 2]](
            out_dist.unsafe_ptr(),
            out_idx.unsafe_ptr(),
            queries.unsafe_ptr(),
            index.unsafe_ptr(),
            query_norm.unsafe_ptr(),
            index_norm.unsafe_ptr(),
            Int32(n_queries),
            Int32(n_index),
            Int32(n_features),
            Int32(k),
            grid_dim=(1, grid_y, 1),
            block_dim=(FKNN_THREADS, 1, 1),
        )
    else:
        ctx.enqueue_function[fused_l2_knn_kernel[64, 3]](
            out_dist.unsafe_ptr(),
            out_idx.unsafe_ptr(),
            queries.unsafe_ptr(),
            index.unsafe_ptr(),
            query_norm.unsafe_ptr(),
            index_norm.unsafe_ptr(),
            Int32(n_queries),
            Int32(n_index),
            Int32(n_features),
            Int32(k),
            grid_dim=(1, grid_y, 1),
            block_dim=(FKNN_THREADS, 1, 1),
        )

    # `knn_brute_force.cuh:463-475`. Only for the Sqrt metrics, and only
    # over the `n * k` outputs.
    if is_sqrt:
        var cells = n_queries * k
        ctx.enqueue_function[sqrt_postprocess_kernel](
            out_dist.unsafe_ptr(),
            Int32(cells),
            grid_dim=((cells + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
    ctx.synchronize()
