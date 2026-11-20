# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IDENTICAL tiled k-NN arm's distances through a SHARED-MEMORY tile
(DEVIATION 3000), and the tile's own per-row top-k with a key merge in place
of the distance matrix (DEVIATION 3001). lane/knn-tiled-distance, 2026-09-17.

Kernel-matrix rows `knn_smem_distance_tile_for` and `knn_block_topk_select_for`.

WHAT THE REGISTER TILE PAID FOR
--------------------------------
`pinned_distance_tile.mojo::pinned_distance_register_tile_kernel` gives each
thread an RT_ROWS x 4 register tile and walks the feature axis once for all
of them, but every feature step still issues eight query loads and one
four-wide index load from global memory (L1 and L2 hits, mostly), and every
loaded operand goes through `ftz`, which under IDENTICAL is a bit test and a
select. Twelve loads and twelve flushes for thirty-two FMA steps. The index
column tile (up to 65,536 columns x d floats, 57.7 MB at d = 220) is read
back from L2 once per EIGHT query rows, so a 4,096-query tile reads it 512
times.

WHAT THIS FILE DOES INSTEAD
---------------------------
One block of SMT_TPB threads owns SMT_BM query rows x SMT_BN index columns.
For every slice of SMT_BK features it stages the slice of the query rows and
the slice of the transposed index columns into shared memory ONCE, flushed
through `ftz` at the staging store, then every thread reads its SMT_TM
query values and SMT_TN index values from shared memory (three 16-byte
loads per feature step for thirty-two FMA steps) and advances its
SMT_TM x SMT_TN accumulators. The index slice is read from L2 once per
SMT_BM query rows instead of once per RT_ROWS, and no operand is flushed
more than once.

WHY THE BITS ARE THE REGISTER TILE'S
------------------------------------
Every output cell `(row, col)` is still ONE ascending chain over the feature
axis, `acc = _rt_step(ftz(q[row, f]), ftz(yt[f, col]), acc)` from +0.0 for
f = 0, 1, ..., d - 1, the very step function the register tile calls, with
the same epilogue `ftz(fma(-2, acc, ftz(ftz(qn) + ftz(yn))))`, the same clamp
and the same `ftz(identical_sqrt(.))`. Staging through shared memory changes
WHERE an operand is read from and HOW MANY cells share its load; `ftz` is
idempotent, so flushing at the staging store and reading the flushed value
is the same operand `_rt_load` produced per step. No chain is split, folded
or reordered, and no feature past `d` is ever stepped (a zero-padded slice
would turn a -0.0 accumulator into +0.0 under round-to-nearest, so the
inner trip count is the slice's real length).

THE BLOCK TOP-K (DEVIATION 3001)
--------------------------------
With `TOPK`, the block writes NO distance matrix. Each warp holds SMT_TM
complete rows of the block's SMT_BN columns (one thread row of SMT_TX lanes
x SMT_TN columns), and for each row it pops the k smallest composite keys
(`select_radix_identical.mojo::composite_key(distance, tile-local column,
select_min=True)`, the selector's key) with `shuffle_min_u64` over the
lanes, writing them ascending to a partial-key buffer
`part[(row * n_col_blocks + col_block) * k + rank]`. Keys are unique (they
carry the column), so the k smallest keys of the row's whole column tile
are a subset of the union of the per-block lists, and
`partial_keys_select_kernel` selects them from that union with the same
UInt64 minima: per row, every thread keeps the CAP smallest keys of the
keys it visits, then the rank phase pops k block minima ascending. The
distance value is `twiddle_out` of the key's high half, the exact inverse
of the `twiddle_in` the key was built from, so the returned bits are the
epilogue's bits, as the unfused selector's gather of the matrix cell was.
The union does not depend on which thread saw which column, so the answer
is the same set in the same ascending order whatever the partition, which
is the argument the small-k selector and DEVIATION 2667 already rest on.
`partial_topk_merge_kernel` then merges column tiles exactly as before,
because the per-tile output has the same meaning: k ascending
(distance, tile-local index) pairs.

SABOTAGE (reach proof, never shipped): `-D MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1`
flips the lowest mantissa bit of every distance AFTER its chain and
epilogue, so any request that ran through this file returns moved bits.
"""
from std.bit import count_trailing_zeros
from max.gpu import block_idx, thread_idx
from max.gpu.primitives.warp import shuffle_xor, vote
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import TARGET_COLUMN, knn_block_topk_key32_for, lib_lane_width_for
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,
)
from neighbors.checks.lane_minimum import shuffle_min_u64
from neighbors.checks.pinned_distance_tile import (
    RT_EXACT_MAX_EXPONENT_SUM,
    RT_EXACT_MIN_EXPONENT_SUM,
    _rt_step,
    _rt_step_exact,
)
from neighbors.checks.select_radix_identical import composite_key
from neighbors.impl.matrix.detail.select_radix import twiddle_in
from neighbors.impl.matrix.detail.select_warpsort import twiddle_out

#: Query rows per thread and index columns per thread (the register tile's
#: NVIDIA shape, 8 x 4), thread rows and thread columns per block.
comptime SMT_TM = 8
comptime SMT_TN = 4
comptime SMT_TY = 8
comptime SMT_TX = 32
comptime SMT_BM = SMT_TY * SMT_TM  # 64 query rows per block
comptime SMT_BN = SMT_TX * SMT_TN  # 128 index columns per block
comptime SMT_BK = 16  # features per shared-memory slice
comptime SMT_TPB = SMT_TY * SMT_TX  # 256
#: The staged query slice is feature-major, `qs[f * SMT_QS_STRIDE + row]`,
#: so a thread's SMT_TM rows are one contiguous 16-byte-aligned span; the
#: stride is padded so the staging stores of consecutive features spread
#: over the banks.
comptime SMT_QS_STRIDE = SMT_BM + 4
comptime SMT_Q_PER_THREAD = SMT_BM * SMT_BK // SMT_TPB  # 4
comptime SMT_Y_PER_THREAD = SMT_BK * SMT_BN // SMT_TPB  # 8
comptime SMT_SENTINEL = UInt64(18446744073709551615)
comptime SMT_SABOTAGE = is_defined["MOJOLEARN_KNN_SMEM_TILE_SABOTAGE"]()
comptime SMT_MAX_K = 64
#: Reach control of the bounded rank loop (DEVIATION 3062), never shipped:
#: the bound is halved in key space, so true neighbors of later tiles drop.
comptime SMT_BOUNDED_SABOTAGE = is_defined["MOJOLEARN_KNN_BOUNDED_TOPK_SABOTAGE"]()
#: Diagnostic arms of the bounded kernel (output still valid): no early
#: leave of the rank loop, and no bound load or compare at all.
comptime SMT_DIAG_NOBREAK = is_defined["MOJOLEARN_KNN_BOUNDED_DIAG_NOBREAK"]()
comptime SMT_DIAG_NOBOUND = is_defined["MOJOLEARN_KNN_BOUNDED_DIAG_NOBOUND"]()
#: DEVIATION 3063 (kernel-matrix row `knn_block_topk_key32_for`): the block
#: top-k's rank loop on 32-bit distance halves and a slot mask.
comptime SMT_KEY32 = knn_block_topk_key32_for[TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL]()
#: The ballot's width on this column: a 64-lane wave holds two thread rows,
#: and a thread row takes its own half of the mask.
comptime SMT_MASK_DT = DType.uint64 if lib_lane_width_for[TARGET_COLUMN]() == 64 else DType.uint32
#: A row whose admission metadata is nonfinite takes this minimum exponent,
#: so the sum test below can never pass for it.
comptime SMT_META_REFUSED = -100000


@always_inline
def _smt_epilogue(
    acc: Float32, qn: Float32, yn: Float32, is_sqrt: Bool
) -> Float32:
    """The register tile's epilogue, statement for statement."""
    var dist = ftz(identical_mul_add(Float32(-2.0), acc, ftz(qn + ftz(yn))))
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    if is_sqrt:
        dist = ftz(identical_sqrt(dist))
    comptime if SMT_SABOTAGE:
        dist = bitcast[DType.float32](bitcast[DType.uint32](dist) ^ UInt32(1))
    return dist


@always_inline
def _smt_warp_min_u32(v: UInt32) -> UInt32:
    """The minimum over the SMT_TX lanes of one thread row (five xor levels;
    an integer minimum, so the same value under any tree)."""
    var r = v
    comptime for i in range(5):
        var o = shuffle_xor(r, UInt32(1 << i))
        if o < r:
            r = o
    return r


@always_inline
def _smt_ballot(predicate: Bool, ty: Int) -> UInt32:
    """The thread row's 32-lane ballot; every lane must reach it."""
    var m = vote[SMT_MASK_DT](predicate)
    comptime if SMT_MASK_DT == DType.uint64:
        return UInt32((UInt64(m) >> UInt64((ty & 1) * 32)) & UInt64(0xffffffff))
    else:
        return UInt32(m)


@always_inline
def _smt_slices[qo: MutOrigin, yo: MutOrigin, //, EXACT: Bool](
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    qs: MutPointer[Float32, qo, address_space = AddressSpace.SHARED],
    ys: MutPointer[Float32, yo, address_space = AddressSpace.SHARED],
    q_off: SIMD[DType.int32, SMT_Q_PER_THREAD],
    rq: Int, fq: Int, fy: Int, cy: Int, ycol: Int, ty: Int, tx: Int,
    d: Int, y_stride: Int,
) -> SIMD[DType.float32, SMT_TM * SMT_TN]:
    """The staged feature slices and the chains, f ascending over each
    slice's real length only. EXACT takes DEVIATION 2629's unflushed step,
    admitted by the caller for the whole block."""
    var acc = SIMD[DType.float32, SMT_TM * SMT_TN](0.0)
    var f0 = 0
    while f0 < d:
        var kk = d - f0
        if kk > SMT_BK:
            kk = SMT_BK
        comptime for i in range(SMT_Q_PER_THREAD):
            var v = Float32(0.0)
            if fq < kk:
                v = ftz(q.unsafe_load(Int(q_off[i]) + f0))
            qs.unsafe_store(fq * SMT_QS_STRIDE + rq + i * (SMT_TPB // SMT_BK), v)
        comptime for i in range(SMT_Y_PER_THREAD):
            var f = fy + i * (SMT_TPB // SMT_BN)
            var v = Float32(0.0)
            if f < kk:
                v = ftz(yt.unsafe_load((f0 + f) * y_stride + ycol))
            ys.unsafe_store(f * SMT_BN + cy, v)
        barrier()
        # THE CHAIN: f ascending over the slice's real length only.
        for f in range(kk):
            var qv = qs.unsafe_load[width=SMT_TM, alignment=16](
                f * SMT_QS_STRIDE + ty * SMT_TM
            )
            var yv = ys.unsafe_load[width=SMT_TN, alignment=16](
                f * SMT_BN + tx * SMT_TN
            )
            comptime for r in range(SMT_TM):
                comptime for c in range(SMT_TN):
                    comptime if EXACT:
                        acc[r * SMT_TN + c] = _rt_step_exact(qv[r], yv[c], acc[r * SMT_TN + c])
                    else:
                        acc[r * SMT_TN + c] = _rt_step(qv[r], yv[c], acc[r * SMT_TN + c])
        barrier()
        f0 += SMT_BK
    return acc


def smem_distance_tile_kernel[TOPK: Bool, EXACT: Bool, BOUNDED: Bool = False](
    z: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[UInt64, MutAnyOrigin],
    bound: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    q_meta: MutPointer[Float32, MutAnyOrigin],
    y_meta: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    y_stride_in: Int32,
    n_features_in: Int32,
    is_sqrt_in: Int32,
    k_in: Int32,
):
    """`z[row, col] = ||q_row||^2 + ||y_col||^2 - 2 q_row . y_col` for the
    block's SMT_BM x SMT_BN cells, clamped, rooted when asked, one ascending
    chain per cell; or, with TOPK, the block's per-row k smallest composite
    keys into `part` and nothing into `z`. `q` is the query tile's first
    row, `yt` the transposed index at the column tile's first column with
    stride `y_stride`; `n_cols` is the column tile's width, so a key's index
    half is the TILE-LOCAL column, as the small-k selector writes it.
    With EXACT, `q_meta` and `y_meta` are DEVIATION 2629's per-row
    admission metadata (`vector_exponent_admission_kernel`) and the block
    takes the unflushed step when its 64 rows and 128 columns are admitted
    together (the same three clauses, over the block's rows and columns);
    a block that fails keeps the flushed chain, bit for bit.

    BOUNDED (DEVIATION 3062, lane/knn-selector-speed; TOPK only): `bound`
    is the query tile's RUNNING top-k distances (`k` per row, ascending, the
    merge of every EARLIER column tile), and the rank loop stops as soon as
    no row of the thread row has a remaining key whose distance half is
    below its row's k-th running distance (such keys are replaced by the
    sentinel when they are built, from the block's row bounds staged in
    shared memory), writing one sentinel at that rank
    as the list's terminator (slots past it are NOT written; the list
    selector stops at the first sentinel). WHY NO BIT MOVES: column tiles
    are taken in ascending column order, so the k running entries of a row
    all carry smaller global columns than any key of this tile; a key whose
    distance half is at or above the k-th running distance therefore has k
    smaller composite keys among the running entries alone and cannot be in
    the row's top-k, and the partial merge would rank it at k or beyond.
    Dropping it before the merge instead of in the merge leaves the merged
    list the same. The keys that ARE emitted are popped by the unchanged
    rank rounds, ascending, so a list is a prefix of the unbounded list.
    Without BOUNDED `bound` is a placeholder the kernel never reads.
    """
    comptime assert TOPK or not BOUNDED, "the bound gates the block top-k's rank loop"
    comptime assert SMT_BK * SMT_BM % SMT_TPB == 0 and SMT_BK * SMT_BN % SMT_TPB == 0
    comptime assert SMT_TPB % SMT_BK == 0 and SMT_TPB % SMT_BN == 0
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var y_stride = Int(y_stride_in)
    var d = Int(n_features_in)
    var tid = Int(thread_idx.x)
    var ty = tid // SMT_TX
    var tx = tid % SMT_TX
    var row0 = Int(block_idx.y) * SMT_BM
    var col0 = Int(block_idx.x) * SMT_BN
    if row0 >= n_rows or col0 >= n_cols:
        return

    var qs = stack_allocation[
        SMT_BK * SMT_QS_STRIDE, Scalar[DType.float32],
        alignment=16, address_space=AddressSpace.SHARED,
    ]()
    var ys = stack_allocation[
        SMT_BK * SMT_BN, Scalar[DType.float32],
        alignment=16, address_space=AddressSpace.SHARED,
    ]()

    # The staging assignment: query element `tid + i * SMT_TPB` of the
    # SMT_BM x SMT_BK slice is row `tid // SMT_BK + i * (SMT_TPB // SMT_BK)`,
    # feature `tid % SMT_BK`, so SMT_BK consecutive threads read SMT_BK
    # consecutive features of one row; index element `tid + i * SMT_TPB`
    # of the SMT_BK x SMT_BN slice is feature `tid // SMT_BN + i *
    # (SMT_TPB // SMT_BN)`, column `tid % SMT_BN`, so a warp reads
    # consecutive columns of one feature row of the transposed index.
    # Rows and columns past the edge are clamped to the last valid one
    # (their chains are computed and discarded, never stored).
    var rq = tid // SMT_BK
    var fq = tid % SMT_BK
    var q_off = SIMD[DType.int32, SMT_Q_PER_THREAD](0)
    comptime for i in range(SMT_Q_PER_THREAD):
        var rr = row0 + rq + i * (SMT_TPB // SMT_BK)
        if rr > n_rows - 1:
            rr = n_rows - 1
        q_off[i] = Int32(rr * d + fq)
    var fy = tid // SMT_BN
    var cy = tid % SMT_BN
    var ycol = col0 + cy
    if ycol > n_cols - 1:
        ycol = n_cols - 1

    # DEVIATION 3062: the block's 64 row bounds (each row's k-th running
    # distance as a key half), loaded once per row and staged in shared
    # memory before the block's first barrier.
    var bs = stack_allocation[SMT_BM, Scalar[DType.uint32], address_space=AddressSpace.SHARED]()
    comptime if BOUNDED and not SMT_DIAG_NOBOUND:
        if tid < SMT_BM:
            var brow = row0 + tid
            if brow > n_rows - 1:
                brow = n_rows - 1
            var kk_bound = Int(k_in)
            var bh = twiddle_in(bound.unsafe_load(brow * kk_bound + kk_bound - 1), True)
            comptime if SMT_BOUNDED_SABOTAGE:
                bh = bh >> UInt32(1)
            bs.unsafe_store(tid, bh)
        comptime if not EXACT:
            barrier()

    var acc: SIMD[DType.float32, SMT_TM * SMT_TN]
    comptime if EXACT:
        # DEVIATION 2629's admission for the whole block: the minimum nonzero
        # and maximum exponents over the block's rows and columns, reduced
        # through shared memory; a nonfinite row refuses the block.
        var s_qlo = stack_allocation[SMT_TPB, Int32, address_space=AddressSpace.SHARED]()
        var s_qhi = stack_allocation[SMT_TPB, Int32, address_space=AddressSpace.SHARED]()
        var s_ylo = stack_allocation[SMT_TPB, Int32, address_space=AddressSpace.SHARED]()
        var s_yhi = stack_allocation[SMT_TPB, Int32, address_space=AddressSpace.SHARED]()
        var qlo = 255
        var qhi = 0
        var ylo = 255
        var yhi = 0
        if tid < SMT_BM:
            var rr = row0 + tid
            if rr > n_rows - 1:
                rr = n_rows - 1
            var m = q_meta.unsafe_load(rr)
            if m < Float32(0.0):
                qlo = SMT_META_REFUSED
            else:
                var v = Int(UInt32(m))
                qlo = v & 255
                qhi = v >> 8
        elif tid < SMT_BM + SMT_BN:
            var cc = col0 + tid - SMT_BM
            if cc > n_cols - 1:
                cc = n_cols - 1
            var m = y_meta.unsafe_load(cc)
            if m < Float32(0.0):
                ylo = SMT_META_REFUSED
            else:
                var v = Int(UInt32(m))
                ylo = v & 255
                yhi = v >> 8
        s_qlo.unsafe_store(tid, Int32(qlo))
        s_qhi.unsafe_store(tid, Int32(qhi))
        s_ylo.unsafe_store(tid, Int32(ylo))
        s_yhi.unsafe_store(tid, Int32(yhi))
        barrier()
        var stride = SMT_TPB // 2
        while stride > 0:
            if tid < stride:
                var a = s_qlo.unsafe_load(tid + stride)
                if a < s_qlo.unsafe_load(tid):
                    s_qlo.unsafe_store(tid, a)
                var b = s_qhi.unsafe_load(tid + stride)
                if b > s_qhi.unsafe_load(tid):
                    s_qhi.unsafe_store(tid, b)
                var e = s_ylo.unsafe_load(tid + stride)
                if e < s_ylo.unsafe_load(tid):
                    s_ylo.unsafe_store(tid, e)
                var g = s_yhi.unsafe_load(tid + stride)
                if g > s_yhi.unsafe_load(tid):
                    s_yhi.unsafe_store(tid, g)
            barrier()
            stride //= 2
        var log2d = 0
        while (1 << log2d) < d:
            log2d += 1
        var admitted = (
            Int(s_qlo.unsafe_load(0)) + Int(s_ylo.unsafe_load(0)) >= RT_EXACT_MIN_EXPONENT_SUM
            and Int(s_qhi.unsafe_load(0)) + Int(s_yhi.unsafe_load(0)) + log2d <= RT_EXACT_MAX_EXPONENT_SUM
        )
        if admitted:
            acc = _smt_slices[True](q, yt, qs, ys, q_off, rq, fq, fy, cy, ycol, ty, tx, d, y_stride)
        else:
            acc = _smt_slices[False](q, yt, qs, ys, q_off, rq, fq, fy, cy, ycol, ty, tx, d, y_stride)
    else:
        acc = _smt_slices[False](q, yt, qs, ys, q_off, rq, fq, fy, cy, ycol, ty, tx, d, y_stride)

    var is_sqrt = is_sqrt_in != 0
    comptime if TOPK:
        var k = Int(k_in)
        var n_cb = (n_cols + SMT_BN - 1) // SMT_BN
        var cb = Int(block_idx.x)
        comptime if SMT_KEY32:
            # DEVIATION 3063: the rank loop on DISTANCE HALVES alone. A key's
            # low half is its tile-local column, which the slot names (lane
            # tx, slot c is column col0 + tx * SMT_TN + c), so the thread
            # keeps 32 UInt32 halves and one 32-bit mask of the slots that
            # hold no key (past the edge, at or above the bound, or popped)
            # instead of 32 UInt64 keys, and rebuilds the key it writes.
            # THE ORDER IS THE COMPOSITE KEY'S: a lane offers the smallest
            # half among its live slots; the row's minimum half is a 32-bit
            # warp minimum; among the lanes that hold it the LOWEST lane
            # wins (lane order is column order) and pops its FIRST live slot
            # with that half (slot order is column order), which is the
            # smallest column among equal halves. A lane with no live slot
            # offers 0xFFFFFFFF but never enters the ballot, so a real half
            # of 0xFFFFFFFF still pops before the row is called empty; an
            # empty row gets the sentinel at that rank.
            var hk = SIMD[DType.uint32, SMT_TM * SMT_TN](UInt32(4294967295))
            var gone = UInt32(4294967295)
            comptime for r in range(SMT_TM):
                var row = row0 + ty * SMT_TM + r
                if row < n_rows:
                    var qn = ftz(q_norm.unsafe_load(row))
                    comptime for c in range(SMT_TN):
                        var col = col0 + tx * SMT_TN + c
                        if col < n_cols:
                            var dist = _smt_epilogue(
                                acc[r * SMT_TN + c], qn, y_norm.unsafe_load(col), is_sqrt
                            )
                            var half = twiddle_in(dist, True)
                            var offered = True
                            comptime if BOUNDED and not SMT_DIAG_NOBOUND:
                                # A key at or above the row's bound is never
                                # offered (the kernel docstring's argument).
                                offered = half < bs.unsafe_load(ty * SMT_TM + r)
                            if offered:
                                hk[r * SMT_TN + c] = half
                                gone = gone & ~(UInt32(1) << UInt32(r * SMT_TN + c))
            var lane_row = row0 + ty * SMT_TM
            for rank in range(k):
                var mine = SIMD[DType.uint32, SMT_TM](UInt32(4294967295))
                comptime for r in range(SMT_TM):
                    var m = UInt32(4294967295)
                    comptime for c in range(SMT_TN):
                        if ((gone >> UInt32(r * SMT_TN + c)) & UInt32(1)) == UInt32(0):
                            if hk[r * SMT_TN + c] < m:
                                m = hk[r * SMT_TN + c]
                    mine[r] = m
                var wmin = SIMD[DType.uint32, SMT_TM](0)
                comptime for r in range(SMT_TM):
                    wmin[r] = _smt_warp_min_u32(mine[r])
                var any_popped = False
                comptime for r in range(SMT_TM):
                    var holds = ((gone >> UInt32(r * SMT_TN)) & UInt32(15)) != UInt32(15)
                    var mask = _smt_ballot(holds and mine[r] == wmin[r], ty)
                    var row = lane_row + r
                    if mask != UInt32(0):
                        any_popped = True
                        if tx == Int(count_trailing_zeros(mask)):
                            # pop the FIRST live slot holding the minimum half
                            var popped = False
                            comptime for c in range(SMT_TN):
                                if not popped:
                                    if ((gone >> UInt32(r * SMT_TN + c)) & UInt32(1)) == UInt32(0):
                                        if hk[r * SMT_TN + c] == mine[r]:
                                            popped = True
                                            gone = gone | (UInt32(1) << UInt32(r * SMT_TN + c))
                                            if row < n_rows:
                                                part.unsafe_store(
                                                    (row * n_cb + cb) * k + rank,
                                                    (UInt64(mine[r]) << UInt64(32))
                                                    | UInt64(UInt32(col0 + tx * SMT_TN + c)),
                                                )
                    else:
                        # the row holds no key: the sentinel at this rank
                        # (the terminator of a bounded list), from the row's
                        # own lane so the eight stores leave together
                        if tx == r and row < n_rows:
                            part.unsafe_store((row * n_cb + cb) * k + rank, SMT_SENTINEL)
                comptime if BOUNDED and not SMT_DIAG_NOBREAK:
                    # Every ballot is the same value on every lane of the
                    # thread row, so the thread row leaves together; a
                    # 64-lane wave holds two thread rows and takes one more
                    # ballot so that both leave together.
                    comptime if SMT_MASK_DT == DType.uint64:
                        any_popped = vote[SMT_MASK_DT](any_popped) != Scalar[SMT_MASK_DT](0)
                    if not any_popped:
                        break
        else:
            var keys = SIMD[DType.uint64, SMT_TM * SMT_TN](SMT_SENTINEL)
            comptime for r in range(SMT_TM):
                var row = row0 + ty * SMT_TM + r
                if row < n_rows:
                    var qn = ftz(q_norm.unsafe_load(row))
                    comptime for c in range(SMT_TN):
                        var col = col0 + tx * SMT_TN + c
                        if col < n_cols:
                            var dist = _smt_epilogue(
                                acc[r * SMT_TN + c], qn, y_norm.unsafe_load(col), is_sqrt
                            )
                            var built = composite_key(dist, UInt32(col), True)
                            comptime if BOUNDED and not SMT_DIAG_NOBOUND:
                                # A key at or above the row's bound is never
                                # offered (the docstring's argument), so the rank
                                # loop sees only keys that can enter the top-k.
                                if UInt32(built >> UInt64(32)) >= bs.unsafe_load(ty * SMT_TM + r):
                                    built = SMT_SENTINEL
                            keys[r * SMT_TN + c] = built
            # THE RANK LOOP, the SMT_TM rows of the thread row interleaved so
            # their shuffle chains overlap. Per rank and row: every lane offers
            # its smallest remaining key; the row's minimum key is the one with
            # the smallest distance half, and among equal halves the smallest
            # column, which is the LOWEST LANE that holds that half (lane order
            # is column order: lane tx owns columns tx * SMT_TN .. + SMT_TN - 1),
            # so a 32-bit warp minimum and one ballot name it. That lane writes
            # the key (a sentinel where the row has no key left, so every slot
            # of the partial list is written) and retires it. Rows past the edge
            # take part with sentinels only, so the shuffles stay convergent.
            var lane_row = row0 + ty * SMT_TM
            for rank in range(k):
                var mine = SIMD[DType.uint64, SMT_TM](SMT_SENTINEL)
                var hi = SIMD[DType.uint32, SMT_TM](0)
                comptime for r in range(SMT_TM):
                    var m = keys[r * SMT_TN]
                    comptime for c in range(1, SMT_TN):
                        if keys[r * SMT_TN + c] < m:
                            m = keys[r * SMT_TN + c]
                    mine[r] = m
                    hi[r] = UInt32(m >> UInt64(32))
                var wmin = SIMD[DType.uint32, SMT_TM](0)
                comptime for r in range(SMT_TM):
                    wmin[r] = _smt_warp_min_u32(hi[r])
                comptime if BOUNDED and not SMT_DIAG_NOBREAK:
                    # Keys at or above the bound were never built, so a row is
                    # live while its warp minimum is a real key half. `wmin` is
                    # the same on every lane of the thread row, so the thread
                    # row leaves the loop together; a 64-lane wave holds two
                    # thread rows and takes one ballot so that both leave
                    # together. (A real key half of 0xFFFFFFFF is at or above
                    # every bound and was never built either.)
                    var live = False
                    comptime for r in range(SMT_TM):
                        if wmin[r] != UInt32(4294967295):
                            live = True
                    comptime if SMT_MASK_DT == DType.uint64:
                        live = vote[SMT_MASK_DT](live) != Scalar[SMT_MASK_DT](0)
                    if not live:
                        # One terminator per row, each from its own lane so the
                        # eight stores leave the thread row together.
                        comptime for r in range(SMT_TM):
                            if tx == r:
                                var trow = lane_row + r
                                if trow < n_rows:
                                    part.unsafe_store((trow * n_cb + cb) * k + rank, SMT_SENTINEL)
                        break
                comptime for r in range(SMT_TM):
                    var mask = _smt_ballot(hi[r] == wmin[r], ty)
                    var wl = Int(count_trailing_zeros(mask))
                    if tx == wl:
                        var row = lane_row + r
                        if row < n_rows:
                            part.unsafe_store((row * n_cb + cb) * k + rank, mine[r])
                        comptime for c in range(SMT_TN):
                            if keys[r * SMT_TN + c] == mine[r]:
                                keys[r * SMT_TN + c] = SMT_SENTINEL
    else:
        comptime for r in range(SMT_TM):
            var row = row0 + ty * SMT_TM + r
            if row < n_rows:
                var qn = ftz(q_norm.unsafe_load(row))
                comptime for c in range(SMT_TN):
                    var col = col0 + tx * SMT_TN + c
                    if col < n_cols:
                        z.unsafe_store(
                            row * n_cols + col,
                            _smt_epilogue(
                                acc[r * SMT_TN + c], qn, y_norm.unsafe_load(col), is_sqrt
                            ),
                        )


# ---------------------------------------------------------------------------
# The partial-key selector (DEVIATION 3001's second launch): per row, the k
# smallest of the `n_col_blocks * k` keys the tile's blocks wrote, ascending,
# as (distance, tile-local index) pairs into the selection destination.
# ---------------------------------------------------------------------------

comptime PKS_BLOCK = 256
comptime PKS_LANES = lib_lane_width_for[TARGET_COLUMN]()
comptime PKS_WARPS = PKS_BLOCK // PKS_LANES


def partial_keys_select_kernel[CAP: Int, TERMINATED: Bool = False](
    part: MutPointer[UInt64, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt32, MutAnyOrigin],
    n_col_blocks_in: Int32,
    k_in: Int32,
):
    """One block per row. Every thread keeps the CAP smallest keys of the
    keys it visits (a carry insertion behind a threshold, the small-k
    selector's), then k rounds of the block minimum (lane groups through
    `shuffle_min_u64`, one shared slot per group, double-buffered pages)
    pop the row's k smallest keys ascending. CAP >= k; the extra slots
    only admit more than the rank phase pops.

    TERMINATED (DEVIATION 3062): the per-block lists end at their first
    sentinel (the bounded rank loop's terminator; slots past it are not
    written), so thread `t` walks lists `t, t + 256, ...` and leaves each at
    its first sentinel instead of striding the flat buffer; and `flags` is a
    per-row array: a block whose row's flag is 0 returns at once. This is
    the flagged launch of `bound_compact_lists_launch`. Which thread sees
    which key does not matter (the union argument above). Without it
    `flags` is a placeholder the kernel never reads."""
    comptime assert CAP >= 1 and CAP <= SMT_MAX_K
    comptime assert PKS_BLOCK % PKS_LANES == 0
    var k = Int(k_in)
    var length = Int(n_col_blocks_in) * k
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var local_keys = SIMD[DType.uint64, CAP](SMT_SENTINEL)
    var threshold = SMT_SENTINEL
    var heads = stack_allocation[
        2 * PKS_WARPS, Scalar[DType.uint64], address_space=AddressSpace.SHARED,
    ]()
    var base = row * length
    comptime if TERMINATED:
        if flags.unsafe_load(row) == UInt32(0):
            return
        var n_lists = Int(n_col_blocks_in)
        var lst = tid
        while lst < n_lists:
            for s in range(k):
                var pending = part.unsafe_load(base + lst * k + s)
                if pending == SMT_SENTINEL:
                    break
                if pending < threshold:
                    comptime for slot in range(CAP):
                        if pending < local_keys[slot]:
                            var previous = local_keys[slot]
                            local_keys[slot] = pending
                            pending = previous
                    threshold = local_keys[CAP - 1]
            lst += PKS_BLOCK
    else:
        var i = tid
        while i < length:
            var pending = part.unsafe_load(base + i)
            if pending < threshold:
                comptime for slot in range(CAP):
                    if pending < local_keys[slot]:
                        var previous = local_keys[slot]
                        local_keys[slot] = pending
                        pending = previous
                threshold = local_keys[CAP - 1]
            i += PKS_BLOCK
    var warp = tid // PKS_LANES
    var lane = tid % PKS_LANES
    for rank in range(k):
        var mine = local_keys[0]
        var group_min = shuffle_min_u64[PKS_LANES](mine)
        var page = (rank & 1) * PKS_WARPS
        if lane == 0:
            heads[page + warp] = group_min
        barrier()
        var winner = heads[page]
        comptime for w in range(1, PKS_WARPS):
            var other = heads[page + w]
            if other < winner:
                winner = other
        if tid == 0:
            out_indices.unsafe_store(row * k + rank, UInt32(winner & UInt64(4294967295)))
            out_values.unsafe_store(row * k + rank, twiddle_out(UInt32(winner >> UInt64(32))))
        if mine == winner:
            comptime for slot in range(CAP - 1):
                local_keys[slot] = local_keys[slot + 1]
            local_keys[CAP - 1] = SMT_SENTINEL


def smem_tile_col_blocks(cols: Int) -> Int:
    """How many SMT_BN-wide blocks one column tile of `cols` columns takes,
    which is the partial-key buffer's second axis."""
    return (cols + SMT_BN - 1) // SMT_BN


@always_inline
def _smt_enqueue[TOPK: Bool, EXACT: Bool, BOUNDED: Bool = False](
    ctx: DeviceContext,
    z: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[UInt64, MutAnyOrigin],
    bound: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    q_meta: MutPointer[Float32, MutAnyOrigin],
    y_meta: MutPointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int, y_stride: Int, d: Int, k: Int, is_sqrt: Bool,
) raises:
    ctx.enqueue_function[smem_distance_tile_kernel[TOPK, EXACT, BOUNDED]](
        z, part, bound, q, yt, q_norm, y_norm, q_meta, y_meta,
        Int32(rows), Int32(cols), Int32(y_stride), Int32(d),
        Int32(1 if is_sqrt else 0), Int32(k),
        grid_dim=(smem_tile_col_blocks(cols), (rows + SMT_BM - 1) // SMT_BM, 1),
        block_dim=(SMT_TPB, 1, 1),
    )


def smem_distance_tile_launch(
    ctx: DeviceContext,
    z: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    q_meta: MutPointer[Float32, MutAnyOrigin],
    y_meta: MutPointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int, y_stride: Int, d: Int, is_sqrt: Bool, exact: Bool,
) raises:
    """One column tile's distance matrix, `rows x cols`, through the
    shared-memory tile (DEVIATION 3000). `exact` hands the block DEVIATION
    2629's admission metadata (`q_meta`, `y_meta`); otherwise both are
    placeholders the kernel never reads."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("the shared-memory distance tile requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or cols <= 0 or cols > 2147483647:
        raise Error("smem distance tile requires positive Int32 dimensions")
    if d <= 0 or d > 2147483647 or y_stride <= 0 or y_stride > 2147483647:
        raise Error("smem distance tile requires positive Int32 feature and stride")
    if exact:
        _smt_enqueue[False, True](ctx, z, z.bitcast[UInt64](), z, q, yt, q_norm, y_norm, q_meta, y_meta, rows, cols, y_stride, d, 0, is_sqrt)
    else:
        _smt_enqueue[False, False](ctx, z, z.bitcast[UInt64](), z, q, yt, q_norm, y_norm, q_meta, y_meta, rows, cols, y_stride, d, 0, is_sqrt)


@always_inline
def _pks_enqueue[CAP: Int](
    ctx: DeviceContext,
    part: MutPointer[UInt64, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, n_cb: Int, k: Int,
) raises:
    ctx.enqueue_function[partial_keys_select_kernel[CAP]](
        part, out_values, out_indices, out_indices, Int32(n_cb), Int32(k),
        grid_dim=(rows, 1, 1), block_dim=(PKS_BLOCK, 1, 1),
    )


def smem_block_topk_launch(
    ctx: DeviceContext,
    part: MutPointer[UInt64, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    q_meta: MutPointer[Float32, MutAnyOrigin],
    y_meta: MutPointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int, y_stride: Int, d: Int, k: Int, is_sqrt: Bool, exact: Bool,
    bounded: Bool = False,
    bound: Optional[MutPointer[Float32, MutAnyOrigin]] = None,
) raises:
    """`bounded` (DEVIATION 3062): `bound` is the query tile's running top-k
    distances, `k` per row, merged from every EARLIER column tile, and the
    lists are sentinel-terminated (`bound_compact_lists_launch` reads them).

    One column tile's block top-k (DEVIATION 3001) into `part` (at least
    `rows * smem_tile_col_blocks(cols) * k` keys); `partial_keys_select_launch`
    is the second launch. `cols >= k` is the caller's to guarantee, as it is
    for the small-k selector. `out_values` is the unread matrix placeholder."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("the block top-k requires IDENTICAL")
    if rows <= 0 or rows > 2147483647 or cols <= 0 or cols > 2147483647:
        raise Error("block top-k requires positive Int32 dimensions")
    if d <= 0 or d > 2147483647 or y_stride <= 0 or y_stride > 2147483647:
        raise Error("block top-k requires positive Int32 feature and stride")
    if k < 1 or k > SMT_MAX_K or k > cols:
        raise Error("block top-k supports only 1 <= k <= min(64, cols)")
    if bounded:
        if not bound:
            raise Error("block top-k: a bounded launch needs the running top-k")
        if exact:
            _smt_enqueue[True, True, True](ctx, out_values, part, bound.value(), q, yt, q_norm, y_norm, q_meta, y_meta, rows, cols, y_stride, d, k, is_sqrt)
        else:
            _smt_enqueue[True, False, True](ctx, out_values, part, bound.value(), q, yt, q_norm, y_norm, q_meta, y_meta, rows, cols, y_stride, d, k, is_sqrt)
    elif exact:
        _smt_enqueue[True, True](ctx, out_values, part, out_values, q, yt, q_norm, y_norm, q_meta, y_meta, rows, cols, y_stride, d, k, is_sqrt)
    else:
        _smt_enqueue[True, False](ctx, out_values, part, out_values, q, yt, q_norm, y_norm, q_meta, y_meta, rows, cols, y_stride, d, k, is_sqrt)


def partial_keys_select_launch(
    ctx: DeviceContext,
    part: MutPointer[UInt64, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, cols: Int, k: Int,
) raises:
    """The row's k smallest from the per-block lists `smem_block_topk_launch`
    wrote, into the selection destination."""
    if rows <= 0 or rows > 2147483647 or cols <= 0 or cols > 2147483647:
        raise Error("partial-key select requires positive Int32 dimensions")
    if k < 1 or k > SMT_MAX_K or k > cols:
        raise Error("partial-key select supports only 1 <= k <= min(64, cols)")
    var n_cb = smem_tile_col_blocks(cols)
    if k <= 16:
        _pks_enqueue[16](ctx, part, out_values, out_indices, rows, n_cb, k)
    elif k <= 32:
        _pks_enqueue[32](ctx, part, out_values, out_indices, rows, n_cb, k)
    else:
        _pks_enqueue[64](ctx, part, out_values, out_indices, rows, n_cb, k)


def partial_lists_flagged_launch(
    ctx: DeviceContext,
    part: MutPointer[UInt64, MutAnyOrigin],
    out_values: MutPointer[Float32, MutAnyOrigin],
    out_indices: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt32, MutAnyOrigin],
    rows: Int, cols: Int, k: Int,
) raises:
    """The partial-key selector over sentinel-terminated lists for the rows
    whose `flags[row]` is nonzero (DEVIATION 3062); rows whose flag is 0 are
    not touched."""
    if rows <= 0 or rows > 2147483647 or cols <= 0 or cols > 2147483647:
        raise Error("partial-list select requires positive Int32 dimensions")
    if k < 1 or k > SMT_MAX_K or k > cols:
        raise Error("partial-list select supports only 1 <= k <= min(64, cols)")
    var n_cb = smem_tile_col_blocks(cols)
    if k <= 16:
        ctx.enqueue_function[partial_keys_select_kernel[16, True]](
            part, out_values, out_indices, flags, Int32(n_cb), Int32(k),
            grid_dim=(rows, 1, 1), block_dim=(PKS_BLOCK, 1, 1),
        )
    elif k <= 32:
        ctx.enqueue_function[partial_keys_select_kernel[32, True]](
            part, out_values, out_indices, flags, Int32(n_cb), Int32(k),
            grid_dim=(rows, 1, 1), block_dim=(PKS_BLOCK, 1, 1),
        )
    else:
        ctx.enqueue_function[partial_keys_select_kernel[64, True]](
            part, out_values, out_indices, flags, Int32(n_cb), Int32(k),
            grid_dim=(rows, 1, 1), block_dim=(PKS_BLOCK, 1, 1),
        )
