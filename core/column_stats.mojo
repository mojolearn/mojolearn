"""Column means, centering, and the covariance product.

MOSTLY NOT PORTS OF FILES. These stand in for `raft::stats::mean`,
`raft::stats::cov` and `raft::stats::meanAdd`, which `raft/linalg/detail/
pca.cuh::pca_fit` calls in that order. RAFT is a general library this tree
does not mirror file for file, so what is reproduced is the CALL SITE and its
semantics. Same rule as `core/row_norms.mojo`.

`covariance_kernel` IS AN EXCEPTION TO THAT, as of the register-tiling round.
Its body is a port of the `isRowMajor == false` arm of
`raft/linalg/detail/contractions.cuh` under `ColKernelPolicy`
(`raft/linalg/contractions.cuh`) at RAFT `9aa17e5`, the column-major twin of
what `core/gemm.mojo` ported. `raft::stats::cov` itself still only calls
cuBLAS; the kernel underneath it is RAFT's own and it fits this shape. Its
split-K row partition is ours and is labelled as such.

THE ONE SEMANTIC THAT IS EASY TO GET WRONG
------------------------------------------
`pca_fit` centers the input IN PLACE, computes the covariance from the
centered data, and then calls `raft::stats::meanAdd` to put the input BACK
(`detail/pca.cuh:186`). The input is an in-out parameter that ends the call
unchanged. Skipping the restore leaves the caller's matrix silently centered,
which is invisible until they use it for something else.

`cov` uses the `n_rows - 1` denominator, the sample covariance, because
`pca_fit` later multiplies the explained variances by exactly `n_rows - 1` to
recover singular values.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


comptime STATS_TPB = 128


# `ColKernelPolicy<float, Veclen=4, Kblk=32, AccRowsPerTh=4, AccColsPerTh=4,
# AccThRows=16, AccThCols=16>`: the COLUMN-major half of RAFT's
# `Policy4x4<float>` (`raft/linalg/contractions.cuh`). `core/gemm.mojo` ported
# the row-major half. Which half applies is not a preference, it is forced by
# the shape; `covariance_kernel`'s docstring says why.
comptime COV_VECLEN = 4
comptime COV_KBLK = 32
comptime COV_ACC_ROWS_PER_TH = 4
comptime COV_ACC_COLS_PER_TH = 4
comptime COV_ACC_TH_ROWS = 16
comptime COV_ACC_TH_COLS = 16

comptime COV_THREADS = COV_ACC_TH_ROWS * COV_ACC_TH_COLS
comptime COV_MBLK = COV_ACC_ROWS_PER_TH * COV_ACC_TH_ROWS
comptime COV_NBLK = COV_ACC_COLS_PER_TH * COV_ACC_TH_COLS

# `SmemStride = Mblk + Veclen`, and note it is `Mblk` and NOT `Kblk` the way
# the row-major policy has it. In the column-major policy the padded axis is
# the FEATURE axis, because that is the axis threads read across during
# accumulation, so that is the axis whose bank conflicts have to be staggered.
comptime COV_SMEM_STRIDE = COV_MBLK + COV_VECLEN
comptime COV_SMEM_PAGE = COV_SMEM_STRIDE * COV_KBLK

# RAFT `static_assert`s `Mblk == Nblk` on the column-major policy. We rely on
# it too: one stride constant and one load loop serve both pages.

# Split-K, and this part is NOT from a RAFT header. See `covariance_kernel`.
comptime COV_TARGET_BLOCKS = 512
comptime COV_SPLIT_MIN_ROWS = 4 * COV_KBLK

# Kept because `decomposition/ported/linalg/detail/pca.mojo` imports the name.
# The output tile is `COV_MBLK x COV_NBLK` now, not `COV_TILE x COV_TILE`, and
# nothing should compute a grid from this.
comptime COV_TILE = COV_MBLK


def column_mean_kernel(
    mu: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`raft::stats::mean` along columns. One block per column."""
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var r = tid
    while r < n_rows:
        acc += x.unsafe_load(r * n_cols + col)
        r += STATS_TPB

    # `cub::BlockReduce`'s counterpart. MAX ships the block
    # collectives at `max.gpu.primitives.block`, so the hand-written
    # shared-memory tree reduction this replaced is gone. Same
    # arithmetic, one call, and the reduction shape is Modular's to
    # tune rather than ours to guess. See VENDOR_LIBRARIES.md.
    var s0 = block_sum[block_size=STATS_TPB](acc)
    if tid == 0:
        mu.unsafe_store(col, s0 / Float32(n_rows))


def shift_columns_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    sign_in: Float32,
):
    """Center (`sign = -1`) or restore (`sign = +1`), in place.

    Both directions in one kernel because they are the same operation and
    because pairing them makes the restore hard to forget.
    """
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return
    var col = idx % n_cols
    x.unsafe_store(
        idx, x.unsafe_load(idx) + sign_in * mu.unsafe_load(col)
    )


def covariance_kernel(
    cov: MutPointer[Float32, MutAnyOrigin],
    xc: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    scale_in: Float32,
):
    """`X^T X * scale`, register-tiled. Serves BOTH of RAFT's users of it.

    `raft::stats::cov` on centered data with `scale = 1 / (n_rows - 1)` is
    PCA's covariance. `tsvd_fit` calls `raft::linalg::gemm` with
    `CUBLAS_OP_T, CUBLAS_OP_N` and `alpha = 1` on UNCENTERED data
    (`detail/tsvd.cuh`), which is the same product with `scale = 1`.

    One kernel with a scale rather than two, because the two differ only in
    that constant and in whether the caller centered first. Which of those
    happens is the caller's business and is exactly what separates PCA from
    truncated SVD.

    Launch through `core/gemm.mojo::gemm_tn`, which owns the geometry:

        block_dim = (COV_THREADS, 1, 1)
        grid_dim  = (ceil(n_cols / COV_NBLK), ceil(n_cols / COV_MBLK), splits)

    and, when `splits > 1`, hands `cov` a buffer of `splits * n_cols * n_cols`
    floats and follows with `covariance_reduce_kernel`. At `splits == 1` the
    kernel writes the finished matrix straight into `cov` and the geometry is
    the plain two-dimensional one.

    IT IS RAFT'S COLUMN-MAJOR CONTRACTION, AND THAT IS NOT A CHOICE
    ---------------------------------------------------------------
    `raft/linalg/contractions.cuh` ships TWO policies, `KernelPolicy` and
    `ColKernelPolicy`, and `detail/contractions.cuh` carries both arms behind
    `isRowMajor`. `core/gemm.mojo::gemm_nt_kernel` ported the row-major arm.
    This kernel is the column-major one, and the reason is mechanical: a
    row-major `X[n_rows][n_cols]` is bit-for-bit a COLUMN-major matrix of
    shape `n_cols x n_rows` with leading dimension `n_cols`, and `X^T X` is
    that matrix against itself in the N-T shape. So the T-N product MAX's
    matmul refuses is the shape RAFT's other arm was already written for.

    Everything that differs from `gemm_nt_kernel` follows from that one fact:

    - the shared pages are `[row][feature]`, `Kblk x Mblk`, so the padded
      stride is `Mblk + Veclen` and not `Kblk + Veclen` (`ColKernelPolicy`);
    - a thread's four output rows are STRIDED by `AccThRows`, not four
      contiguous rows. That is theirs (`ldsX`, `isRowMajor == false`:
      `regx[i][v] = saddr[i * P::AccThRows + v * P::SmemStride]`), and it is
      what makes 16 consecutive threads read 16 consecutive shared floats.

    SYMMETRY, WHICH IS A CORRECTNESS PROPERTY HERE AND NOT AN AESTHETIC ONE
    ----------------------------------------------------------------------
    Both shared tiles are `[contraction_row][feature]`, because the contracted
    axis is the ROW axis and both operands are the same matrix. Getting those
    two indices the wrong way round produces a plausible but NON-SYMMETRIC
    matrix, which makes the Jacobi eigensolver run to its sweep limit and
    raise rather than return a wrong number. That is how the bug was found the
    first time; see PORTING.md 23.

    The tiling preserves exact symmetry, and it is worth writing down why,
    because "it is symmetric in exact arithmetic" is not the claim. Element
    `(i, j)` is `sum_r s_a[r][i] * s_b[r][j]` and element `(j, i)` is
    `sum_r s_a[r][j] * s_b[r][i]`, accumulated over the same `r` in the same
    order by a thread in the mirrored block, and float multiplication is
    exactly commutative. Split-K does not disturb it either: every split
    covers the same row range for both elements and the reduction sums the
    splits in index order. So `cov[i][j] == cov[j][i]` BITWISE, and an
    assertion on that is a cheap way to catch any future edit here.

    SPLIT-K IS OURS, AND IT IS THE PART THAT ACTUALLY MOVES THE BENCHMARK
    --------------------------------------------------------------------
    Register tiling alone would have made this kernel SLOWER at the shapes
    this repository measures, and it is worth being blunt about that. The
    output is `n_cols x n_cols`. At `n_cols = 32` a 64x64 output tile is ONE
    thread block for the whole 200,000-row contraction, where the 16x16
    kernel it replaces at least had four. Growing the tile spends the only
    parallelism this shape has.

    So the row axis is partitioned across `grid_dim.z`, each block reduces its
    own row range, and a second pass sums the partial matrices. That is what
    cuBLAS does for a T-N with small `m`, `n` and enormous `k`, and it is the
    reason `raft::stats::cov` could hand this shape to cuBLAS and not think
    about it. It is not in any RAFT header, because RAFT never wrote this
    kernel; recorded here rather than presented as a port.

    Metal has no float `atomicAdd`, so the partials go to memory and get a
    reduction kernel instead of accumulating in place.

    THREADGROUP BUDGET
    ------------------
    Two pages of `COV_SMEM_PAGE` floats: `2 * 68 * 32 * 4 = 17,408 bytes`
    against Metal's 32 KB (PORTING.md 1). Double buffering would need 34,816
    and is deferred for the same reason it is deferred in `gemm_nt_kernel`.

    **This is the only part of PCA that scales with rows.** Everything after
    it works on an `n_cols x n_cols` matrix, which is why the eigen step can
    sit on the host without breaking `HOST_AND_DEVICE.md`'s rule.

    NOT DONE YET, and named so it does not get forgotten
    ----------------------------------------------------
    **The diagonal block loads the same tile twice.** When
    `block_idx.x == block_idx.y` the two pages hold identical data, and at
    `n_cols <= COV_MBLK` that is the ONLY block, so half the global traffic of
    a small-feature PCA is a duplicate read.

    **`n_cols < COV_MBLK` wastes threads.** At `n_cols = 32` only a quarter of
    each thread's 4x4 output block is in range. RAFT's answer is a second
    policy (`Policy4x4Skinny`) and a dispatch, which is a real option here and
    needs its own measurement rather than a guess.

    **Vectorized loads.** `Veclen = 4` sets the padding, as in `gemm_nt`, but
    the loads still move one float at a time.

    NUMERIC: the contraction order over rows fixes the summation order, so
    both `COV_KBLK` and the split count move the last bits. They are fixed
    constants and a fixed policy for that reason.
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)

    var tid = Int(thread_idx.x)
    var accrowid = tid // COV_ACC_TH_COLS
    var acccolid = tid % COV_ACC_TH_COLS

    # `block_idx.y` walks the output ROW features, `block_idx.x` the output
    # COLUMN features. Both index the same feature axis of the same matrix.
    var i0 = Int(block_idx.y) * COV_MBLK
    var j0 = Int(block_idx.x) * COV_NBLK

    # The row range this block owns. `grid_dim.z == 1` gives the whole matrix
    # and the store below lands directly in `cov`, which is the pre-split-K
    # behaviour exactly.
    var n_splits = Int(grid_dim.z)
    var sid = Int(block_idx.z)
    var chunk = (n_rows + n_splits - 1) // n_splits
    chunk = ((chunk + COV_KBLK - 1) // COV_KBLK) * COV_KBLK
    var r_begin = sid * chunk
    var r_end = r_begin + chunk
    if r_end > n_rows:
        r_end = n_rows

    var s_a = stack_allocation[
        COV_SMEM_PAGE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_b = stack_allocation[
        COV_SMEM_PAGE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # SIMD values, NOT `stack_allocation`. `stack_allocation` without an
    # address space is thread-local MEMORY, and using it here turns every
    # accumulator update into a load and a store, which is the exact traffic
    # register tiling exists to remove. It made the first attempt at
    # `gemm_nt_kernel` slower than the naive kernel. See PORTING.md 26.
    var acc0 = SIMD[DType.float32, COV_ACC_COLS_PER_TH](0.0)
    var acc1 = SIMD[DType.float32, COV_ACC_COLS_PER_TH](0.0)
    var acc2 = SIMD[DType.float32, COV_ACC_COLS_PER_TH](0.0)
    var acc3 = SIMD[DType.float32, COV_ACC_COLS_PER_TH](0.0)

    var kt = r_begin
    while kt < r_end:
        # --- ldgXY + stsXY. One sweep fills both pages because `Mblk == Nblk`
        # and both pages come from the same matrix. Consecutive threads take
        # consecutive FEATURES of one row, which is the contiguous axis.
        var e = tid
        while e < COV_KBLK * COV_MBLK:
            var rr = e // COV_MBLK
            var ff = e % COV_MBLK
            var gr = kt + rr
            var ga = i0 + ff
            var gb = j0 + ff
            var va = Float32(0.0)
            var vb = Float32(0.0)
            if gr < r_end:
                if ga < n_cols:
                    va = xc.unsafe_load(gr * n_cols + ga)
                if gb < n_cols:
                    vb = xc.unsafe_load(gr * n_cols + gb)
            s_a[rr * COV_SMEM_STRIDE + ff] = va
            s_b[rr * COV_SMEM_STRIDE + ff] = vb
            e += COV_THREADS
        barrier()

        # --- ldsXY + accumulate, the `isRowMajor == false` arm. The four
        # rows and four columns a thread owns are strided by `AccThRows` and
        # `AccThCols`; they are not contiguous.
        for kk in range(COV_KBLK):
            var row_base = kk * COV_SMEM_STRIDE
            var regb = SIMD[DType.float32, COV_ACC_COLS_PER_TH](0.0)
            for q in range(COV_ACC_COLS_PER_TH):
                regb[q] = s_b[
                    row_base + acccolid + q * COV_ACC_TH_COLS
                ]
            acc0 += s_a[row_base + accrowid + 0 * COV_ACC_TH_ROWS] * regb
            acc1 += s_a[row_base + accrowid + 1 * COV_ACC_TH_ROWS] * regb
            acc2 += s_a[row_base + accrowid + 2 * COV_ACC_TH_ROWS] * regb
            acc3 += s_a[row_base + accrowid + 3 * COV_ACC_TH_ROWS] * regb

        barrier()
        kt += COV_KBLK

    # A split whose range is empty still writes its zeros: the reduction reads
    # every slot and an unwritten one is whatever the allocation held.
    var base = sid * n_cols * n_cols
    for p in range(COV_ACC_ROWS_PER_TH):
        var gi = i0 + accrowid + p * COV_ACC_TH_ROWS
        if gi >= n_cols:
            continue
        for q in range(COV_ACC_COLS_PER_TH):
            var gj = j0 + acccolid + q * COV_ACC_TH_COLS
            if gj < n_cols:
                var v = acc0[q]
                if p == 1:
                    v = acc1[q]
                elif p == 2:
                    v = acc2[q]
                elif p == 3:
                    v = acc3[q]
                cov.unsafe_store(base + gi * n_cols + gj, v * scale_in)


def covariance_reduce_kernel(
    cov: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    n_elems_in: Int32,
    n_splits_in: Int32,
):
    """Sum the split-K partial matrices. The second half of `covariance_kernel`.

    cuBLAS hides this inside its split-k GEMM and CUDA would fold it into the
    product with a float `atomicAdd`. Metal has neither, so it is a pass over
    `n_splits * n_cols^2` floats, which is nothing beside the `n_rows *
    n_cols^2` product that produced them.

    `scale` is NOT applied here. `covariance_kernel` already applied it to
    each partial, so the two paths through `gemm_tn` differ in launch count
    and in nothing else.

    Consecutive threads take consecutive elements of one partial matrix, so
    every split's read is coalesced, and every element sums the splits in
    index order, which is what keeps the result bitwise symmetric.
    """
    var n_elems = Int(n_elems_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n_elems:
        return
    var acc = Float32(0.0)
    for s in range(Int(n_splits_in)):
        acc += partials.unsafe_load(s * n_elems + idx)
    cov.unsafe_store(idx, acc)


def xty_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`A^T b`. Stands in for `raft::linalg::gemv(..., trans=true)`.

    One block per feature, striding rows. This and `covariance_kernel` are
    the only two things in ordinary least squares that touch rows at all;
    everything after them is `n_cols x n_cols`.
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var r = tid
    while r < n_rows:
        acc += x.unsafe_load(r * n_cols + col) * y.unsafe_load(r)
        r += STATS_TPB

    # `cub::BlockReduce`'s counterpart. MAX ships the block
    # collectives at `max.gpu.primitives.block`, so the hand-written
    # shared-memory tree reduction this replaced is gone. Same
    # arithmetic, one call, and the reduction shape is Modular's to
    # tune rather than ours to guess. See VENDOR_LIBRARIES.md.
    var s0 = block_sum[block_size=STATS_TPB](acc)
    if tid == 0:
        out_v.unsafe_store(col, s0)


def divide_columns_by_nonzero_kernel(
    qs: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    s_vec: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    thresh_in: Float32,
):
    """`QS <- Q invS` with `DivideByNonZero`, `lstsq.cuh`'s matrixVectorOp.

    Column `k` of `Q` is divided by eigenvalue `k`, and a column whose
    eigenvalue is at or below the threshold is ZEROED rather than divided.

    That zeroing is the whole numerical story of solving least squares
    through the normal equations. `A^T A` squares the condition number, so a
    direction the data barely constrains shows up as a tiny eigenvalue, and
    dividing by it would amplify noise without bound. Dropping the direction
    instead is a pseudo-inverse, and it is why their default OLS solver is
    SVD rather than this one.
    """
    var n = Int(n_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * n:
        return
    var col = idx % n
    var lam = s_vec.unsafe_load(col)
    if lam > thresh_in or lam < -thresh_in:
        qs.unsafe_store(idx, q.unsafe_load(idx) / lam)
    else:
        qs.unsafe_store(idx, Float32(0.0))


def diagonal_to_vector_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Pull the eigenvalues off the diagonal Jacobi leaves behind.

    cuSOLVER hands back a separate eigenvalue array; Jacobi leaves them on
    the diagonal of the matrix it consumed. One kernel bridges the two
    conventions so the rest of the port reads like theirs.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        out_v.unsafe_store(i, a.unsafe_load(i * n + i))


def scale_in_place_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scale_in: Float32,
):
    """Apply cuBLAS's `alpha` after the fact.

    `raft::linalg::gemm` takes `alpha` and folds the scale into the product.
    MAX's `matmul` has no alpha argument, so the `1 / (n_rows - 1)` that turns
    a Gram matrix into a covariance is a separate pass over `n_cols^2`
    elements. That is a deviation in launch count, not in arithmetic, and it
    is tiny: the product is O(rows * cols^2) and this is O(cols^2).
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        a.unsafe_store(i, a.unsafe_load(i) * scale_in)
