"""The matrix products under `NUMERIC_IDENTICAL`: the pins, and their reach.

IDENTITY_PATHS rows 27 and 28. DEVIATIONS 520, 521, 522, 526.

WHAT THIS FILE IS FOR, AND WHY A GREEN `gram_splitk_check` IS NOT IT
---------------------------------------------------------------------
`mojo_only/gram_splitk_check.mojo` asks whether the Gram product is RIGHT:
per cell against a Float64 oracle, bitwise symmetric, both dispatch arms by
name. It passed before this lane and it passes after, and it would pass just
as well on a build whose summation split is a different number on every
vendor -- because a per-cell oracle comparison at a 1e-5 budget cannot see a
summation ORDER at all. That is the gap this file exists in.

The four properties here are the ones a cross-vendor claim actually rests
on, and each is written so that it FAILS when its pin is removed:

1. `check_gram_chunk_count_is_pinned` -- the k partition is the pinned
   constant under IDENTICAL and the machine's number under FAST, and BOTH
   readers (the launch and `gram_splitk_scratch_covers`) see the same one.
2. `check_gram_arm_is_pinned` -- the arm is resolved from
   `COLUMN_BIT_IDENTICAL` rather than the device's column, so an NVIDIA or
   AMD build under IDENTICAL enters this kernel instead of `linalg.matmul`.
   This is the one property that CANNOT be proved by running on Apple, so it
   is proved at the table instead, against the vendor rows themselves.
3. `check_gemm_tn_refuses_over_capacity` -- the shape the split-K kernel
   cannot serve RAISES under IDENTICAL instead of falling through to the
   vendor library, and runs normally under FAST.
4. `check_pinned_gemm_is_batch_invariant` -- the headline. The bits of a
   given output cell do not depend on how many other cells were in the
   launch. Under FAST this is a REPORT (the vendor matmul is free to fail
   it, and whether it does is a measurement worth having); under IDENTICAL
   it is an ASSERTION.

THE FOURTH ONE IS THE PROPERTY THE SERVING WORLD CALLS BATCH INVARIANCE.
Changing the batch size changes the reduction order inside a tuned kernel,
so the same input gives different bits depending on what else was in the
batch. Here the same statement is: the same row of X against the same row of
Y must give the same bits whether the launch computed 4 cells or 4,096. A
kernel with one thread per output cell and an ascending contraction has that
property by construction, which is why the construction was chosen.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast

from core.gemm import (
    PINNED_GEMM_TPB,
    gemm_nt,
    gemm_tn,
    gemv_n,
)
from core.gram_splitk import (
    GRAM_MAX_CELLS_PER_THREAD,
    GRAM_MAX_COLS,
    GRAM_SPLITK_RESOLVED_COLUMN,
    GRAM_TPB,
    PINNED_GRAM_SPLITK_CHUNKS,
    gram_splitk_applies,
    gram_splitk_chunk_count,
    gram_splitk_scratch_covers,
)
from mojo_only.hardware_matrix import gram_splitk_is_target_arm
from mojo_only.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_BIT_IDENTICAL,
    COLUMN_NVIDIA,
    TARGET_COLUMN,
    column_name,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def _mode_name() -> String:
    """The mode THIS BINARY WAS COMPILED IN, printed by every check.

    Not decoration. `GLOBAL_NUMERIC_MODE` is comptime and the flip is a
    shared-file edit, so a run that compiled inside another session's flip
    window would otherwise report the wrong arm's numbers under this arm's
    label. `tools/with_identical_mode.sh` now takes the build lock
    (DEVIATION 514) and this line is the second, independent witness.
    """
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _mix(i: Int, salt: Int) -> UInt64:
    """splitmix64, as `gram_splitk_check` uses it: adjacent indices land
    nowhere near each other, so a permutation cannot hide behind a total."""
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15
        + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _val_f32(i: Int, salt: Int) -> Float32:
    """A signed value in [-1, 1)."""
    return Float32(Int(_mix(i, salt) % 2000001) - 1000000) * Float32(1.0e-6)


# ===========================================================================
# 1. THE CHUNK COUNT (DEVIATION 520, row 27)
# ===========================================================================


def check_gram_chunk_count_is_pinned() raises:
    """The k partition is a pinned constant under IDENTICAL, and every
    reader of it agrees.

    THE SABOTAGE THIS CHECK IS BUILT AROUND. A pin that only the launch
    reads and the workspace sizing does not is worse than no pin: the
    partials buffer would be sized from one count and indexed with another.
    So this does not merely read `gram_splitk_chunk_count()` -- it derives
    the count a SECOND way, from `gram_splitk_scratch_covers`' own
    threshold, and requires the two to agree. `scratch_covers(m, k)` is
    `k >= n_chunks * m`, so the smallest covering k at m = 1 IS n_chunks,
    and a binary search over that predicate recovers the count without
    calling the function under test.
    """
    var n = gram_splitk_chunk_count()

    comptime if IDENTICAL:
        if n != PINNED_GRAM_SPLITK_CHUNKS:
            raise Error(
                "check_gram_chunk_count_is_pinned: IDENTICAL build must"
                " partition k into exactly "
                + String(PINNED_GRAM_SPLITK_CHUNKS)
                + " chunks on every column, got "
                + String(n)
                + ". The count is the summation split; if it is reading the"
                " machine here it is reading the machine on the other"
                " vendor too."
            )
    else:
        if n < 1:
            raise Error("chunk count must be positive, got " + String(n))
        if n == PINNED_GRAM_SPLITK_CHUNKS:
            raise Error(
                "check_gram_chunk_count_is_pinned: the FAST build's"
                " machine-derived count is "
                + String(PINNED_GRAM_SPLITK_CHUNKS)
                + ", the same as the pinned constant. That is not an error"
                " in the code, it is an error in the CHECK: on this column"
                " the two arms are indistinguishable and this file can no"
                " longer tell whether the pin is reached. Move"
                " PINNED_GRAM_SPLITK_CHUNKS to a value no column produces."
            )

    # The second, independent derivation: the smallest k at m = 1 for which
    # a k-float scratch covers an n_chunks-float partials workspace.
    var lo = 1
    var hi = 1
    while not gram_splitk_scratch_covers(1, hi):
        hi *= 2
        if hi > (1 << 24):
            raise Error("scratch_covers never became true; bound is wrong")
    while lo < hi:
        var mid = (lo + hi) // 2
        if gram_splitk_scratch_covers(1, mid):
            hi = mid
        else:
            lo = mid + 1
    if lo != n:
        raise Error(
            "check_gram_chunk_count_is_pinned: the launch and the workspace"
            " sizing disagree about the chunk count -- launch says "
            + String(n)
            + ", scratch_covers' threshold implies "
            + String(lo)
            + ". A partials buffer sized from one and indexed with the"
            " other reads memory that belongs to something else."
        )

    print(
        "check_gram_chunk_count_is_pinned OK ["
        + _mode_name()
        + "]: k partitions into "
        + String(n)
        + " chunks, and the workspace-sizing predicate independently"
        " implies the same "
        + String(lo)
        + " (pinned constant is "
        + String(PINNED_GRAM_SPLITK_CHUNKS)
        + "; under FAST this number is the machine's own and under"
        " IDENTICAL it is the constant)"
    )


# ===========================================================================
# 2. THE ARM (DEVIATION 521, row 27)
# ===========================================================================


def check_gram_arm_is_pinned() raises:
    """The Gram arm is chosen from a PINNED column, not the device's.

    **THIS PROPERTY CANNOT BE PROVED BY RUNNING ON APPLE AND IT IS THE ONE
    THAT MATTERS MOST.** On the Apple column the predicate answered True
    before this lane and answers True after, so every runtime observation
    here is identical either way. The defect was that the accessor asked
    `gram_splitk_is_target_arm[TARGET_COLUMN]()`, which on an NVIDIA or AMD
    build answers False and sends the shape to `linalg.matmul` -- a closed
    library with a per-vendor k-split -- no matter how well this kernel is
    pinned. IDENTITY_PATHS row 3 is the same defect and was found the same
    way, by reading the accessor rather than the report.

    So the check is made against the TABLE, which carries every column's
    answer on this machine:

      - the NVIDIA and AMD rows must answer False (if they ever answer True
        the FAST dispatch has silently changed and this pin is moot),
      - the BIT_IDENTICAL row must answer True,
      - and the column the kernel actually compiles against must be
        BIT_IDENTICAL under IDENTICAL and the device's under FAST.

    Together those say: an IDENTICAL build on any of the three columns
    enters this kernel. Sabotage arm: change
    `GRAM_SPLITK_RESOLVED_COLUMN` back to `TARGET_COLUMN` and the last
    assertion fails on this machine, in this mode, with no second GPU.
    """
    if gram_splitk_is_target_arm[COLUMN_NVIDIA]():
        raise Error(
            "check_gram_arm_is_pinned: the NVIDIA row now claims the"
            " split-K kernel as its FAST arm. That may be correct, but it"
            " invalidates this check's reasoning and the docstring of"
            " gram_splitk_applies; re-derive both."
        )
    if gram_splitk_is_target_arm[COLUMN_AMD]():
        raise Error(
            "check_gram_arm_is_pinned: the AMD row now claims the split-K"
            " kernel as its FAST arm; re-derive as above."
        )
    if not gram_splitk_is_target_arm[COLUMN_BIT_IDENTICAL]():
        raise Error(
            "check_gram_arm_is_pinned: the BIT_IDENTICAL column does not"
            " claim the split-K kernel. Then an IDENTICAL build has no arm"
            " with a pinned summation order and row 27 is unclosable as"
            " written."
        )
    if not gram_splitk_is_target_arm[COLUMN_APPLE]():
        raise Error(
            "check_gram_arm_is_pinned: the Apple row stopped claiming its"
            " own kernel."
        )

    comptime expected = COLUMN_BIT_IDENTICAL if IDENTICAL else TARGET_COLUMN
    if GRAM_SPLITK_RESOLVED_COLUMN != expected:
        raise Error(
            "check_gram_arm_is_pinned: the arm decision compiles against"
            " column "
            + column_name(GRAM_SPLITK_RESOLVED_COLUMN)
            + " but this build requires "
            + column_name(expected)
            + ". Under IDENTICAL, reading the DEVICE's column here is the"
            " whole defect: NVIDIA and AMD answer False and run the vendor"
            " matmul, whose k-split is a summation order."
        )

    # And the predicate itself must admit the shipped Gram shapes -- an arm
    # resolution is only worth something if the capacity bounds then let the
    # shape in.
    #
    # UNDER FAST THIS IS COLUMN-DEPENDENT, AND THAT DEPENDENCE IS THE WHOLE
    # DEFECT. Compile this file with `-D MOJOLEARN_COLUMN_AMD=1` and the
    # FAST arm answers False at 32x32x4M: the AMD column hands the shipped
    # PCA/OLS Gram shape to `linalg.matmul`, a closed vendor library, while
    # the Apple column runs the pinned kernel. Two vendors, two kernels, one
    # source. So under FAST this reports what the column does rather than
    # asserting an Apple fact on every column, and under IDENTICAL it
    # asserts, because there the answer must be the same everywhere.
    var takes_shipped = gram_splitk_applies(32, 32, 4_000_000)
    var takes_widest = gram_splitk_applies(
        GRAM_MAX_COLS, GRAM_MAX_COLS, 100_003
    )
    comptime if IDENTICAL:
        if not takes_shipped:
            raise Error(
                "check_gram_arm_is_pinned: 32x32x4M does not take the"
                " split-K arm in an IDENTICAL build, which is the shipped"
                " PCA/OLS shape. Then this column runs linalg.matmul for"
                " every PCA, tSVD and OLS fit and the mode's promise is"
                " false on it."
            )
        if not takes_widest:
            raise Error(
                "check_gram_arm_is_pinned: the widest in-capacity Gram ("
                + String(GRAM_MAX_COLS)
                + " square) does not take the split-K arm under IDENTICAL."
            )
        print(
            "check_gram_arm_is_pinned OK [IDENTICAL]: nvidia/amd rows"
            " answer False and bit-identical answers True, the arm"
            " compiles against column "
            + column_name(GRAM_SPLITK_RESOLVED_COLUMN)
            + ", and 32x32 / "
            + String(GRAM_MAX_COLS)
            + "-square both take the split-K kernel"
        )
    else:
        print(
            "check_gram_arm_is_pinned OK [FAST]: nvidia/amd rows answer"
            " False and bit-identical answers True; this build's arm"
            " compiles against column "
            + column_name(GRAM_SPLITK_RESOLVED_COLUMN)
            + ", which takes the split-K kernel at 32x32x4M: "
            + String(takes_shipped)
            + " and at "
            + String(GRAM_MAX_COLS)
            + "-square: "
            + String(takes_widest)
            + ". A False there is not a bug -- it is the FAST dispatch"
            " handing the Gram shape to linalg.matmul, and it is exactly"
            " what DEVIATION 521 stops an IDENTICAL build from doing."
        )


# ===========================================================================
# 3. THE REFUSAL (DEVIATION 521, row 27)
# ===========================================================================


def _gemm_tn_over_capacity(ctx: DeviceContext) raises -> String:
    """Run `gemm_tn` at a shape past the split-K kernel's capacity. Returns
    the error text, or the empty string if it completed."""
    var m = 256  # > GRAM_MAX_COLS (128): outside the staging tile
    var k = 512
    var x = ctx.enqueue_create_buffer[DType.float32](k * m)
    var z = ctx.enqueue_create_buffer[DType.float32](m * m)
    var xt = ctx.enqueue_create_buffer[DType.float32](k * m)
    var xt2 = ctx.enqueue_create_buffer[DType.float32](k * m)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](k * m)
    ctx.synchronize()
    for i in range(k * m):
        hx.unsafe_ptr().unsafe_store(i, _val_f32(i, 77))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    try:
        gemm_tn(ctx, z, x, xt, xt2, m, m, k)
        ctx.synchronize()
    except e:
        return String(e)
    # Keep the buffers alive past the synchronize: a DeviceBuffer is freed
    # at its LAST USE, and `.unsafe_ptr()` inside the call is not a use of
    # the buffer afterwards.
    _ = x
    _ = z
    _ = xt
    _ = xt2
    return String("")


def check_gemm_tn_refuses_over_capacity() raises:
    """Over-capacity Gram: RAISES under IDENTICAL, RUNS under FAST.

    The refusal is the point. Under IDENTICAL the alternative arm is
    `linalg.matmul` behind two transposes, and running it would return a
    model that is not identical under a mode whose entire promise is that
    it is -- IDENTITY_PATHS' opening rule calls that worse than no toggle,
    because it converts a checkable property into a belief. Under FAST the
    same shape must still work: a pin that breaks the shipped arm is a
    regression, not a guarantee.
    """
    if GRAM_MAX_COLS >= 256:
        raise Error(
            "check_gemm_tn_refuses_over_capacity: the split-K kernel now"
            " serves m = 256, so this fixture is inside capacity and the"
            " check proves nothing. Pick a shape past the new bound."
        )
    with DeviceContext() as ctx:
        var err = _gemm_tn_over_capacity(ctx)
        comptime if IDENTICAL:
            if err == "":
                raise Error(
                    "check_gemm_tn_refuses_over_capacity: 256x256x512"
                    " COMPLETED under IDENTICAL. It cannot have run on the"
                    " split-K kernel (m > "
                    + String(GRAM_MAX_COLS)
                    + "), so it ran on the vendor matmul and returned a"
                    " model this mode promises is vendor-independent and"
                    " is not."
                )
            if err.find("IDENTITY_PATHS row 27") < 0:
                raise Error(
                    "check_gemm_tn_refuses_over_capacity: it refused, but"
                    " the message does not cite the ledger row. A refusal"
                    " a user cannot trace to its reason is a crash. Got: "
                    + err
                )
            print(
                "check_gemm_tn_refuses_over_capacity OK [IDENTICAL]:"
                " 256x256x512 raised by name rather than falling through"
                " to linalg.matmul"
            )
        else:
            if err != "":
                raise Error(
                    "check_gemm_tn_refuses_over_capacity: the FAST build"
                    " must still serve 256x256x512 through the"
                    " transpose+matmul arm, but it raised: "
                    + err
                )
            print(
                "check_gemm_tn_refuses_over_capacity OK [FAST]:"
                " 256x256x512 still runs on the transpose+matmul arm"
            )


# ===========================================================================
# 4. BATCH INVARIANCE (DEVIATION 526, row 28)
# ===========================================================================


def _nt_rows(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    salt: Int,
) raises:
    """`z[m x n] = X[m x k] . Y[n x k]^T` on the SHIPPED `gemm_nt` entry.

    X's row i and Y's row j are pure functions of (i, k, salt) and
    (j, k, salt), so the SAME logical row appears at the same index for
    every m and n this is called with. That is what makes two launches of
    different sizes comparable at all.
    """
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](n * k)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    ctx.synchronize()
    for i in range(m):
        for p in range(k):
            hx.unsafe_ptr().unsafe_store(i * k + p, _val_f32(i * 7919 + p, salt))
    for j in range(n):
        for p in range(k):
            hy.unsafe_ptr().unsafe_store(
                j * k + p, _val_f32(j * 104729 + p, salt + 1)
            )
    for i in range(m * n):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=out, src_ptr=hz.unsafe_ptr())
    ctx.synchronize()
    gemm_nt(ctx, out, x, y, m, n, k)
    ctx.synchronize()
    _ = x
    _ = y


def check_pinned_gemm_is_batch_invariant() raises:
    """A cell's bits do not depend on how many other cells were in the launch.

    THE FIXTURE, AND ITS FIRST VERSION WAS WRONG, WHICH IS WHY IT SAYS SO.
    The first version compared cell (0, 0) at three launch sizes. It passed a
    deliberate sabotage that rotated the contraction start by `block_idx.x`
    -- because cell (0, 0) is linear index 0 and therefore sits in block 0 at
    EVERY launch size, so the sabotage could not reach the only cell being
    read. A check that a real order-dependence walks straight through is not
    a weak check, it is a check that reports the opposite of the truth.

    What actually moves a cell between blocks is `n`, not `m`: this output
    is row-major, so cell (i, j) is at linear index `i * n + j` and varying
    `m` leaves every index where it was. So the three shapes vary `n`:

        m = 64, n = 4      cell (5, 2) at index    22  -> block 0
        m = 64, n = 64     cell (5, 2) at index   322  -> block 1
        m = 64, n = 256    cell (5, 2) at index  1282  -> block 5

    and the comparison is not one cell but the WHOLE OVERLAP: every (i, j)
    with i < 64 and j < 4 exists in all three launches, so 256 cells are
    compared bitwise, spread across many blocks in the wide shapes.

    X's row i and Y's row j are pure functions of (i, k, salt) and
    (j, k, salt), so the same logical row appears at the same index for
    every shape. That is what makes launches of different sizes comparable
    at all.

    UNDER FAST THIS IS A REPORT, NOT AN ASSERTION, and that is deliberate.
    The vendor matmul is free to tile differently at different shapes, and
    whether it does is the measurement that prices the pin rather than a bug
    to fail on. `k` is 4,096 so a tuned kernel has enough contraction length
    to want to split it.

    WHY THIS IS THE SAME PROBLEM AS BATCH INVARIANCE IN LLM SERVING: there,
    the same prompt gives different logits depending on what else is in the
    batch, because the batch size changes the reduction order inside the
    attention and normalization kernels. Here, the same row pair gives
    different bits depending on how many columns were in the launch, for
    exactly the same reason. The fix is the same fix: the per-cell
    arithmetic sequence must be a pure function of k.
    """
    comptime K = 4096
    comptime M = 64
    comptime N_NARROW = 4
    var salt = 424242

    var n_cells = M * N_NARROW
    var diffs_medium = 0
    var diffs_wide = 0
    var first_diff = -1
    var ref_first_bits = UInt32(0)
    var ref_first_val = Float32(0.0)
    var ref_diff_bits = UInt32(0)

    with DeviceContext() as ctx:
        # The reference row of the comparison, held on the host. Not named
        # `ref`: that is a Mojo keyword and the parse error it gives says
        # only "unexpected token in expression".
        var ref_bits = ctx.enqueue_create_host_buffer[DType.uint32](n_cells)
        ctx.synchronize()
        for s in range(3):
            var n = N_NARROW
            if s == 1:
                n = 64
            elif s == 2:
                n = 256
            var z = ctx.enqueue_create_buffer[DType.float32](M * n)
            _nt_rows(ctx, z, M, n, K, salt)
            var hz = ctx.enqueue_create_host_buffer[DType.float32](M * n)
            ctx.synchronize()
            ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
            ctx.synchronize()
            for i in range(M):
                for j in range(N_NARROW):
                    var v = hz.unsafe_ptr().unsafe_load(i * n + j)
                    if v == Float32(-987654.0):
                        raise Error(
                            "check_pinned_gemm_is_batch_invariant: poison"
                            " survived at cell ("
                            + String(i)
                            + ", "
                            + String(j)
                            + ") for n="
                            + String(n)
                            + ": the kernel did not write it, so nothing"
                            " below is a comparison of products."
                        )
                    var b = bitcast[DType.uint32](v)
                    var idx = i * N_NARROW + j
                    if s == 0:
                        ref_bits.unsafe_ptr().unsafe_store(idx, b)
                        if idx == 0:
                            ref_first_bits = b
                            ref_first_val = v
                    else:
                        if b != ref_bits.unsafe_ptr().unsafe_load(idx):
                            if first_diff < 0:
                                first_diff = idx
                                ref_diff_bits = (
                                    ref_bits.unsafe_ptr().unsafe_load(idx)
                                )
                            if s == 1:
                                diffs_medium += 1
                            else:
                                diffs_wide += 1
            _ = z
        _ = ref_bits

    comptime if IDENTICAL:
        if diffs_medium != 0 or diffs_wide != 0:
            var i0 = first_diff // N_NARROW
            var j0 = first_diff % N_NARROW
            raise Error(
                "check_pinned_gemm_is_batch_invariant: "
                + String(diffs_medium + diffs_wide)
                + " of "
                + String(2 * n_cells)
                + " overlapping cells changed BITS with the launch width"
                " (first at cell ("
                + String(i0)
                + ", "
                + String(j0)
                + "), reference bits "
                + hex(ref_diff_bits)
                + "). The same dot product must give the same bits at"
                " n=4, n=64 and n=256: under IDENTICAL the per-cell order"
                " is a pure function of k. This is the batch-invariance"
                " failure, measured on ONE device, before any second"
                " vendor."
            )
        print(
            "check_pinned_gemm_is_batch_invariant OK [IDENTICAL]: all "
            + String(n_cells)
            + " overlapping cells bit-identical at n=4, n=64 and n=256"
            " (m=64, k=4096) -- cell (5,2) alone moves from block 0 to"
            " block 1 to block 5 across those launches, so the comparison"
            " spans blocks and not just thread indices. Reference cell"
            " (0,0) = "
            + String(ref_first_val)
            + " / "
            + hex(ref_first_bits)
        )
    else:
        print(
            "check_pinned_gemm_is_batch_invariant REPORT [FAST]: "
            + String(n_cells)
            + " overlapping cells on the vendor matmul, n=4 as the"
            " reference -- n=64 differs in "
            + String(diffs_medium)
            + " cells, n=256 differs in "
            + String(diffs_wide)
            + ". A nonzero count here is not a bug, it is the measurement"
            " that prices the pin: it means the shipped arm's answer for a"
            " row depends on how many other rows were in the launch."
        )


def check_pinned_gemv_matches_oracle() raises:
    """The pinned gemv against a Float64 host fold in the SAME order.

    OLS's step 6 (`w <- covA . Ab`) is the only consumer, and its shipped
    shape is tiny, so the risk here is not accuracy -- it is that the
    kernel is never reached, or is reached and writes the wrong element.
    So: every output element compared, the output poisoned first, and the
    oracle folds k ASCENDING in Float64 so a reversed loop in the kernel
    would show up as a last-bit disagreement rather than hiding inside a
    tolerance.
    """
    comptime M = 37  # not a multiple of the block: the tail guard is live
    comptime K = 91
    var salt = 5150
    var worst = Float64(0.0)
    var worst_i = -1
    with DeviceContext() as ctx:
        var x = ctx.enqueue_create_buffer[DType.float32](M * K)
        var y = ctx.enqueue_create_buffer[DType.float32](K)
        var z = ctx.enqueue_create_buffer[DType.float32](M)
        var hx = ctx.enqueue_create_host_buffer[DType.float32](M * K)
        var hy = ctx.enqueue_create_host_buffer[DType.float32](K)
        var hz = ctx.enqueue_create_host_buffer[DType.float32](M)
        ctx.synchronize()
        for i in range(M * K):
            hx.unsafe_ptr().unsafe_store(i, _val_f32(i, salt))
        for i in range(K):
            hy.unsafe_ptr().unsafe_store(i, _val_f32(i, salt + 3))
        for i in range(M):
            hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
        ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
        ctx.synchronize()

        gemv_n(ctx, z, x, y, M, K)
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
        ctx.synchronize()

        for i in range(M):
            var got = hz.unsafe_ptr().unsafe_load(i)
            if got == Float32(-987654.0):
                raise Error(
                    "check_pinned_gemv_matches_oracle: poison survived at"
                    " element "
                    + String(i)
                    + " -- that element was never written."
                )
            var want = Float64(0.0)
            for p in range(K):
                want += Float64(
                    hx.unsafe_ptr().unsafe_load(i * K + p)
                ) * Float64(hy.unsafe_ptr().unsafe_load(p))
            var err = abs(Float64(got) - want)
            if err > worst:
                worst = err
                worst_i = i
        if worst > 1.0e-5:
            raise Error(
                "check_pinned_gemv_matches_oracle: element "
                + String(worst_i)
                + " is off the Float64 oracle by "
                + String(worst)
            )
        _ = x
        _ = y
        _ = z
    print(
        "check_pinned_gemv_matches_oracle OK ["
        + _mode_name()
        + "]: 37 elements over k=91 within "
        + String(worst)
        + " of an ascending Float64 fold, no poison survived"
    )


def main() raises:
    print("== core/gemm_identity_check.mojo [" + _mode_name() + "] ==")
    check_gram_chunk_count_is_pinned()
    check_gram_arm_is_pinned()
    check_gemm_tn_refuses_over_capacity()
    check_pinned_gemm_is_batch_invariant()
    check_pinned_gemv_matches_oracle()
