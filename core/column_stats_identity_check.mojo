"""What `IDENTICAL` promises about the column statistics PCA and OLS ride on.

DEVIATION 523, IDENTITY_PATHS row 20's defect in row 20's own directory.
`core/column_stats.mojo` folded Float32 with `max.gpu.primitives.block.sum`
in TWO kernels -- `column_mean_kernel` (the mean subtracted from every row
before the covariance) and `xty_kernel` (`A^T b`, the right-hand side the
normal equations are solved against) -- and that library fold's internal
cross-lane stage follows the HARDWARE warp width: 32 on Apple and NVIDIA,
64 on AMD's CDNA wavefront. Float addition is not associative, so the AMD
column produced a different model. The ledger named this and left it for
this lane; this file is the gate that says it is now done.

NO RAFT COUNTERPART. `raft::stats::mean` reduces with whatever CUB gives it
on the one backend RAFT ships. The question "does the fold combine the same
partials in the same order on Metal, CUDA and HIP" only exists because we
ship three backends from one source.

WHY THE WIDTH PIN WAS NOT THE FIX
----------------------------------
DEVIATION 510 added `K_LIB_COLUMN_STATS` to
`lib_block_bounds_a_float_fold`, so the fold's WIDTH is one number on every
column under IDENTICAL. `check_column_stats_row_is_pinned` below asserts
that, because it is load-bearing and it is one table edit away from being
lost. But pinning the width does not change what the library does INSIDE
that width, and the AMD wavefront is inside it. The width pin and the fold
replacement are two halves and this file gates both.

WHAT A ONE-DEVICE CHECK CAN PROVE
----------------------------------
Not that two vendors agree -- only E1 can (`IDENTITY_PATHS.md`). What it
proves is that the pinned construction is REACHED, which is the defect this
repository has found repeatedly: a pin nobody's kernel calls. So
`check_column_stats_fold_shape` requires the DEVICE to equal a HOST halving
tree bit for bit under IDENTICAL, and it FIRST proves its fixture can tell
a halving tree apart from a sequential fold. A fixture of uniform or small
values cannot (measured, and it is why the generator below looks odd), and
on such a fixture a fold that ignored the pin entirely would pass.

Modelled on `cluster/mojo_only/kmeans_identity_check.mojo`, deliberately: a
second shape for the same property would be a second thing to get wrong.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_idx, thread_idx
from std.math import fma
from std.memory import bitcast

from core.column_stats import STATS_TPB, column_mean_kernel, xty_kernel
from core.pinned_reduce import pinned_block_sum
from mojo_only.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_BIT_IDENTICAL,
    COLUMN_NVIDIA,
    K_LIB_COLUMN_STATS,
    K_LIB_GEMM_CONTRACTION,
    K_LIB_JACOBI_EIGH,
    K_LIB_TRANSPOSE,
    lib_block_bounds_a_float_fold,
    lib_block_size_for,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, numeric_mode_name


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: The fold check runs at EXACTLY one row per thread, so nothing is summed
#: before the fold gets it and the fold's SHAPE is the only thing under
#: test. `n_rows == STATS_TPB` is what makes that true.
comptime FOLD_ROWS = STATS_TPB
comptime FOLD_COLS = 8

#: The contraction check needs a real accumulator chain, so each thread
#: takes 16 rows. Wider in columns than the fold check because the
#: separating-fixture count is reported per column.
comptime CONTRACT_ROWS = 16 * STATS_TPB
comptime CONTRACT_COLS = 32


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _hash64(i: Int, f: Int) -> UInt64:
    """A deterministic 64-bit mix. Not a port; the fixture generator."""
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(f + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _spread_full(i: Int, f: Int) -> Float32:
    """A value in [1, 2) with a FULL 24-bit mantissa.

    The FOLD checks want values whose ADDITION order matters. A short
    mantissa makes a 128-value halving fold agree exactly with a
    sequential one, and then the fixture cannot see a fold that ignored
    the pin. Same generator and same reasoning as
    `kmeans_identity_check._spread_full`.
    """
    var frac = Float32(Int(_hash64(i, f) & UInt64(0xFFFFFF))) / Float32(
        16777216.0
    )
    return Float32(1.0) + frac


def _spread_half(i: Int, f: Int) -> Float32:
    """A value in [1, 2) with its mantissa in the TOP 12 BITS ONLY.

    The CONTRACTION check wants values whose PRODUCTS carry a tail: two
    12-bit mantissas need 24 bits, which is exactly what a naive multiply
    rounds away and an `fma` keeps. Full-width mantissas do not reliably
    separate the two spellings -- measured in the k-means lane, 0 of 256
    rows -- because the accumulator's own rounding swallows the tail.
    """
    var frac = Float32(Int(_hash64(i, f) & UInt64(0xFFF))) / Float32(4096.0)
    return Float32(1.0) + frac


def _magnitude(r: Int) -> Float32:
    """A wide, EXACT scale, so magnitudes sit orders apart and the ORDER of
    the adds decides how much of each addend survives.

    `1 + (r % 11) * 4096` and not the k-means lane's `1 + (r % 7) * 1000`.
    The period matters and was MEASURED, not chosen for looks: with period
    7 the halving tree pairs `r` with `r + 64`, and `64 % 7 == 1`, so one
    of eight columns of this fixture folded to EXACTLY the sequential
    answer and the check refused itself (which is the guard working).
    11 does not divide any of the halving steps and all eight columns
    separate.

    Every multiplier is exactly representable in binary32, so it changes a
    value's exponent without touching its mantissa: the fixture's
    separating property comes from the spread, not from a rounding smuggled
    into the generator."""
    return Float32(1.0) + Float32(r % 11) * Float32(4096.0)


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _show(x: Float32) -> String:
    """`String(Float32)` does not round trip (`[[mojo-string-float-roundtrip]]`),
    so every number this file prints carries its hex bits beside it."""
    return String(x) + "/0x" + hex(_bits(x))


def _host_halving_fold(values: List[Float32]) -> Float32:
    """`pinned_block_sum`'s IDENTICAL arm, on the host, in Float32.

    `red[t] += red[t + step]` for `step = n/2 ... 1`, which is the whole
    kernel. Written out rather than approximated, because the point of the
    check is that the DEVICE takes this exact sequence of additions.
    """
    var red = values.copy()
    var step = len(red) // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return red[0]


def _host_sequential_fold(values: List[Float32]) -> Float32:
    """Ascending, one at a time. A DIFFERENT shape, used only to prove the
    fixture can tell two shapes apart."""
    var acc = Float32(0.0)
    for i in range(len(values)):
        acc = acc + values[i]
    return acc


def check_column_stats_row_is_pinned() raises:
    """DEVIATION 510's half: `K_LIB_COLUMN_STATS` is a NUMERIC row.

    The fold replacement below is the fix, and this is the half that keeps
    the fix meaningful: `pinned_block_sum[STATS_TPB]` folds a slab of
    `STATS_TPB` floats, so if `STATS_TPB` were free to differ per vendor
    the halving tree would have a different DEPTH on each of them and the
    pin would buy nothing. The row must stay classified as a float fold,
    the width must resolve to one number on every column under IDENTICAL,
    and it must be a power of two or the halving tree is not exact.

    The negative arm matters as much: rows that are NOT float folds must
    stay free, or "pin it, it's cheap" quietly turns the whole table into
    one column and the cost measurement stops meaning anything.
    """
    if not lib_block_bounds_a_float_fold[K_LIB_COLUMN_STATS]():
        raise Error(
            "check_column_stats_row_is_pinned: K_LIB_COLUMN_STATS is no"
            " longer classified as a float fold. Both kernels in"
            " core/column_stats.mojo fold Float32 at this width and feed"
            " PCA and OLS; unclassifying it lets a vendor number land in"
            " lib_block_size and split the columns silently."
        )
    if lib_block_bounds_a_float_fold[K_LIB_TRANSPOSE]():
        raise Error(
            "check_column_stats_row_is_pinned: K_LIB_TRANSPOSE is now"
            " pinned as a float fold. It moves data and folds nothing;"
            " pinning it costs speed for no identity."
        )
    # K_LIB_JACOBI_EIGH IS listed since 2026-08-23 (the fold was fixed
    # first -- DEVIATION 511/524, `pinned_block_sum` -- and then the row
    # went NUMERIC so that AMD's FAST Jacobi could read its 64-wide
    # wavefront while IDENTICAL keeps the floor's 32 on every vendor).
    # Until then this check asserted the OPPOSITE and was the sentence the
    # kernel_matrix docstring deleted; `check_jacobi_fold_width_is_pinned`
    # owns the per-mode expectation now. Asserted here so that un-listing
    # it again is loud.
    if not lib_block_bounds_a_float_fold[K_LIB_JACOBI_EIGH]():
        raise Error(
            "check_column_stats_row_is_pinned: K_LIB_JACOBI_EIGH is no"
            " longer classified as a float fold. Its width is the fold's"
            " AND the stride of the per-thread partials; unclassifying it"
            " lets a FAST vendor number (AMD = 64) reach an IDENTICAL build."
        )
    if lib_block_bounds_a_float_fold[K_LIB_GEMM_CONTRACTION]():
        raise Error(
            "check_column_stats_row_is_pinned: K_LIB_GEMM_CONTRACTION is"
            " now pinned here. Its numeric geometry is pinned separately"
            " (`PINNED_ACC_*`, `PINNED_KBLK`, `PINNED_VECLEN`), and two"
            " tables with an opinion about one kernel is the defect this"
            " matrix exists to prevent."
        )

    var w_apple = lib_block_size_for[K_LIB_COLUMN_STATS, COLUMN_APPLE]()
    var w_nv = lib_block_size_for[K_LIB_COLUMN_STATS, COLUMN_NVIDIA]()
    var w_amd = lib_block_size_for[K_LIB_COLUMN_STATS, COLUMN_AMD]()
    var w_id = lib_block_size_for[
        K_LIB_COLUMN_STATS, COLUMN_BIT_IDENTICAL
    ]()

    if STATS_TPB <= 0 or (STATS_TPB & (STATS_TPB - 1)) != 0:
        raise Error(
            "check_column_stats_row_is_pinned: STATS_TPB is "
            + String(STATS_TPB)
            + ", which is not a power of two. `pinned_block_sum`'s"
            " halving tree requires one -- an odd width drops the tail"
            " element silently rather than failing."
        )

    comptime if IDENTICAL_BUILD:
        if not (w_apple == w_id and w_nv == w_id and w_amd == w_id):
            raise Error(
                "check_column_stats_row_is_pinned (IDENTICAL): the fold"
                " width still differs per column -- apple "
                + String(w_apple)
                + ", nvidia "
                + String(w_nv)
                + ", amd "
                + String(w_amd)
                + " vs bit-identical "
                + String(w_id)
                + ". The accessor gate is not reached."
            )
        print(
            "check_column_stats_row_is_pinned OK (IDENTICAL): every column"
            " resolves K_LIB_COLUMN_STATS to",
            w_id,
            "and STATS_TPB is",
            STATS_TPB,
            "-- a power of two, so the halving tree is exact",
        )
    else:
        print(
            "check_column_stats_row_is_pinned OK (FAST): K_LIB_COLUMN_STATS"
            " is classified as a float fold; per-column widths apple",
            w_apple,
            "nvidia",
            w_nv,
            "amd",
            w_amd,
            "(identical column",
            w_id,
            "). STATS_TPB",
            STATS_TPB,
        )


def check_column_stats_fold_shape() raises:
    """DEVIATION 523. Both column folds have ONE shape on every vendor.

    THE GATE. Under IDENTICAL the device must equal a HOST halving tree bit
    for bit, in both kernels, in every column of the fixture. That is the
    reach proof: a `block.sum` left in place combines different partials
    and lands somewhere else entirely.

    THE GUARD THAT COMES FIRST. The fixture must separate a halving fold
    from a sequential one in EVERY column, or the equality above proves
    nothing -- on values where the two shapes agree, so would a fold that
    read the hardware's lane width. The check refuses rather than passes.

    Under FAST the fold is Modular's shape, which is theirs to change, so
    that arm requires only that the answer is numerically the same sum
    (a loose relative bound, which a broken port still fails) and REPORTS
    the bits. It also re-asserts the fixture separates, which is what gives
    the IDENTICAL arm's equality its teeth.

    `n_rows == STATS_TPB` on purpose: one row per thread, so nothing is
    accumulated before the fold and the fold shape is the only variable.
    """
    var ctx = DeviceContext()
    var n = FOLD_ROWS
    var c = FOLD_COLS

    var x = ctx.enqueue_create_buffer[DType.float32](n * c)
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var mu = ctx.enqueue_create_buffer[DType.float32](c)
    var xty = ctx.enqueue_create_buffer[DType.float32](c)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * c)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hmu = ctx.enqueue_create_host_buffer[DType.float32](c)
    var hxty = ctx.enqueue_create_host_buffer[DType.float32](c)
    ctx.synchronize()

    for r in range(n):
        hy.unsafe_ptr().unsafe_store(
            r, _spread_full(r, 7777) * _magnitude(r + 3)
        )
        for j in range(c):
            # Each column gets its own PHASE of the magnitude cycle, so
            # eight columns are eight independent chances for the fixture
            # to fail to separate rather than one repeated.
            hx.unsafe_ptr().unsafe_store(
                r * c + j, _spread_full(r, j) * _magnitude(r + 3 * j)
            )
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[column_mean_kernel](
        mu.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n),
        Int32(c),
        grid_dim=(c, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    ctx.enqueue_function[xty_kernel](
        xty.unsafe_ptr(),
        x.unsafe_ptr(),
        y.unsafe_ptr(),
        Int32(n),
        Int32(c),
        grid_dim=(c, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hmu.unsafe_ptr(), src_buf=mu)
    ctx.enqueue_copy(dst_ptr=hxty.unsafe_ptr(), src_buf=xty)
    ctx.synchronize()
    # KEEP A USE PAST THE SYNC. A `DeviceBuffer` is freed at its LAST use
    # (`[[mojo-buffer-freed-at-last-use]]`), and `.unsafe_ptr()` above is
    # the last one; without this the buffers can be released while the
    # copies are still in flight.
    _ = x^
    _ = y^
    _ = mu^
    _ = xty^

    var worst_mean_rel = Float32(0.0)
    var worst_xty_rel = Float32(0.0)
    var mean_off = 0
    var xty_off = 0
    var first_mean_halving = Float32(0.0)
    var first_mean_sequential = Float32(0.0)
    var first_xty_halving = Float32(0.0)
    var first_xty_sequential = Float32(0.0)
    var first_mean_got = Float32(0.0)
    var first_xty_got = Float32(0.0)

    for j in range(c):
        var colvals = List[Float32]()
        var products = List[Float32]()
        for r in range(n):
            var xv = hx.unsafe_ptr().unsafe_load(r * c + j)
            colvals.append(xv)
            # A plain product, never a mul-add: nothing here for the host
            # codegen to contract. The device's single `identical_mul_add`
            # per thread is `fma(x, y, 0.0)`, which is the correctly
            # rounded product and therefore these same bits.
            products.append(xv * hy.unsafe_ptr().unsafe_load(r))

        var mh = _host_halving_fold(colvals)
        var ms = _host_sequential_fold(colvals)
        var ph = _host_halving_fold(products)
        var ps = _host_sequential_fold(products)
        if mh == ms or ph == ps:
            raise Error(
                "check_column_stats_fold_shape: column "
                + String(j)
                + " cannot tell two fold shapes apart (mean halving "
                + _show(mh)
                + " vs sequential "
                + _show(ms)
                + ", xty halving "
                + _show(ph)
                + " vs sequential "
                + _show(ps)
                + "). On this fixture a fold that ignored the pin would"
                " pass. Fix the fixture, not the check."
            )

        # The oracle mirrors the kernel STATEMENT BY STATEMENT, because
        # row 10 requires the intermediates to be stored through `ftz`
        # rather than left inside one expression.
        var want_mean = ftz(ftz(mh) / Float32(n))
        var want_xty = ftz(ph)
        var got_mean = hmu.unsafe_ptr().unsafe_load(j)
        var got_xty = hxty.unsafe_ptr().unsafe_load(j)
        if j == 0:
            first_mean_halving = want_mean
            first_mean_sequential = ftz(ftz(ms) / Float32(n))
            first_xty_halving = want_xty
            first_xty_sequential = ftz(ps)
            first_mean_got = got_mean
            first_xty_got = got_xty
        if got_mean != want_mean:
            mean_off += 1
        if got_xty != want_xty:
            xty_off += 1
        var rm = abs(got_mean - want_mean) / abs(want_mean)
        var rx = abs(got_xty - want_xty) / abs(want_xty)
        if rm > worst_mean_rel:
            worst_mean_rel = rm
        if rx > worst_xty_rel:
            worst_xty_rel = rx

    comptime if IDENTICAL_BUILD:
        if mean_off != 0 or xty_off != 0:
            raise Error(
                "check_column_stats_fold_shape (IDENTICAL): the pin is NOT"
                " reached. column_mean_kernel disagrees with the host"
                " halving tree in "
                + String(mean_off)
                + " of "
                + String(c)
                + " columns and xty_kernel in "
                + String(xty_off)
                + ". Column 0: mean device "
                + _show(first_mean_got)
                + " vs host halving "
                + _show(first_mean_halving)
                + " (sequential would be "
                + _show(first_mean_sequential)
                + "); xty device "
                + _show(first_xty_got)
                + " vs host halving "
                + _show(first_xty_halving)
                + " (sequential "
                + _show(first_xty_sequential)
                + "). A `block.sum` left in place produces exactly this."
            )
        print(
            "check_column_stats_fold_shape OK (IDENTICAL):",
            c,
            "columns of both kernels equal the host halving tree exactly,"
            " and the fixture separates it from a sequential fold in every"
            " one of them. Column 0 mean",
            _show(first_mean_halving),
            "vs sequential",
            _show(first_mean_sequential),
            "| xty",
            _show(first_xty_halving),
            "vs sequential",
            _show(first_xty_sequential),
        )
    else:
        if worst_mean_rel > Float32(1.0e-4) or worst_xty_rel > Float32(
            1.0e-4
        ):
            raise Error(
                "check_column_stats_fold_shape (FAST): the library fold is"
                " not merely a different association -- it is a different"
                " ANSWER. Worst relative gap to the halving tree: mean "
                + String(worst_mean_rel)
                + ", xty "
                + String(worst_xty_rel)
                + ". That is a broken port, not a fold shape."
            )
        print(
            "check_column_stats_fold_shape OK (FAST): the library fold ran"
            " (column 0 mean",
            _show(first_mean_got),
            "| xty",
            _show(first_xty_got),
            "); host halving mean",
            _show(first_mean_halving),
            "vs sequential",
            _show(first_mean_sequential),
            "| host halving xty",
            _show(first_xty_halving),
            "vs sequential",
            _show(first_xty_sequential),
            ". The device differs from the host halving tree in",
            mean_off,
            "of",
            c,
            "mean columns and",
            xty_off,
            "of",
            c,
            "xty columns, and halving vs sequential differ in every"
            " column -- so the IDENTICAL arm's equality has teeth. Those"
            " two counts ARE the local strength of the reach proof: they"
            " are how many columns of this fixture would catch a"
            " `block.sum` left in an IDENTICAL build on THIS backend. The"
            " defect they stand in for -- AMD's 64-wide wavefront -- is"
            " not observable here at all; only E1 sees that one.",
        )


def xty_contraction_oracle_kernel(
    out_fma: MutPointer[Float32, MutAnyOrigin],
    out_naive: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`xty_kernel`'s accumulator in BOTH spellings, side by side, ON THE
    DEVICE.

    THE ORACLE IS ON THE DEVICE ON PURPOSE. A host model cannot tell the
    two spellings apart: this toolchain's HOST codegen contracts
    `acc + x * y` inside a loop, and it folds a float64 rewrite back to
    the float32 product first, so the contraction-proof spelling gets
    contracted anyway. The k-means lane measured this and deleted its host
    oracle; that finding is reused here rather than rediscovered.

    Everything except the multiply-add is held identical to the shipped
    kernel -- same stride, same `pinned_block_sum[STATS_TPB]`, same `ftz`
    -- so the ONLY difference between the two outputs is one rounding
    against two. Both arms are spelled out literally rather than routed
    through `identical_mul_add`, which would make both arms the same arm
    under IDENTICAL.
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc_f = Float32(0.0)
    var acc_n = Float32(0.0)
    var r = tid
    while r < n_rows:
        var xv = x.unsafe_load(r * n_cols + col)
        var yv = y.unsafe_load(r)
        acc_f = fma(xv, yv, acc_f)
        acc_n = acc_n + xv * yv
        r += STATS_TPB

    # Back-to-back folds are safe: `pinned_block_sum` carries a trailing
    # `barrier()` for exactly this, and both arms therefore see the same
    # shared slab in the same state.
    var sf = ftz(pinned_block_sum[STATS_TPB](acc_f))
    var sn = ftz(pinned_block_sum[STATS_TPB](acc_n))
    if tid == 0:
        out_fma.unsafe_store(col, sf)
        out_naive.unsafe_store(col, sn)


def check_xty_contraction_pin() raises:
    """IDENTITY_PATHS row 9 at `xty_kernel`'s accumulator. DEVIATION 523.

    Pinning the FOLD and leaving `acc += x * y` unpinned would still hand
    two vendors two different `A^T b`, because the contraction is a codegen
    decision no runtime row reaches. Under IDENTICAL the accumulator goes
    through `identical_mul_add`, which is one rounding by construction.

    THE SHAPE OF THE GATE, and why it reports where it cannot assert.
    Row 9's Apple sentence was CORRECTED on 2026-08-23: Metal through MAX
    contracts `a*b+c` too (FUSED on 1,629 of 1,629 built-to-separate
    patterns). On a backend that contracts both spellings the two oracle
    arms compile to the same instruction and no fixture can separate them,
    so the pin is bit-inert there -- exactly as `ftz` is bit-inert on an
    FTZ backend -- and its value is the backend that does NOT contract.
    The pin must still be REACHED for that to be worth anything, which is
    what the equality below tests: a kernel computing anything other than
    the one-rounding answer fails it whether or not the arms agree.
    """
    var ctx = DeviceContext()
    var n = CONTRACT_ROWS
    var c = CONTRACT_COLS

    var x = ctx.enqueue_create_buffer[DType.float32](n * c)
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var xty = ctx.enqueue_create_buffer[DType.float32](c)
    var ofma = ctx.enqueue_create_buffer[DType.float32](c)
    var onaive = ctx.enqueue_create_buffer[DType.float32](c)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * c)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hxty = ctx.enqueue_create_host_buffer[DType.float32](c)
    var hfma = ctx.enqueue_create_host_buffer[DType.float32](c)
    var hnaive = ctx.enqueue_create_host_buffer[DType.float32](c)
    ctx.synchronize()

    # HALF-WIDTH MANTISSAS, for the reason `_spread_half` documents: the
    # product of two of them needs 24 significant bits, which is precisely
    # the tail a naive multiply rounds away and an `fma` keeps.
    for r in range(n):
        hy.unsafe_ptr().unsafe_store(r, _spread_half(r, 4242))
        for j in range(c):
            hx.unsafe_ptr().unsafe_store(r * c + j, _spread_half(r, j))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[xty_kernel](
        xty.unsafe_ptr(),
        x.unsafe_ptr(),
        y.unsafe_ptr(),
        Int32(n),
        Int32(c),
        grid_dim=(c, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    ctx.enqueue_function[xty_contraction_oracle_kernel](
        ofma.unsafe_ptr(),
        onaive.unsafe_ptr(),
        x.unsafe_ptr(),
        y.unsafe_ptr(),
        Int32(n),
        Int32(c),
        grid_dim=(c, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hxty.unsafe_ptr(), src_buf=xty)
    ctx.enqueue_copy(dst_ptr=hfma.unsafe_ptr(), src_buf=ofma)
    ctx.enqueue_copy(dst_ptr=hnaive.unsafe_ptr(), src_buf=onaive)
    ctx.synchronize()
    _ = x^
    _ = y^
    _ = xty^
    _ = ofma^
    _ = onaive^

    var separating = 0
    var off_fma = 0
    var off_naive = 0
    for j in range(c):
        var got = hxty.unsafe_ptr().unsafe_load(j)
        var vf = hfma.unsafe_ptr().unsafe_load(j)
        var vn = hnaive.unsafe_ptr().unsafe_load(j)
        if vf != vn:
            separating += 1
        if got != vf:
            off_fma += 1
        if got != vn:
            off_naive += 1

    comptime if IDENTICAL_BUILD:
        if off_fma != 0:
            raise Error(
                "check_xty_contraction_pin (IDENTICAL): "
                + String(off_fma)
                + " of "
                + String(c)
                + " columns disagree with the ONE-ROUNDING oracle. Column"
                " 0: kernel "
                + _show(hxty.unsafe_ptr().unsafe_load(0))
                + " vs fma "
                + _show(hfma.unsafe_ptr().unsafe_load(0))
                + " vs naive "
                + _show(hnaive.unsafe_ptr().unsafe_load(0))
                + ". Under IDENTICAL every multiply-add on this path goes"
                " through `identical_mul_add`, so this is a broken pin or"
                " a broken port."
            )
        if separating == 0:
            print(
                "check_xty_contraction_pin OK (IDENTICAL):",
                c,
                "columns match the one-rounding oracle bit for bit. This"
                " backend CONTRACTS the naive spelling too (0 of",
                c,
                "columns separate the arms), so the pin is bit-inert here"
                " and buys its property on a backend that does not.",
            )
        else:
            print(
                "check_xty_contraction_pin OK (IDENTICAL):",
                c,
                "columns match the fused oracle and",
                off_naive,
                "differ from the unfused one, on a backend where",
                separating,
                "columns separate the two spellings",
            )
    else:
        if separating == 0:
            if off_fma != 0:
                raise Error(
                    "check_xty_contraction_pin (FAST): the two oracle arms"
                    " agree (this backend contracts both spellings) and"
                    " the shipped kernel matches NEITHER on "
                    + String(off_fma)
                    + " columns. Something other than the contraction"
                    " differs between the kernel and the oracle."
                )
            print(
                "check_xty_contraction_pin OK (FAST): this backend"
                " contracts both spellings (0 of",
                c,
                "columns separate the arms), and the shipped kernel"
                " matches both. The contraction pin is bit-inert on this"
                " column; the FAST bits are untouched by DEVIATION 523.",
            )
        else:
            print(
                "check_xty_contraction_pin REPORT (FAST):",
                separating,
                "of",
                c,
                "columns separate the two spellings on this backend; the"
                " shipped kernel differs from the fused arm in",
                off_fma,
                "columns and from the unfused arm in",
                off_naive,
                ". Which arm FAST takes is the backend's business -- this"
                " arm reports it rather than gating on it.",
            )


def check_transpose_grid_stride() raises:
    """DEVIATION 1883. `transpose_kernel` walks its row tiles with a
    grid-stride loop; this runs the SAME transpose at four different Y grid
    heights and demands they agree with the host, cell for cell.

    WHY IT CANNOT BE TESTED THE OBVIOUS WAY. The bug is that CUDA caps
    `grid_dim.y` at 65,535, so the kernel could not transpose a matrix
    taller than 65,535 * 32 = 2,097,120 rows -- the LAUNCH was rejected
    before a thread ran. Reproducing that literally needs a two-million-row
    fixture on a CUDA box, which is not a unit test.

    So the test attacks the FIX instead of the bug. Capping Y at 1, 2 or 3
    forces the loop to wrap many times at a tiny size, which is exactly the
    control flow a 4,000,000-row launch takes and which a full-height launch
    never enters. Y = n_row_tiles is the old single-pass behavior and is the
    control.

    AND THE CAPPED ARMS ARE THEMSELVES THE SABOTAGE. Against the code as it
    stood yesterday -- row tile taken straight from `block_idx.y`, no loop --
    the Y = 1 arm would transpose ONLY THE FIRST 32 ROWS and leave the rest
    of `dst` untouched. It cannot pass without the loop. There is no version
    of this check that is green on both the old kernel and the new one.

    HASHED VALUES, NOT A RAMP, per the standing rule: a transpose is a
    PERMUTATION, and a fixture whose cells are interchangeable cannot see a
    permutation go wrong. Every cell here is `_spread_full(r, c)`, distinct
    per (row, column), so a misplaced element is a mismatch at a named cell
    rather than a plausible-looking matrix.
    """
    from core.column_stats import TRANSPOSE_TILE, transpose_kernel

    var ctx = DeviceContext()
    # Deliberately NOT a multiple of the tile in either dimension, so the
    # ragged edges of the first and last tile are inside the test.
    var n_rows = 5 * TRANSPOSE_TILE + 7
    var n_cols = 2 * TRANSPOSE_TILE + 5
    var n_row_tiles = (n_rows + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    var n_col_tiles = (n_cols + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE

    var src = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var hsrc = ctx.enqueue_create_host_buffer[DType.float32](n_rows * n_cols)
    ctx.synchronize()
    for r in range(n_rows):
        for c in range(n_cols):
            hsrc.unsafe_ptr().unsafe_store(r * n_cols + c, _spread_full(r, c))
    ctx.enqueue_copy(dst_buf=src, src_ptr=hsrc.unsafe_ptr())
    ctx.synchronize()

    var heights = List[Int]()
    heights.append(n_row_tiles)   # the control: one block per row tile
    heights.append(1)             # every tile through ONE block
    heights.append(2)
    heights.append(3)

    var failures = 0
    for hi in range(len(heights)):
        var gy = heights[hi]
        var dst = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
        var hdst = ctx.enqueue_create_host_buffer[DType.float32](
            n_rows * n_cols
        )
        ctx.synchronize()
        # POISON THE DESTINATION. A kernel that writes nothing must fail,
        # not inherit a previous arm's answer or a zeroed allocation that
        # happens to match an all-zero expectation.
        for i in range(n_rows * n_cols):
            hdst.unsafe_ptr().unsafe_store(i, Float32(-7777.0))
        ctx.enqueue_copy(dst_buf=dst, src_ptr=hdst.unsafe_ptr())
        ctx.synchronize()

        ctx.enqueue_function[transpose_kernel](
            dst.unsafe_ptr(),
            src.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            grid_dim=(n_col_tiles, gy, 1),
            block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=hdst.unsafe_ptr(), src_buf=dst)
        ctx.synchronize()

        var bad = 0
        var first_r = -1
        var first_c = -1
        for r in range(n_rows):
            for c in range(n_cols):
                # dst is [n_cols x n_rows]; dst[c][r] must be src[r][c].
                var got = hdst.unsafe_ptr().unsafe_load(c * n_rows + r)
                var want = _spread_full(r, c)
                if _bits(got) != _bits(want):
                    if bad == 0:
                        first_r = r
                        first_c = c
                    bad += 1
        if bad != 0:
            failures += 1
            print(
                "  FAIL grid_dim.y=",
                gy,
                ": ",
                bad,
                " of ",
                n_rows * n_cols,
                " cells wrong, first at src[",
                first_r,
                "][",
                first_c,
                "] want ",
                _show(_spread_full(first_r, first_c)),
                " got ",
                _show(hdst.unsafe_ptr().unsafe_load(first_c * n_rows + first_r)),
            )
        else:
            print(
                "  ok   grid_dim.y=",
                gy,
                " (",
                (n_row_tiles + gy - 1) // gy,
                " row tiles per block): all ",
                n_rows * n_cols,
                " cells bit-exact",
            )
        _ = dst^
        _ = hdst^
    _ = src^
    _ = hsrc^

    if failures != 0:
        raise Error(
            "transpose_kernel grid-stride: "
            + String(failures)
            + " of "
            + String(len(heights))
            + " grid heights disagreed with the host. DEVIATION 1883."
        )
    print(
        "  transpose_kernel: ",
        n_rows,
        "x",
        n_cols,
        " transposed identically at ",
        len(heights),
        " grid heights including Y=1, which the pre-1883 kernel could not"
        " do at all.",
    )


def main() raises:
    print("core/column_stats identity checks, mode", _mode_name())
    check_column_stats_row_is_pinned()
    check_column_stats_fold_shape()
    check_xty_contraction_pin()
    check_transpose_grid_stride()
