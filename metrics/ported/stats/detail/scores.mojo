"""RAFT `cpp/include/raft/stats/detail/scores.cuh` (ebf9268): `r2_score`
(:46-73) and `accuracy_score` (:84-101). `regression_metrics` (:146-206,
mean/median absolute error) is NOT ported (UNPORTED.tsv).

accuracy_score, THEIRS (:90-100):

    eltwiseSub(diffs, predictions, ref_predictions, n)
    correctly_predicted = thrust::count(diffs, diffs + n, 0)
    return correctly_predicted * 1.0f / n

An INTEGER count and ONE float division. Ours fuses the subtraction and the
count into one kernel -- `diffs[i] == 0` is `predictions[i] == ref[i]`, and
a materialized difference array is a RAFT convenience, not arithmetic --
and lands the count through an integer atomic (order-free; contingency_
matrix.mojo's header). The division `Float32(count) / Float32(n)` is one
correctly-rounded op on every vendor (check-ieee-arith: div 0 wrong on
every column), so the whole metric is identity-safe in both modes with no
IDENTICAL arm. sklearn: `accuracy_score(y_true, y_pred) = mean(y_true ==
y_pred)`, the same number.

r2_score, THEIRS (:46-73):

    y_bar = mean(y)  = sum(y) * (1/n)                      stats::mean (mean.cuh:22,39)
    sse   = sum((y - y_hat)^2)                             eltwiseSub, powerScalar, thrust::reduce
    ssto  = sum((y - y_bar)^2)                             subtractDevScalar, powerScalar, thrust::reduce
    return 1.0 - sse / ssto

Three float reductions over n. DEVIATION 653: under IDENTICAL each is ONE
fixed tree (`metrics/mojo_only/pinned_sum.mojo`): `PINNED_SUM_W` chunks
folded as a halving tree, chunk totals folded ascending on the host, every
stored partial through `ftz`. `thrust::reduce`'s shape is whatever the
vendor's CUB does; `stats::mean`'s is a block fold plus `atomicAdd`
(`raft/stats/detail/mean.cuh`). Under FAST the per-chunk fold is
`block.sum` (the library's shape) and FAST bits are a report.

Two spellings of theirs are re-spelled and named:
  - `powerScalar(x, 2.0)` is `raft::pow(x, 2)` (`raft/linalg/detail/
    power.cuh`), a vendor `powf`. `x * x` is what it means and is one
    correctly-rounded multiply everywhere; `powf(x, 2)` is a vendor
    transcendental whose last bit is the vendor's. Ours is `x * x` in BOTH
    modes (FAST too: there is no vendor powf to be faithful to on Metal
    through MAX that is better than the multiply).
  - `y - y_hat` then square then sum: the square is a bare multiply, the
    sum a separate add in the tree, so no `a*b+c` seam exists to contract.
    The difference itself is stored through `ftz` under IDENTICAL.
The mean's division by n and the final `1 - sse/ssto` are one division
and one subtraction each, correctly rounded everywhere; `math_t` is
Float32 throughout as cuML's `r2_score_py(float*)` is.

`accuracy_score` with `n == 0` is `0 / 0` in theirs; ours REFUSES `n <= 0`
by name (a NaN must not reach the recorded scalar, IDENTITY_PATHS row 39,
and there is no accuracy of nothing to report).

=========================================================================
DEVIATION 657 (metrics lane, 2026-08-23): r2's UNDEFINED CASES ARE cuML's
OWN SURFACE'S `force_finite=True` VALUES, AND ANY OTHER NaN IS THE ONE
CANONICAL PAYLOAD.
=========================================================================
THEIRS (RAFT :72) returns `1 - sse / ssto` unguarded: a constant `y`
(`ssto == 0`) gives `-inf` (sse > 0) or `0 / 0 = NaN` (sse == 0, a
perfect prediction of a constant). cuML's OWN Python `r2_score`
(`python/cuml/cuml/metrics/regression.py`, the function cuML users call)
does not call this kernel at all and defaults `force_finite=True`, exactly
sklearn's: `ssto == 0` -> `1.0` when `sse == 0`, else `0.0`. OURS mirrors
that surface in `r2_epilogue`: it is the value the user of either library
sees. WHY HERE: `metrics.r2_score` is a RECORDED SCALAR and a computed NaN
carries the vendor's payload (row 39), so the raw spelling would give a
different card per vendor on one legal input. The second arm: an
OVERFLOW-SCALE `y` (`|y - y_hat|` or `|y - y_bar|` above `1.8e19`
squares to `+inf`) makes `sse` and `ssto` `+inf` and `inf / inf` is NaN
in theirs, in sklearn (force_finite guards only `ssto == 0`) and in ours;
that NaN is returned AS A NaN but through `canonicalize_nan` (pinned_sum.
mojo), so its bits are `0x7fc00000` on every vendor and host. No finite
r2 moves: both arms are compares on the finished sums. MEASURED:
`regression_metrics_check.mojo::check_r2_undefined_cases` (constant y
with y_hat == y -> 0x3f800000, with y_hat != y -> 0x00000000, overflow ->
0x7fc00000, both modes, the oracle through the same epilogue).
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.mojo_only.pinned_sum import (
    PINNED_SUM_TPB,
    PINNED_SUM_W,
    canonicalize_nan,
    chunk_count,
    host_fold_partials,
    linear_block_id,
    physical_block_count,
    virtual_block_sum,
)
from mojo_only.numerics import ftz


#: `N_THREADS 512` (:26) is the file's elementwise width. SCHEDULING.
comptime SCORES_TPB = 512


# ===========================================================================
# accuracy_score
# ===========================================================================


def count_equal_kernel(
    predictions: MutPointer[Int32, MutAnyOrigin],
    ref_predictions: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
    counter: MutPointer[Int32, MutAnyOrigin],
):
    """`eltwiseSub` + `thrust::count(== 0)` fused: one integer atomic per
    agreeing sample."""
    var i = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if i < Int(n):
        if predictions.unsafe_load(i) == ref_predictions.unsafe_load(i):
            _ = Atomic.fetch_add(counter.unsafe_offset(0), Int32(1))


def accuracy_score(
    ctx: DeviceContext,
    mut predictions: DeviceBuffer[DType.int32],
    mut ref_predictions: DeviceBuffer[DType.int32],
    n: Int,
) raises -> Float32:
    """`accuracy_score(predictions, ref_predictions, n, stream)` (:84-101)."""
    if n <= 0:
        raise Error(
            "accuracy_score: n must be positive, got "
            + String(n)
            + " (0 / 0 is refused by name)"
        )
    var count = count_correct(ctx, predictions, ref_predictions, n)
    # `correctly_predicted * 1.0f / n` (:99)
    return Float32(count) / Float32(n)


def count_correct(
    ctx: DeviceContext,
    mut predictions: DeviceBuffer[DType.int32],
    mut ref_predictions: DeviceBuffer[DType.int32],
    n: Int,
) raises -> Int:
    """The integer half, exposed so the check can gate it EXACTLY."""
    var counter = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(counter, Int32(0))
    var grid = ceildiv(n, SCORES_TPB)
    if grid < 1:
        grid = 1
    ctx.enqueue_function[count_equal_kernel](
        predictions.unsafe_ptr(),
        ref_predictions.unsafe_ptr(),
        Int32(n),
        counter.unsafe_ptr(),
        grid_dim=(grid, 1, 1),
        block_dim=(SCORES_TPB, 1, 1),
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=counter)
    ctx.synchronize()
    var c = Int(h.unsafe_ptr().unsafe_load(0))
    _ = h^
    _ = counter^
    return c


# ===========================================================================
# r2_score (DEVIATION 653)
# ===========================================================================


def sum_chunks_kernel[
    block_size: Int
](
    y: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
):
    """`stats::mean`'s sum: `partials[chunk] = tree(y[chunk*W : chunk*W+W])`.
    A physical block serves chunks `linear_block_id(), += physical_block_
    count()` so any grid shape covers `chunk_count(n)` chunks in the same
    tree."""
    comptime R = PINNED_SUM_W // block_size
    var tid = Int(thread_idx.x)
    var chunks = chunk_count(Int(n))
    var chunk = linear_block_id()
    while chunk < chunks:
        var vals = SIMD[DType.float32, R](0.0)
        comptime for r in range(R):
            var i = chunk * PINNED_SUM_W + tid + r * block_size
            if i < Int(n):
                vals[r] = y.unsafe_load(i)
        var total = virtual_block_sum[block_size](vals)
        if tid == 0:
            partials.unsafe_store(chunk, ftz(total))
        chunk += physical_block_count()


def sse_ssto_chunks_kernel[
    block_size: Int
](
    y: MutPointer[Float32, MutAnyOrigin],
    y_hat: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
    y_bar: Float32,
    sse_partials: MutPointer[Float32, MutAnyOrigin],
    ssto_partials: MutPointer[Float32, MutAnyOrigin],
):
    """`sse = sum((y - y_hat)^2)` and `ssto = sum((y - y_bar)^2)` as two
    trees of the same shape, one pass over the data. `y_bar` is the scalar
    the host folded from `sum_chunks_kernel` (theirs keeps it in a device
    scalar and reads it in `subtractDevScalar`; same bits, one fewer
    launch)."""
    comptime R = PINNED_SUM_W // block_size
    var tid = Int(thread_idx.x)
    var chunks = chunk_count(Int(n))
    var chunk = linear_block_id()
    while chunk < chunks:
        var se = SIMD[DType.float32, R](0.0)
        var st = SIMD[DType.float32, R](0.0)
        comptime for r in range(R):
            var i = chunk * PINNED_SUM_W + tid + r * block_size
            if i < Int(n):
                var yi = y.unsafe_load(i)
                var d1 = ftz(yi - y_hat.unsafe_load(i))
                var d2 = ftz(yi - y_bar)
                se[r] = ftz(d1 * d1)
                st[r] = ftz(d2 * d2)
        var sse = virtual_block_sum[block_size](se)
        var ssto = virtual_block_sum[block_size](st)
        if tid == 0:
            sse_partials.unsafe_store(chunk, ftz(sse))
            ssto_partials.unsafe_store(chunk, ftz(ssto))
        chunk += physical_block_count()


def _fold_partials(
    ctx: DeviceContext, mut partials: DeviceBuffer[DType.float32], chunks: Int
) raises -> Float32:
    var h = ctx.enqueue_create_host_buffer[DType.float32](chunks)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=partials)
    ctx.synchronize()
    var lst = List[Float32]()
    for c in range(chunks):
        lst.append(h.unsafe_ptr().unsafe_load(c))
    _ = h^
    return host_fold_partials(lst, chunks)


def r2_score(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float32:
    """`r2_score(y, y_hat, n, stream)` (:46-73) at the default launch."""
    return r2_score_launch[PINNED_SUM_TPB](ctx, y, y_hat, n, 0)


def r2_score_launch[
    block_size: Int
](
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
    grid_x_override: Int,
) raises -> Float32:
    """The launch-parameterized form for the invariance gate: `block_size`
    threads per block, and a grid of `grid_x_override` columns (0 = one
    block per chunk, 1-D) arranged 2-D so the same chunks are served by a
    different grid shape."""
    var parts = r2_score_parts[block_size](ctx, y, y_hat, n, grid_x_override)
    return parts[3]


def r2_score_parts[
    block_size: Int
](
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
    grid_x_override: Int,
) raises -> Tuple[Float32, Float32, Float32, Float32]:
    """`(y_bar, sse, ssto, r2)`. The three sums are exposed because `r2 =
    1 - sse/ssto` ABSORBS a last-bit move in either sum whenever `sse <<
    ssto` (measured: a shifted chunk partition moved sse and not r2 on
    the 4099-row fixture), so the checks gate the sums, not only the
    ratio -- and since 2026-08-24 so does the CARD, which recorded only
    the ratio through the E3 round 11 certification.

    The untraced entry. `r2_score_parts_traced` is the implementation;
    a disabled trace records nothing and costs one boolean test per
    would-be record."""
    var off = IdentityTrace.disabled()
    return r2_score_parts_traced[block_size](
        ctx, off, y, y_hat, n, grid_x_override
    )


def r2_score_parts_traced[
    block_size: Int
](
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
    grid_x_override: Int,
) raises -> Tuple[Float32, Float32, Float32, Float32]:
    """The same computation carrying a card: `metrics.r2.sum_partials`,
    `metrics.r2.sse_partials`, `metrics.r2.ssto_partials`, one f32 per
    chunk.

    WHY THE PARTIALS ARE RULE-LEGAL, AND WHY THEY ARE THE RIGHT STAGE.
    `core/identity_trace.mojo` rule 3 forbids hashing a MACHINE-SIZED
    SCRATCH, because two backends legitimately hold different amounts of
    it. These are not that. `chunk_count(n) = ceil(n / PINNED_SUM_W)` is a
    PURE FUNCTION OF n (`pinned_sum.mojo:72-74`, and its docstring says
    so): `PINNED_SUM_W` is a NUMERIC constant of that file ("a different W
    is a different tree"), not a launch width, and `block_size` and the
    grid shape reach only WHICH PHYSICAL BLOCK SERVES WHICH CHUNK, never
    which values share a chunk. So this buffer has the same length and the
    same contents under every legal launch -- which is exactly the
    property `check_r2_launch_invariant` proves at two block widths and
    two grid shapes -- and recording it cannot break the launch-invariance
    claim these cards are asserted under. The partition is the ALGORITHM's
    structure; the scheduler's choices are not in it.

    They are also the buffer the absorption HIDES IN. `sse` and `ssto` are
    single scalars folded from these; a chunk partial that moved and a
    host fold that rounded it away leaves the recorded sums equal and the
    partials unequal, which is the mamba lesson (a fold sabotage that
    moves 13 of 16 stages with the output bit-identical) applied to the
    one fold this lane owns."""
    if n <= 0:
        raise Error("r2_score: n must be positive, got " + String(n))
    var chunks = chunk_count(n)
    var gx = chunks if grid_x_override <= 0 else grid_x_override
    var gy = ceildiv(chunks, gx)
    var sum_p = ctx.enqueue_create_buffer[DType.float32](chunks)
    var sse_p = ctx.enqueue_create_buffer[DType.float32](chunks)
    var ssto_p = ctx.enqueue_create_buffer[DType.float32](chunks)
    # raft::stats::mean<false>(y_bar, y, 1, n): sum / n
    ctx.enqueue_function[sum_chunks_kernel[block_size]](
        y.unsafe_ptr(),
        Int32(n),
        sum_p.unsafe_ptr(),
        grid_dim=(gx, gy, 1),
        block_dim=(block_size, 1, 1),
    )
    trace.record_device[DType.float32](
        ctx, "metrics.r2.sum_partials", sum_p, chunks
    )
    var y_sum = _fold_partials(ctx, sum_p, chunks)
    # `mean.cuh:22,39`: `ratio = 1 / N` then `mul_const_op(ratio)` -- TWO
    # roundings, not `sum / n`. Copied, not improved.
    var ratio = ftz(Float32(1.0) / Float32(n))
    var y_bar = ftz(y_sum * ratio)
    ctx.enqueue_function[sse_ssto_chunks_kernel[block_size]](
        y.unsafe_ptr(),
        y_hat.unsafe_ptr(),
        Int32(n),
        y_bar,
        sse_p.unsafe_ptr(),
        ssto_p.unsafe_ptr(),
        grid_dim=(gx, gy, 1),
        block_dim=(block_size, 1, 1),
    )
    trace.record_device[DType.float32](
        ctx, "metrics.r2.sse_partials", sse_p, chunks
    )
    trace.record_device[DType.float32](
        ctx, "metrics.r2.ssto_partials", ssto_p, chunks
    )
    var sse = _fold_partials(ctx, sse_p, chunks)
    var ssto = _fold_partials(ctx, ssto_p, chunks)
    _ = sum_p^
    _ = sse_p^
    _ = ssto_p^
    return (y_bar, sse, ssto, r2_epilogue(sse, ssto))


@always_inline
def r2_epilogue(sse: Float32, ssto: Float32) -> Float32:
    """`return 1.0 - sse / ssto;` (:72) with DEVIATION 657's two guards.
    The oracle in regression_metrics_check.mojo calls this same function
    on its own sums, so what it gates is the SUMS; the epilogue is host
    arithmetic (one division, one subtraction, two compares)."""
    # DEVIATION 657: cuML's surface / sklearn `force_finite=True`.
    if ssto == Float32(0.0):
        return Float32(1.0) if sse == Float32(0.0) else Float32(0.0)
    var r2 = ftz(Float32(1.0) - ftz(sse / ssto))
    # DEVIATION 657: `inf / inf` from an overflow-scale y stays a NaN but
    # with ONE payload (IDENTITY_PATHS row 39).
    return canonicalize_nan(r2)
