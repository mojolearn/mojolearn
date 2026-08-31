# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""RAFT `cpp/include/raft/stats/detail/kl_divergence.cuh` (ebf9268).

THEIRS (:33-72):

    KLDOp(modelPDF, candidatePDF) = modelPDF == 0 ? 0
                                  : modelPDF * (log(modelPDF) - log(candidatePDF))
    mapThenSumReduce<DataT, KLDOp, size_t, 256>(d_KLDVal, size, op, modelPDF, candidatePDF)
    return h_KLDVal

`scipy.stats.entropy(pk, qk)` (what sklearn users call): `sum(pk * log(pk /
qk))` after normalizing both -- RAFT does NOT normalize and spells the log
of a quotient as a difference of logs; ours mirrors RAFT. `p > 0, q == 0`
is `p * (log p - (-inf)) = +inf` in both. `DataT` is Float32 (cuML's float
overload; the double one is refused, NOT_IMPLEMENTED.tsv).

DEVIATION 653 (pinned_sum.mojo's banner): the `mapThenSumReduce` fold --
a 256-thread CUB block sum plus `atomicAdd` of the block partials -- is
under IDENTICAL one fixed slab tree over `n` with the host folding the
chunk totals ascending. The per-term map runs on the device: the two logs
through `identical_log` (row 12), their difference stored through `ftz`,
the product `p * diff` one multiply stored through `ftz` (no addition in
the term, so no contraction seam; the tree's additions are separate).
FAST: the stdlib `log` and `block.sum`, a report.

=========================================================================
DEVIATION 658 (metrics lane, 2026-08-23): THE PER-TERM OPERANDS ARE
FLUSHED ON LOAD, AND A NaN RESULT CARRIES THE ONE CANONICAL PAYLOAD.
=========================================================================
(1) THE OPERAND FLUSH. `KLDOp` tests `modelPDF == 0` on the raw load. An
FTZ device (Apple) reads a SUBNORMAL p as zero and takes the `0` branch; a
denormal-honoring device (NVIDIA, AMD) reads it as nonzero, and under
IDENTICAL `identical_log` then flushes it (row 10) and returns `-inf`, so
the term is `p * (-inf - log q) = -inf`: ONE legal input (a probability
below 1.18e-38), two answers, a DIVERGENCE the first draft carried. The
row-10 policy is "flush operands to signed zero" (numerics.mojo's `ftz`
docstring); the first draft flushed every stored intermediate and missed
the two LOADS. OURS: `p = ftz(model_pdf)`, `q = ftz(candidate_pdf)` before
the compare, so every vendor sees the same zero and the term is `0`.
Bit-inert on Apple (the hardware already flushed); on the others it moves
the subnormal-p term from `-inf` to `0` under IDENTICAL (FAST is the
vendor's stdlib `log` of the unflushed p, a finite report). MEASURED:
`regression_metrics_check.mojo::check_kl_subnormal_p_and_nan` plants a
subnormal p and reads a FINITE sum bitwise equal to the host model on
Apple; without the flush the host model (which keeps denormals and
flushes inside `identical_log`) says `-inf` while Apple's device says
finite -- the check FAILS on Apple before the fix, which is how the hole
was found and is the sabotage of record.
(2) THE NaN. On every NON-NEGATIVE FINITE (p, q) the sum is NaN-free: `p >
0, q == 0` is `+inf` (as RAFT and scipy), `p == 0` is `0`, and an
all-`>= -finite` sum with at most `+inf` terms never forms `inf - inf`.
A NEGATIVE or non-finite entry is outside cuML's contract (P and Q are
probability distributions) and is not scanned for; should one reach the
kernel its NaN (`log` of a negative) is returned AS A NaN but through
`canonicalize_nan` (pinned_sum.mojo): `0x7fc00000` on every vendor and
host, never the vendor's payload (IDENTITY_PATHS row 39; NVIDIA's
arithmetic re-canonicalizes any NaN input to 0x7fffffff, so the payload
`identical_log` writes does not survive the multiply there). MEASURED:
the same check plants `q[7] = -1` and reads `0x7fc00000` in both modes.
"""

from std.gpu import thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.original.pinned_sum import (
    PINNED_SUM_TPB,
    PINNED_SUM_W,
    canonicalize_nan,
    chunk_count,
    host_fold_partials,
    linear_block_id,
    physical_block_count,
    virtual_block_sum,
)
from original.numerics import ftz, identical_log


@always_inline
def kld_op(model_pdf_in: Float32, candidate_pdf_in: Float32) -> Float32:
    """`KLDOp` (:34-43), the per-term map, through the row-12 log; the
    operands flushed on load (DEVIATION 658, IDENTITY_PATHS rows 10/39)."""
    var model_pdf = ftz(model_pdf_in)
    var candidate_pdf = ftz(candidate_pdf_in)
    if model_pdf == Float32(0.0):
        return Float32(0.0)
    var lp = ftz(identical_log(model_pdf))
    var lq = ftz(identical_log(candidate_pdf))
    return ftz(model_pdf * ftz(lp - lq))


def kld_chunks_kernel[
    block_size: Int
](
    model_pdf: MutPointer[Float32, MutAnyOrigin],
    candidate_pdf: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
):
    comptime R = PINNED_SUM_W // block_size
    var tid = Int(thread_idx.x)
    var chunks = chunk_count(Int(n))
    var chunk = linear_block_id()
    while chunk < chunks:
        var vals = SIMD[DType.float32, R](0.0)
        comptime for r in range(R):
            var i = chunk * PINNED_SUM_W + tid + r * block_size
            if i < Int(n):
                vals[r] = kld_op(
                    model_pdf.unsafe_load(i), candidate_pdf.unsafe_load(i)
                )
        var total = virtual_block_sum[block_size](vals)
        if tid == 0:
            partials.unsafe_store(chunk, ftz(total))
        chunk += physical_block_count()


def kl_divergence(
    ctx: DeviceContext,
    mut model_pdf: DeviceBuffer[DType.float32],
    mut candidate_pdf: DeviceBuffer[DType.float32],
    size: Int,
) raises -> Float32:
    """`kl_divergence(modelPDF, candidatePDF, size, stream)` (:56-72)."""
    return kl_divergence_launch[PINNED_SUM_TPB](
        ctx, model_pdf, candidate_pdf, size, 0
    )


def kl_divergence_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut model_pdf: DeviceBuffer[DType.float32],
    mut candidate_pdf: DeviceBuffer[DType.float32],
    size: Int,
) raises -> Float32:
    """The default launch, carrying a card. The same value `kl_divergence`
    returns, from one call."""
    return kl_divergence_launch_traced[PINNED_SUM_TPB](
        ctx, trace, model_pdf, candidate_pdf, size, 0
    )


def kl_divergence_launch[
    block_size: Int
](
    ctx: DeviceContext,
    mut model_pdf: DeviceBuffer[DType.float32],
    mut candidate_pdf: DeviceBuffer[DType.float32],
    size: Int,
    grid_x_override: Int,
) raises -> Float32:
    """The launch-parameterized form (scores.mojo::r2_score_launch's
    twin) for the invariance gate. The untraced entry;
    `kl_divergence_launch_traced` is the implementation."""
    var off = IdentityTrace.disabled()
    return kl_divergence_launch_traced[block_size](
        ctx, off, model_pdf, candidate_pdf, size, grid_x_override
    )


def kl_divergence_launch_traced[
    block_size: Int
](
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut model_pdf: DeviceBuffer[DType.float32],
    mut candidate_pdf: DeviceBuffer[DType.float32],
    size: Int,
    grid_x_override: Int,
) raises -> Float32:
    """The same fold carrying two stages.

    `metrics.kl.partials` (f32, `chunk_count(size)`) is EVERY per-term
    product this metric computes, folded once per chunk. Between the
    recorded inputs and the recorded answer there was nothing: `n` log
    terms and a fold, one scalar out. `chunk_count` is a PURE FUNCTION OF
    `size` (`pinned_sum.mojo:72-74`), so this is not the machine-sized
    scratch `core/identity_trace.mojo` rule 3 forbids -- the buffer has the
    same length and the same contents under every legal launch, which is
    what `check_kl_launch_invariant` proves at two block widths and two
    grid shapes. Nothing about the block width, the grid shape or the
    occupancy is in it; a card may record what the ALGORITHM decides and
    not what the SCHEDULER decides.

    `metrics.kl.sum_raw` (f32, 1) is the fold BEFORE `canonicalize_nan`.
    That epilogue is a WASHER: DEVIATION 658 maps every NaN, whatever the
    vendor made of it, onto the single payload `0x7fc00000`, which is
    correct for the returned scalar and destructive for an instrument. The
    pre-washer value is recorded so a NaN that arrives for two different
    reasons on two vendors is still two different stages. Both values are
    host floats; `size <= 0` is refused before either is formed."""
    if size <= 0:
        raise Error("kl_divergence: size must be positive, got " + String(size))
    var chunks = chunk_count(size)
    var gx = chunks if grid_x_override <= 0 else grid_x_override
    var gy = ceildiv(chunks, gx)
    var partials = ctx.enqueue_create_buffer[DType.float32](chunks)
    ctx.enqueue_function[kld_chunks_kernel[block_size]](
        model_pdf.unsafe_ptr(),
        candidate_pdf.unsafe_ptr(),
        Int32(size),
        partials.unsafe_ptr(),
        grid_dim=(gx, gy, 1),
        block_dim=(block_size, 1, 1),
    )
    var h = ctx.enqueue_create_host_buffer[DType.float32](chunks)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=partials)
    ctx.synchronize()
    var lst = List[Float32]()
    for c in range(chunks):
        lst.append(h.unsafe_ptr().unsafe_load(c))
    _ = h^
    trace.record_device[DType.float32](
        ctx, "metrics.kl.partials", partials, chunks
    )
    _ = partials^
    var raw = host_fold_partials(lst, chunks)
    trace.record_scalar_f32("metrics.kl.sum_raw", raw)
    # DEVIATION 658 (2): a NaN (only from an out-of-contract negative or
    # non-finite entry) leaves with ONE payload.
    return canonicalize_nan(raw)
