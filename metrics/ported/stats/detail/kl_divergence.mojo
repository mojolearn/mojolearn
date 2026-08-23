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
overload; the double one is refused, UNPORTED.tsv).

DEVIATION 653 (pinned_sum.mojo's banner): the `mapThenSumReduce` fold --
a 256-thread CUB block sum plus `atomicAdd` of the block partials -- is
under IDENTICAL one fixed slab tree over `n` with the host folding the
chunk totals ascending. The per-term map runs on the device: the two logs
through `identical_log` (row 12), their difference stored through `ftz`,
the product `p * diff` one multiply stored through `ftz` (no addition in
the term, so no contraction seam; the tree's additions are separate).
FAST: the stdlib `log` and `block.sum`, a report.
"""

from std.gpu import thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.mojo_only.pinned_sum import (
    PINNED_SUM_TPB,
    PINNED_SUM_W,
    chunk_count,
    host_fold_partials,
    linear_block_id,
    physical_block_count,
    virtual_block_sum,
)
from mojo_only.numerics import ftz, identical_log


@always_inline
def kld_op(model_pdf: Float32, candidate_pdf: Float32) -> Float32:
    """`KLDOp` (:34-43), the per-term map, through the row-12 log."""
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
    twin) for the invariance gate."""
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
    _ = partials^
    return host_fold_partials(lst, chunks)
