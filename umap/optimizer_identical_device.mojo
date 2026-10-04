# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL UMAP layout optimizer on the device: one thread per vertex, one
epoch snapshot, every vertex's update a fixed-order fold.

Kernel-matrix row `umap_device_optimizer_for` (2026-09-09, lane/umap-optimizer).
The serial host loops (`umap/optimizer.mojo::optimize_layout_identical_reference`,
`umap/sparse_optimizer.mojo::optimize_sparse_layout_identical_reference`) apply every
attractive and repulsive move in program order into one embedding, so vertex
`v`'s epoch is a Gauss-Seidel sweep that depends on every earlier edge in the
epoch. That order has no parallel form, which is why the 100,000-row IDENTICAL
optimizer took 60 s where cuML's takes 0.3 s.

This file is the Jacobi form of the SAME update rule, written so its bits
are a function of the inputs alone:

* Positive non-self CSR edges in row-major order, ordinal `e`, are the
  edges; edge `(v, u)` is eligible at epoch `t` when
  `Int(Float32(t + 1) * s_e) > Int(Float32(t) * s_e)` with
  `s_e = ftz(Float32(Float64(w_e) / Float64(max_w)))` computed once on the
  host (the serial rule with the ratio rounded to Float32 before the epoch
  product; every product here is one IEEE multiply of two normals).
* Vertex `v` owns `destination[v]`. Its fold visits its own CSR row in
  order. For an eligible edge `(v, u)` it adds the attractive move TWICE:
  once as the head of `(v, u)` and once as the tail of the mirror edge
  `(u, v)`, which the serial rule also applies and which is bit-equal to the
  head move computed from the same snapshot (negation and `_clip` are
  exact, and the graph is validated symmetric). Then it draws
  `negative_sample_rate` vertices from Philox4x32-10 with counter
  `(e lo, e hi, epoch, slot // 4)` and key `seed`, lane `slot % 4`, reduced
  modulo `n` (`core/philox.mojo`, the same generator RAFT uses), and adds
  the repulsive move for each draw that is not `v` itself.
* Every arithmetic step is a row-9/row-10/row-12/row-49 seam call:
  `identical_mul_add`, `identical_mul`, `identical_div`, `identical_pow`,
  `ftz`; the accumulators are Float32 in program order. No atomics, no
  shared memory, no reduction across threads, so the block width and the
  grid are free (`UMAP_IDENTICAL_OPT_TPB`, default 128; the gate builds 64
  and 256 and compares fingerprints).

The bits differ from the host loops (Jacobi versus Gauss-Seidel), which is
recorded as a re-baseline of the UMAP cards. The opt-in host-loop switch
was removed (hr-optin-flags).
FAST keeps `umap/optimizer_fast.mojo` untouched (one attractive move per
edge, SplitMix64 negatives, stdlib pow); it is not compared to this.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import memcpy
from dbscan.impl.adjgraph.algo import exclusive_scan, scan_blocks_needed
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.kernel_matrix import TARGET_COLUMN, umap_device_optimizer_live_row_for
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_pow,
)
from core.philox import philox4x32_10


comptime UMAP_IDENTICAL_OPT_TPB = (
    64 if is_defined["MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_64"]() else (
        256 if is_defined["MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_256"]() else 128
    )
)

comptime UMAP_IDENTICAL_GRAD_CLIP = Float32(4.0)

# DEVIATION 2668 (2026-09-11, lane/knn-finish; kernel-matrix row
# `umap_device_optimizer_live_row_for`): within a vertex's fold, its own
# attractive and repulsive moves are applied to its running position as the
# fold visits them (cuML's per-vertex serial kernel,
# `simpl_set_embed/optimize_batch_kernel.cuh:569-577, 608-616`, which
# `umap.pyx:562-570` selects for a spectral fit), and only the mirror edge's
# tail move is still summed and applied at the end. The fold is still a pure
# function of the epoch snapshot with one writer per vertex, so the bits stay
# independent of launch width and vendor; they differ from the snapshot fold
# (a re-baseline, which is why the row decides it). Measured on taxi 100k
# (H200, 2026-09-11): the snapshot fold summed about twenty undamped moves
# from one point and scored sampled trustworthiness 0.906 where the serial
# host loop on the same graph and init scored 0.980; this fold scores 0.932.
#
# THE ROW IS OFF BY DEFAULT. On the second dataset the sign reverses:
# Istella-S 100k goes 0.9737 to 0.9636 trustworthiness and 0.4832 to 0.4264
# retention, with the time flat on both (1.003 taxi, 1.000 Istella-S).
# CONTRIBUTING.md (Performance claims) gates quality per dataset rather than on the
# average, so this cannot be a default; it is opt-in through
# `-D MOJOLEARN_UMAP_IDENTICAL_LIVE_ROW=1`.
# `-D MOJOLEARN_UMAP_IDENTICAL_SNAPSHOT_FOLD=1` forces the snapshot fold even
# then, and `-D MOJOLEARN_UMAP_LIVE_BOTH_ARM=1` (trial only) applies both
# attractive moves live (worse on taxi at 0.8907).
comptime UMAP_LIVE_ROW = umap_device_optimizer_live_row_for[
    TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
]()
comptime UMAP_LIVE_BOTH_ARM = is_defined["MOJOLEARN_UMAP_LIVE_BOTH_ARM"]()


#: The run-time-dimension kernel's per-vertex row width (UMAP_MAX_COMPONENTS).
comptime UMAP_RT_WIDTH = 32


def _clip(value: Float32) -> Float32:
    """`umap/optimizer.mojo::_clip`, repeated here so this module imports nothing from the host loops (they import this one)."""
    if value > UMAP_IDENTICAL_GRAD_CLIP:
        return UMAP_IDENTICAL_GRAD_CLIP
    if value < -UMAP_IDENTICAL_GRAD_CLIP:
        return -UMAP_IDENTICAL_GRAD_CLIP
    return value


def _finite(v: Float32) -> Bool:
    var bits = bitcast[DType.uint32](v)
    return ((bits >> UInt32(23)) & UInt32(0xFF)) != UInt32(0xFF)


def umap_identical_epoch_kernel[C: Int](
    source: MutPointer[Float32, MutAnyOrigin],
    row_offsets: MutPointer[UInt32, MutAnyOrigin],
    tails: MutPointer[UInt32, MutAnyOrigin],
    scaled: MutPointer[Float32, MutAnyOrigin],
    destination: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    epoch_in: Int32,
    alpha: Float32,
    negative_rate_in: Int32,
    neg2ab: Float32,
    rep2b: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
):
    """One epoch, one thread per vertex; see the module docstring."""
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if v >= n:
        return
    var epoch = Int(epoch_in)
    var epoch_f = Float32(epoch)
    var next_f = Float32(epoch + 1)
    var epoch_u = UInt32(epoch)
    var n_u = UInt32(n)
    var key = SIMD[DType.uint32, 2](
        UInt32(seed & 0xFFFFFFFF), UInt32((seed >> 32) & 0xFFFFFFFF)
    )
    var x = SIMD[DType.float32, 4](0.0)
    var acc = SIMD[DType.float32, 4](0.0)
    comptime for c in range(C):
        x[c] = source.unsafe_load(v * C + c)
    var begin = Int(row_offsets.unsafe_load(v))
    var end = Int(row_offsets.unsafe_load(v + 1))
    var rate = Int(negative_rate_in)
    for e in range(begin, end):
        var s = scaled.unsafe_load(e)
        if Int(next_f * s) <= Int(epoch_f * s):
            continue
        var u = Int(tails.unsafe_load(e))
        var delta = SIMD[DType.float32, 4](0.0)
        var d2 = Float32(0.0)
        comptime for c in range(C):
            delta[c] = ftz(x[c] - source.unsafe_load(u * C + c))
            d2 = ftz(identical_mul_add(delta[c], delta[c], d2))
        if d2 > Float32(0.0):
            var dp = identical_pow(d2, b)
            var coeff = identical_div(
                identical_mul(neg2ab, identical_div(dp, d2)),
                ftz(identical_mul_add(a, dp, Float32(1.0))),
            )
            comptime for c in range(C):
                var g = ftz(
                    identical_mul(alpha, _clip(ftz(identical_mul(coeff, delta[c]))))
                )
                comptime if UMAP_LIVE_BOTH_ARM:
                    # Trial arm only: both moves applied to the running row.
                    x[c] = ftz(x[c] + g)
                    x[c] = ftz(x[c] + g)
                elif UMAP_LIVE_ROW:
                    # DEVIATION 2668: the head move lands on the running
                    # position now, so the next edge's delta sees it; the
                    # mirror (tail) move stays deferred to the epilogue.
                    x[c] = ftz(x[c] + g)
                    acc[c] = ftz(acc[c] + g)
                else:
                    acc[c] = ftz(acc[c] + g)
                    acc[c] = ftz(acc[c] + g)
        var draw = SIMD[DType.uint32, 4](0)
        for j in range(rate):
            var lane = j & 3
            if lane == 0:
                draw = philox4x32_10(
                    SIMD[DType.uint32, 4](
                        UInt32(e & 0xFFFFFFFF),
                        UInt32((e >> 32) & 0xFFFFFFFF),
                        epoch_u,
                        UInt32(j >> 2),
                    ),
                    key,
                )
            var other = Int(draw[lane] % n_u)
            if other == v:
                continue
            var nd = SIMD[DType.float32, 4](0.0)
            var n2 = Float32(0.0)
            comptime for c in range(C):
                nd[c] = ftz(x[c] - source.unsafe_load(other * C + c))
                n2 = ftz(identical_mul_add(nd[c], nd[c], n2))
            if n2 > Float32(0.0):
                var np_ = identical_pow(n2, b)
                var coeff = identical_div(
                    rep2b,
                    identical_mul(
                        ftz(Float32(0.001) + n2),
                        ftz(identical_mul_add(a, np_, Float32(1.0))),
                    ),
                )
                comptime for c in range(C):
                    comptime if UMAP_LIVE_ROW or UMAP_LIVE_BOTH_ARM:
                        # DEVIATION 2668: the repulsive move lands on the
                        # running position, as cuML's serial kernel applies it.
                        x[c] = ftz(
                            x[c]
                            + ftz(
                                identical_mul(
                                    alpha, _clip(ftz(identical_mul(coeff, nd[c])))
                                )
                            )
                        )
                    else:
                        acc[c] = ftz(
                            acc[c]
                            + ftz(
                                identical_mul(
                                    alpha, _clip(ftz(identical_mul(coeff, nd[c])))
                                )
                            )
                        )
    comptime for c in range(C):
        destination.unsafe_store(v * C + c, ftz(x[c] + acc[c]))


def umap_identical_epoch_kernel_rt(
    source: MutPointer[Float32, MutAnyOrigin],
    row_offsets: MutPointer[UInt32, MutAnyOrigin],
    tails: MutPointer[UInt32, MutAnyOrigin],
    scaled: MutPointer[Float32, MutAnyOrigin],
    destination: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    epoch_in: Int32,
    alpha: Float32,
    negative_rate_in: Int32,
    neg2ab: Float32,
    rep2b: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
    c_in: Int32,
):
    """`umap_identical_epoch_kernel[C]` with the dimension C read at run
    time (n_components 1 and 4 to 32; lane/algos-decomp, 2026-09-27,
    DEVIATION 5322): the same statements in the same order, the per-vertex
    rows in local arrays of UMAP_RT_WIDTH instead of 4-wide registers. 2 and
    3 keep the comptime kernel, so their bits are unchanged."""
    var C = Int(c_in)
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if v >= n:
        return
    var epoch = Int(epoch_in)
    var epoch_f = Float32(epoch)
    var next_f = Float32(epoch + 1)
    var epoch_u = UInt32(epoch)
    var n_u = UInt32(n)
    var key = SIMD[DType.uint32, 2](
        UInt32(seed & 0xFFFFFFFF), UInt32((seed >> 32) & 0xFFFFFFFF)
    )
    var x = InlineArray[Float32, UMAP_RT_WIDTH](fill=Float32(0.0))
    var acc = InlineArray[Float32, UMAP_RT_WIDTH](fill=Float32(0.0))
    for c in range(C):
        x[c] = source.unsafe_load(v * C + c)
    var begin = Int(row_offsets.unsafe_load(v))
    var end = Int(row_offsets.unsafe_load(v + 1))
    var rate = Int(negative_rate_in)
    for e in range(begin, end):
        var s = scaled.unsafe_load(e)
        if Int(next_f * s) <= Int(epoch_f * s):
            continue
        var u = Int(tails.unsafe_load(e))
        var delta = InlineArray[Float32, UMAP_RT_WIDTH](fill=Float32(0.0))
        var d2 = Float32(0.0)
        for c in range(C):
            delta[c] = ftz(x[c] - source.unsafe_load(u * C + c))
            d2 = ftz(identical_mul_add(delta[c], delta[c], d2))
        if d2 > Float32(0.0):
            var dp = identical_pow(d2, b)
            var coeff = identical_div(
                identical_mul(neg2ab, identical_div(dp, d2)),
                ftz(identical_mul_add(a, dp, Float32(1.0))),
            )
            for c in range(C):
                var g = ftz(
                    identical_mul(alpha, _clip(ftz(identical_mul(coeff, delta[c]))))
                )
                comptime if UMAP_LIVE_BOTH_ARM:
                    # Trial arm only: both moves applied to the running row.
                    x[c] = ftz(x[c] + g)
                    x[c] = ftz(x[c] + g)
                elif UMAP_LIVE_ROW:
                    # DEVIATION 2668: the head move lands on the running
                    # position now, so the next edge's delta sees it; the
                    # mirror (tail) move stays deferred to the epilogue.
                    x[c] = ftz(x[c] + g)
                    acc[c] = ftz(acc[c] + g)
                else:
                    acc[c] = ftz(acc[c] + g)
                    acc[c] = ftz(acc[c] + g)
        var draw = SIMD[DType.uint32, 4](0)
        for j in range(rate):
            var lane = j & 3
            if lane == 0:
                draw = philox4x32_10(
                    SIMD[DType.uint32, 4](
                        UInt32(e & 0xFFFFFFFF),
                        UInt32((e >> 32) & 0xFFFFFFFF),
                        epoch_u,
                        UInt32(j >> 2),
                    ),
                    key,
                )
            var other = Int(draw[lane] % n_u)
            if other == v:
                continue
            var nd = InlineArray[Float32, UMAP_RT_WIDTH](fill=Float32(0.0))
            var n2 = Float32(0.0)
            for c in range(C):
                nd[c] = ftz(x[c] - source.unsafe_load(other * C + c))
                n2 = ftz(identical_mul_add(nd[c], nd[c], n2))
            if n2 > Float32(0.0):
                var np_ = identical_pow(n2, b)
                var coeff = identical_div(
                    rep2b,
                    identical_mul(
                        ftz(Float32(0.001) + n2),
                        ftz(identical_mul_add(a, np_, Float32(1.0))),
                    ),
                )
                for c in range(C):
                    comptime if UMAP_LIVE_ROW or UMAP_LIVE_BOTH_ARM:
                        # DEVIATION 2668: the repulsive move lands on the
                        # running position, as cuML's serial kernel applies it.
                        x[c] = ftz(
                            x[c]
                            + ftz(
                                identical_mul(
                                    alpha, _clip(ftz(identical_mul(coeff, nd[c])))
                                )
                            )
                        )
                    else:
                        acc[c] = ftz(
                            acc[c]
                            + ftz(
                                identical_mul(
                                    alpha, _clip(ftz(identical_mul(coeff, nd[c])))
                                )
                            )
                        )
    for c in range(C):
        destination.unsafe_store(v * C + c, ftz(x[c] + acc[c]))


def _launch_epoch[C: Int](
    ctx: DeviceContext,
    source: MutPointer[Float32, MutAnyOrigin],
    row_offsets: MutPointer[UInt32, MutAnyOrigin],
    tails: MutPointer[UInt32, MutAnyOrigin],
    scaled: MutPointer[Float32, MutAnyOrigin],
    destination: MutPointer[Float32, MutAnyOrigin],
    n_samples: Int,
    epoch: Int,
    alpha: Float32,
    negative_rate: Int,
    neg2ab: Float32,
    rep2b: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
) raises:
    ctx.enqueue_function[umap_identical_epoch_kernel[C]](
        source, row_offsets, tails, scaled, destination,
        Int32(n_samples), Int32(epoch), alpha, Int32(negative_rate),
        neg2ab, rep2b, a, b, seed,
        grid_dim=(
            (n_samples + UMAP_IDENTICAL_OPT_TPB - 1) // UMAP_IDENTICAL_OPT_TPB,
            1, 1,
        ),
        block_dim=(UMAP_IDENTICAL_OPT_TPB, 1, 1),
    )


def optimize_csr_layout_identical_device(
    ctx: DeviceContext,
    initial: List[Float32],
    row_offsets: List[UInt32],
    tails: List[UInt32],
    scaled: List[Float32],
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    learning_rate: Float32,
    negative_rate: Int,
    repulsion: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
) raises -> List[Float32]:
    """Run `n_epochs` device epochs over a validated positive-edge CSR (`row_offsets` has `n_samples + 1` entries; `scaled[e]` is the edge's weight over the graph maximum, flushed)."""
    if n_samples < 2 or (n_components < 1 or n_components > 32):
        raise Error("UMAP device optimizer supports 1 to 32 dimensions")
    if len(initial) != n_samples * n_components or len(row_offsets) != n_samples + 1:
        raise Error("UMAP device optimizer input shape mismatch")
    if len(tails) != len(scaled) or len(tails) == 0:
        raise Error("UMAP device optimizer graph has no non-self edges")
    if n_samples > 2147483647 or n_epochs > 2147483647 or negative_rate > 2147483647:
        raise Error("UMAP device optimizer scalar exceeds kernel Int32 range")
    if len(tails) > 4294967295:
        raise Error("UMAP device optimizer edge count exceeds CSR UInt32 range")
    if n_epochs < 1 or not (learning_rate > Float32(0.0)) or negative_rate < 0:
        raise Error("UMAP device optimizer parameters are invalid")
    var neg2ab = -Float32(2.0) * a * b
    var rep2b = Float32(2.0) * repulsion * b
    var h_initial = ctx.enqueue_create_host_buffer[DType.float32](len(initial))
    var h_offsets = ctx.enqueue_create_host_buffer[DType.uint32](len(row_offsets))
    var h_tails = ctx.enqueue_create_host_buffer[DType.uint32](len(tails))
    var h_scaled = ctx.enqueue_create_host_buffer[DType.float32](len(scaled))
    for i in range(len(initial)):
        h_initial.unsafe_ptr().unsafe_store(i, ftz(initial[i]))
    for i in range(len(row_offsets)):
        h_offsets.unsafe_ptr().unsafe_store(i, row_offsets[i])
    for i in range(len(tails)):
        h_tails.unsafe_ptr().unsafe_store(i, tails[i])
        h_scaled.unsafe_ptr().unsafe_store(i, scaled[i])
    var first = ctx.enqueue_create_buffer[DType.float32](len(initial))
    var second = ctx.enqueue_create_buffer[DType.float32](len(initial))
    var d_offsets = ctx.enqueue_create_buffer[DType.uint32](len(row_offsets))
    var d_tails = ctx.enqueue_create_buffer[DType.uint32](len(tails))
    var d_scaled = ctx.enqueue_create_buffer[DType.float32](len(scaled))
    ctx.enqueue_copy(dst_buf=first, src_ptr=h_initial.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_offsets, src_ptr=h_offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tails, src_ptr=h_tails.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_scaled, src_ptr=h_scaled.unsafe_ptr())
    var out = _umap_epochs_download(
        ctx, first, second, d_offsets, d_tails, d_scaled, len(initial), n_samples, n_components,
        n_epochs, learning_rate, negative_rate, neg2ab, rep2b, a, b, seed,
    )
    _ = h_initial^
    _ = h_offsets^
    _ = h_tails^
    _ = h_scaled^
    _ = first^
    _ = second^
    _ = d_offsets^
    _ = d_tails^
    _ = d_scaled^
    return out^


def _umap_epochs_download(
    ctx: DeviceContext,
    mut first: DeviceBuffer[DType.float32],
    mut second: DeviceBuffer[DType.float32],
    mut d_offsets: DeviceBuffer[DType.uint32],
    mut d_tails: DeviceBuffer[DType.uint32],
    mut d_scaled: DeviceBuffer[DType.float32],
    n_values: Int,
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    learning_rate: Float32,
    negative_rate: Int,
    neg2ab: Float32,
    rep2b: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
) raises -> List[Float32]:
    """The epoch launches of `optimize_csr_layout_identical_device` over
    buffers already on the device, then the embedding's one download."""
    for epoch in range(n_epochs):
        # The serial schedule's alpha, computed on the host in Float64 as
        # the host loops do, then handed to the kernel as one Float32.
        var alpha = learning_rate * Float32(
            Float64(n_epochs - epoch) / Float64(n_epochs)
        )
        var src = first.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var dst = second.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        if epoch % 2 == 1:
            src = second.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            dst = first.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_offsets = d_offsets.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_tails = d_tails.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_scaled = d_scaled.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        if n_components == 2:
            _launch_epoch[2](
                ctx, src, p_offsets, p_tails, p_scaled, dst, n_samples, epoch,
                alpha, negative_rate, neg2ab, rep2b, a, b, seed,
            )
        elif n_components == 3:
            _launch_epoch[3](
                ctx, src, p_offsets, p_tails, p_scaled, dst, n_samples, epoch,
                alpha, negative_rate, neg2ab, rep2b, a, b, seed,
            )
        else:
            ctx.enqueue_function[umap_identical_epoch_kernel_rt](
                src, p_offsets, p_tails, p_scaled, dst,
                Int32(n_samples), Int32(epoch), alpha, Int32(negative_rate),
                neg2ab, rep2b, a, b, seed, Int32(n_components),
                grid_dim=(
                    (n_samples + UMAP_IDENTICAL_OPT_TPB - 1) // UMAP_IDENTICAL_OPT_TPB,
                    1, 1,
                ),
                block_dim=(UMAP_IDENTICAL_OPT_TPB, 1, 1),
            )
    var host_out = ctx.enqueue_create_host_buffer[DType.float32](n_values)
    if n_epochs % 2 == 0:
        ctx.enqueue_copy(dst_ptr=host_out.unsafe_ptr(), src_buf=first)
    else:
        ctx.enqueue_copy(dst_ptr=host_out.unsafe_ptr(), src_buf=second)
    ctx.synchronize()
    var out = List[Float32](length=n_values, fill=Float32(0.0))
    memcpy(dest=out.unsafe_ptr(), src=host_out.unsafe_ptr(), count=n_values)
    _ = host_out^
    return out^


#: lane/fam2-neighbors (2026-10-04), IDENTICAL on every vendor, default ON:
#: the optimizer's positive-edge CSR is compacted on the device. Before,
#: `optimize_sparse_layout_identical_device` walked every edge on one host
#: thread (a binary search of the partner row per edge for the symmetry
#: refusal, a float64 division per edge), appended to host lists, copied
#: them element by element into pinned buffers and uploaded. Here the graph
#: CSR goes up as it is; one launch counts each row's kept edges and checks
#: symmetry, the device scan turns the counts into offsets, one launch
#: writes the tails and the scaled weights. The scaled weight is
#: `ftz(identical_div(w, max))`: the correctly rounded float32 quotient,
#: which is what the float64 division rounded once to float32 gives (the
#: double rounding of a float32 quotient through float64 is exact), so the
#: optimizer reads the same words: no bit moves, the host column is
#: untouched. -D MOJOLEARN_IDN_UMAP_DEVICE_CSR_OFF (or MOJOLEARN_IDN_ALL_OFF)
#: restores the host walk.
#: lane cpu3-neighbors (2026-10-04): every mode and no _OFF arm (owner
#: rule: the host compaction is not a GPU route), so
#: `MOJOLEARN_IDN_UMAP_DEVICE_CSR_OFF` is retired.
comptime UMAP_IDN_DEVICE_CSR = True

comptime _UC_I32P = MutPointer[Int32, MutAnyOrigin]
comptime _UC_U32P = MutPointer[UInt32, MutAnyOrigin]
comptime _UC_F32P = MutPointer[Float32, MutAnyOrigin]
comptime _UC_TPB = 256


def umap_csr_count_kernel(
    offs: _UC_I32P, idx: _UC_U32P, vals: _UC_F32P, counts: _UC_I32P, flag: _UC_I32P, n_: Int32,
):
    """Row `head`'s kept edges (not the diagonal, weight > 0) counted;
    flag[0] set when a kept edge's partner (tail, head) holds another
    weight (every writer stores the same 1)."""
    var head = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if head >= Int(n_):
        return
    var c = Int32(0)
    for e in range(Int(offs.unsafe_load(head)), Int(offs.unsafe_load(head + 1))):
        var tail = Int(idx.unsafe_load(e))
        var w = vals.unsafe_load(e)
        if head == tail or not (w > Float32(0.0)):
            continue
        var lo = Int(offs.unsafe_load(tail))
        var end = Int(offs.unsafe_load(tail + 1))
        var hi = end
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if Int(idx.unsafe_load(mid)) < head:
                lo = mid + 1
            else:
                hi = mid
        var back = Float32(0.0)
        if lo < end and Int(idx.unsafe_load(lo)) == head:
            back = vals.unsafe_load(lo)
        if w != back:
            flag.unsafe_store(0, Int32(1))
        c += 1
    counts.unsafe_store(head, c)


def umap_csr_fill_kernel[SCALE: Bool](
    offs: _UC_I32P, idx: _UC_U32P, vals: _UC_F32P, out_off: _UC_I32P,
    row_offsets: _UC_U32P, tails: _UC_U32P, scaled: _UC_F32P, max_weight: Float32, n_: Int32,
):
    """Row `head`'s kept edges written at its scanned offset, in edge order;
    thread n writes the terminal offset. `SCALE` (IDENTICAL) stores the
    weight over the maximum, flushed; FAST stores the raw weight."""
    var head = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_)
    if head > n:
        return
    row_offsets.unsafe_store(head, UInt32(out_off.unsafe_load(head)))
    if head == n:
        return
    var at = Int(out_off.unsafe_load(head))
    for e in range(Int(offs.unsafe_load(head)), Int(offs.unsafe_load(head + 1))):
        var tail = Int(idx.unsafe_load(e))
        var w = vals.unsafe_load(e)
        if head == tail or not (w > Float32(0.0)):
            continue
        tails.unsafe_store(at, UInt32(tail))
        comptime if SCALE:
            scaled.unsafe_store(at, ftz(identical_div(w, max_weight)))
        else:
            scaled.unsafe_store(at, w)
        at += 1


# ---------------------------------------------------------------------------
# The optimizer's graph preparation on the device (lane cpu3-neighbors,
# 2026-10-04), every mode. Before, the callers walked the graph on one host
# thread: `validate_sparse_weights` (every edge), the FAST CSR compaction
# with a binary search per edge, the dense n x n validation and compaction,
# the initial embedding's finiteness walk and element-by-element staging.
# Here the graph goes up once and the refusals are decided from device
# words: a per-row first-error code with the smallest erring row (an integer
# min, so the first refusal of the serial walk), the maximum weight as an
# integer max over the bits of non-negative floats (exact, order-free),
# the init finiteness flag, the symmetry flag and the kept-edge count.
# The kept edges, their order and their scaled words are the host walk's, so
# no bit moves; the messages and their order are each caller's own.
# ---------------------------------------------------------------------------

comptime _UC_I64P = MutPointer[Int64, MutAnyOrigin]
comptime UMAP_NO_ROW = Int32(0x7FFFFFFF)


@always_inline
def _uc_finite(w: Float32) -> Bool:
    return (bitcast[DType.uint32](w) & UInt32(0x7F800000)) != UInt32(0x7F800000)


def umap_csr_validate_kernel(
    offs64: _UC_I64P, idx: _UC_U32P, vals: _UC_F32P, offs: _UC_I32P, codes: _UC_I32P,
    err_row: _UC_I32P, maxbits: _UC_I32P, n_: Int32, nnz_: Int32,
):
    """`validate_sparse_weights`'s row walk, one thread per row: code 1 for
    an offset out of range, 2 for a column out of order, out of range or
    the diagonal, 3 for a non-finite or negative weight, the first in edge
    order; the smallest erring row goes to `err_row`. Every positive weight
    raises `maxbits`. Thread `head` also narrows its offset to Int32 (thread
    n the terminal one)."""
    var head = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_)
    if head > n:
        return
    var begin64 = offs64.unsafe_load(head)
    offs.unsafe_store(head, begin64.cast[DType.int32]())
    if head == n:
        return
    var end64 = offs64.unsafe_load(head + 1)
    var code = Int32(0)
    if begin64 < Int64(0) or end64 < begin64 or end64 > Int64(Int(nnz_)):
        code = Int32(1)
    else:
        var previous = -1
        for e in range(Int(begin64), Int(end64)):
            var col = Int(idx.unsafe_load(e))
            if col <= previous or col >= n or col == head:
                code = Int32(2)
                break
            previous = col
            var w = vals.unsafe_load(e)
            if not _uc_finite(w) or w < Float32(0.0):
                code = Int32(3)
                break
            if w > Float32(0.0):
                _ = Atomic.max(maxbits, bitcast[DType.int32](w))
    codes.unsafe_store(head, code)
    if code != Int32(0):
        _ = Atomic.min(err_row, Int32(head))


def umap_dense_scan_kernel(
    w: _UC_F32P, counts: _UC_I32P, codes: _UC_I32P, err_row: _UC_I32P,
    flags: _UC_I32P, maxbits: _UC_I32P, n_: Int32,
):
    """Row `head` of a dense n x n graph in the serial walks' (head, tail)
    order: code 1 at the first invalid weight (non-finite or negative), 2 at
    the first weight its transpose cell differs from, whichever is first;
    the smallest erring row to `err_row`; `flags[0]` any invalid weight,
    `flags[1]` any asymmetric kept edge; kept edges (off the diagonal,
    positive) counted; every positive weight raises `maxbits`."""
    var head = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_)
    if head >= n:
        return
    var base = head * n
    var c = Int32(0)
    var code = Int32(0)
    for tail in range(n):
        var x = w.unsafe_load(base + tail)
        if not _uc_finite(x) or x < Float32(0.0):
            flags.unsafe_store(0, Int32(1))
            if code == Int32(0):
                code = Int32(1)
            continue
        if x > Float32(0.0):
            _ = Atomic.max(maxbits, bitcast[DType.int32](x))
        var kept = head != tail and x > Float32(0.0)
        if x != w.unsafe_load(tail * n + head):
            if code == Int32(0):
                code = Int32(2)
            if kept:
                flags.unsafe_store(1, Int32(1))
        if kept:
            c += 1
    counts.unsafe_store(head, c)
    codes.unsafe_store(head, code)
    if code != Int32(0):
        _ = Atomic.min(err_row, Int32(head))


def umap_dense_fill_kernel[SCALE: Bool](
    w: _UC_F32P, out_off: _UC_I32P, row_offsets: _UC_U32P, tails: _UC_U32P,
    weights: _UC_F32P, max_weight: Float32, n_: Int32,
):
    """Row `head`'s kept dense edges at its scanned offset in tail order;
    thread n writes the terminal offset."""
    var head = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_)
    if head > n:
        return
    row_offsets.unsafe_store(head, UInt32(out_off.unsafe_load(head)))
    if head == n:
        return
    var at = Int(out_off.unsafe_load(head))
    var base = head * n
    for tail in range(n):
        var x = w.unsafe_load(base + tail)
        if head == tail or not (x > Float32(0.0)):
            continue
        tails.unsafe_store(at, UInt32(tail))
        comptime if SCALE:
            weights.unsafe_store(at, ftz(identical_div(x, max_weight)))
        else:
            weights.unsafe_store(at, x)
        at += 1


def umap_init_prep_kernel(v: _UC_F32P, flag: _UC_I32P, n_: Int32, flush: Int32):
    """The initial embedding on the device: `flag[0]` set by a non-finite
    value; flushed in place when `flush` (IDENTICAL stages `ftz(x)`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_):
        return
    var x = v.unsafe_load(i)
    if not _uc_finite(x):
        flag.unsafe_store(0, Int32(1))
    if flush != Int32(0):
        v.unsafe_store(i, ftz(x))


def _uc_read_i32(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], at: Int, n: Int) raises -> List[Int32]:
    """`n` words of `buf` from `at` (a handful of refusal and sizing words)."""
    var out = List[Int32](length=n, fill=Int32(0))
    var view = buf.create_sub_buffer[DType.int32](at, n)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    _ = view^
    return out^


struct UmapDeviceGraph(Movable):
    """A prepared optimizer graph on the device: the initial embedding, the
    kept-edge CSR (UInt32 offsets and tails) and its weights (scaled for
    IDENTICAL, raw for FAST), the maximum weight and the kept-edge count."""

    var first: DeviceBuffer[DType.float32]
    var offsets: DeviceBuffer[DType.uint32]
    var tails: DeviceBuffer[DType.uint32]
    var weights: DeviceBuffer[DType.float32]
    var max_weight: Float32
    var n_edges: Int

    def __init__(
        out self,
        var first: DeviceBuffer[DType.float32],
        var offsets: DeviceBuffer[DType.uint32],
        var tails: DeviceBuffer[DType.uint32],
        var weights: DeviceBuffer[DType.float32],
        max_weight: Float32,
        n_edges: Int,
    ):
        self.first = first^
        self.offsets = offsets^
        self.tails = tails^
        self.weights = weights^
        self.max_weight = max_weight
        self.n_edges = n_edges


def _uc_upload_initial(ctx: DeviceContext, initial: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    """The initial embedding up once (a bulk copy through pinned memory)."""
    var n_init = len(initial)
    var h = ctx.enqueue_create_host_buffer[DType.float32](n_init)
    ctx.synchronize()
    memcpy(dest=h.unsafe_ptr(), src=initial.unsafe_ptr(), count=n_init)
    var first = ctx.enqueue_create_buffer[DType.float32](n_init)
    ctx.enqueue_copy(dst_buf=first, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^
    return first^


def umap_sparse_graph_to_device(
    ctx: DeviceContext,
    initial: List[Float32],
    offsets: List[Int],
    indices: List[UInt32],
    values: List[Float32],
    n_samples: Int,
    fast: Bool,
) raises -> UmapDeviceGraph:
    """`validate_sparse_weights`, the kept-edge compaction with its symmetry
    refusal and the init check, on the device. IDENTICAL (`fast` False)
    refuses in `optimize_sparse_layout_identical_on_device`'s order (graph,
    no positive edge, init, symmetry, no kept edge) and stores scaled
    weights; FAST in `optimize_sparse_layout_fast`'s (graph, symmetry, no
    positive edge, no kept edge, init) and stores raw ones."""
    var n = n_samples
    var nnz = len(indices)
    if n < 2 or len(offsets) != n + 1 or nnz != len(values):
        raise Error("UMAP sparse graph shape mismatch")
    if offsets[0] != 0 or offsets[n] != len(values):
        raise Error("UMAP sparse graph terminal offset mismatch")
    if n > 2147483646 or nnz > 2147483647 or len(initial) > 2147483647 or len(initial) < 1:
        raise Error("UMAP sparse graph exceeds the kernel Int32 range")
    var cap = max(nnz, 1)
    var n_init = len(initial)
    var first = _uc_upload_initial(ctx, initial)
    var h_off = ctx.enqueue_create_host_buffer[DType.int64](n + 1)
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](cap)
    var h_val = ctx.enqueue_create_host_buffer[DType.float32](cap)
    ctx.synchronize()
    memcpy(dest=h_off.unsafe_ptr(), src=offsets.unsafe_ptr().bitcast[Int64](), count=n + 1)
    if nnz > 0:
        memcpy(dest=h_idx.unsafe_ptr(), src=indices.unsafe_ptr(), count=nnz)
        memcpy(dest=h_val.unsafe_ptr(), src=values.unsafe_ptr(), count=nnz)
    var g_off64 = ctx.enqueue_create_buffer[DType.int64](n + 1)
    var g_off = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var g_idx = ctx.enqueue_create_buffer[DType.uint32](cap)
    var g_val = ctx.enqueue_create_buffer[DType.float32](cap)
    var codes = ctx.enqueue_create_buffer[DType.int32](n)
    var err = ctx.enqueue_create_buffer[DType.int32](1)
    # words: [0] max weight bits, [1] init not finite, [2] asymmetric kept edge
    var words = ctx.enqueue_create_buffer[DType.int32](3)
    ctx.enqueue_copy(dst_buf=g_off64, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=g_idx, src_ptr=h_idx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=g_val, src_ptr=h_val.unsafe_ptr())
    ctx.enqueue_memset(err, UMAP_NO_ROW)
    ctx.enqueue_memset(words, Int32(0))
    ctx.enqueue_function[umap_csr_validate_kernel](
        g_off64.unsafe_ptr(), g_idx.unsafe_ptr(), g_val.unsafe_ptr(), g_off.unsafe_ptr(),
        codes.unsafe_ptr(), err.unsafe_ptr(), words.unsafe_ptr(), Int32(n), Int32(nnz),
        grid_dim=((n + 1 + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
    )
    ctx.enqueue_function[umap_init_prep_kernel](
        first.unsafe_ptr(), words.unsafe_ptr().unsafe_offset(1), Int32(n_init),
        Int32(0) if fast else Int32(1),
        grid_dim=((n_init + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
    )
    var er = _uc_read_i32(ctx, err, 0, 1)[0]
    if er != UMAP_NO_ROW:
        var code = _uc_read_i32(ctx, codes, Int(er), 1)[0]
        if code == Int32(1):
            raise Error("UMAP sparse graph offset out of range")
        if code == Int32(2):
            raise Error("UMAP sparse graph needs unique sorted nonself columns")
        raise Error("UMAP sparse graph weight is invalid")
    var w0 = _uc_read_i32(ctx, words, 0, 2)
    var max_weight = bitcast[DType.float32](w0[0])
    var init_bad = w0[1] != Int32(0)
    if not fast:
        if not (max_weight > Float32(0.0)):
            raise Error("UMAP optimizer graph has no positive edges")
        if init_bad:
            raise Error("UMAP optimizer initialization is not finite")
    var counts = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[umap_csr_count_kernel](
        g_off.unsafe_ptr(), g_idx.unsafe_ptr(), g_val.unsafe_ptr(), counts.unsafe_ptr(),
        words.unsafe_ptr().unsafe_offset(2), Int32(n),
        grid_dim=((n + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
    )
    var out_off = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var bs = ctx.enqueue_create_buffer[DType.int32](scan_blocks_needed(n) + 1)
    exclusive_scan(ctx, out_off, counts, bs, n)
    var n_edges = Int(_uc_read_i32(ctx, out_off, n, 1)[0])
    var asym = _uc_read_i32(ctx, words, 2, 1)[0] != Int32(0)
    if fast:
        if asym:
            raise Error("UMAP FAST optimizer requires symmetric weights")
        if not (max_weight > Float32(0.0)):
            raise Error("UMAP FAST optimizer graph has no positive edges")
        if n_edges == 0:
            raise Error("UMAP FAST optimizer graph has no non-self edges")
        if init_bad:
            raise Error("UMAP FAST optimizer initialization is invalid")
    else:
        if asym:
            raise Error("UMAP device optimizer requires symmetric weights")
        if n_edges == 0:
            raise Error("UMAP device optimizer graph has no non-self edges")
    var d_offsets = ctx.enqueue_create_buffer[DType.uint32](n + 1)
    var d_tails = ctx.enqueue_create_buffer[DType.uint32](n_edges)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n_edges)
    if fast:
        ctx.enqueue_function[umap_csr_fill_kernel[False]](
            g_off.unsafe_ptr(), g_idx.unsafe_ptr(), g_val.unsafe_ptr(), out_off.unsafe_ptr(),
            d_offsets.unsafe_ptr(), d_tails.unsafe_ptr(), d_w.unsafe_ptr(), max_weight, Int32(n),
            grid_dim=((n + 1 + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[umap_csr_fill_kernel[True]](
            g_off.unsafe_ptr(), g_idx.unsafe_ptr(), g_val.unsafe_ptr(), out_off.unsafe_ptr(),
            d_offsets.unsafe_ptr(), d_tails.unsafe_ptr(), d_w.unsafe_ptr(), max_weight, Int32(n),
            grid_dim=((n + 1 + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
        )
    # the launches above hold raw pointers into the staging buffers
    ctx.synchronize()
    _ = h_off^
    _ = h_idx^
    _ = h_val^
    _ = g_off64^
    _ = g_off^
    _ = g_idx^
    _ = g_val^
    _ = codes^
    _ = err^
    _ = words^
    _ = counts^
    _ = out_off^
    _ = bs^
    return UmapDeviceGraph(first^, d_offsets^, d_tails^, d_w^, max_weight, n_edges)


def umap_dense_graph_to_device(
    ctx: DeviceContext,
    initial: List[Float32],
    weights: List[Float32],
    n_samples: Int,
    fast: Bool,
) raises -> UmapDeviceGraph:
    """The dense n x n graph's validation, compaction and init check on the
    device. IDENTICAL (`fast` False) refuses in
    `optimize_layout_identical_on_device`'s order (any invalid weight, no
    positive edge, init, asymmetric kept edge, no kept edge) and stores
    scaled weights; FAST in `optimize_layout_fast`'s (the first invalid or
    asymmetric cell in (head, tail) order, no positive edge, no kept edge,
    init) and stores raw ones."""
    var n = n_samples
    if n < 2 or len(weights) != n * n:
        raise Error("UMAP optimizer input shape mismatch")
    if n > 46340 or len(initial) > 2147483647 or len(initial) < 1:
        raise Error("UMAP dense graph exceeds the kernel Int32 range")
    var nn = n * n
    var n_init = len(initial)
    var first = _uc_upload_initial(ctx, initial)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](nn)
    ctx.synchronize()
    memcpy(dest=h_w.unsafe_ptr(), src=weights.unsafe_ptr(), count=nn)
    var g_w = ctx.enqueue_create_buffer[DType.float32](nn)
    var counts = ctx.enqueue_create_buffer[DType.int32](n)
    var codes = ctx.enqueue_create_buffer[DType.int32](n)
    var err = ctx.enqueue_create_buffer[DType.int32](1)
    # flags: [0] any invalid, [1] asymmetric kept edge, [2] init not finite
    var flags = ctx.enqueue_create_buffer[DType.int32](3)
    var maxbits = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=g_w, src_ptr=h_w.unsafe_ptr())
    ctx.enqueue_memset(err, UMAP_NO_ROW)
    ctx.enqueue_memset(flags, Int32(0))
    ctx.enqueue_memset(maxbits, Int32(0))
    ctx.enqueue_function[umap_dense_scan_kernel](
        g_w.unsafe_ptr(), counts.unsafe_ptr(), codes.unsafe_ptr(), err.unsafe_ptr(),
        flags.unsafe_ptr(), maxbits.unsafe_ptr(), Int32(n),
        grid_dim=((n + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
    )
    ctx.enqueue_function[umap_init_prep_kernel](
        first.unsafe_ptr(), flags.unsafe_ptr().unsafe_offset(2), Int32(n_init),
        Int32(0) if fast else Int32(1),
        grid_dim=((n_init + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
    )
    var out_off = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var bs = ctx.enqueue_create_buffer[DType.int32](scan_blocks_needed(n) + 1)
    exclusive_scan(ctx, out_off, counts, bs, n)
    var f = _uc_read_i32(ctx, flags, 0, 3)
    var er = _uc_read_i32(ctx, err, 0, 1)[0]
    var max_weight = bitcast[DType.float32](_uc_read_i32(ctx, maxbits, 0, 1)[0])
    var n_edges = Int(_uc_read_i32(ctx, out_off, n, 1)[0])
    var init_bad = f[2] != Int32(0)
    if fast:
        if er != UMAP_NO_ROW:
            if _uc_read_i32(ctx, codes, Int(er), 1)[0] == Int32(1):
                raise Error("UMAP FAST optimizer graph weight is invalid")
            raise Error("UMAP FAST optimizer requires symmetric weights")
        if not (max_weight > Float32(0.0)):
            raise Error("UMAP FAST optimizer graph has no positive edges")
        if n_edges == 0:
            raise Error("UMAP FAST optimizer graph has no non-self edges")
        if init_bad:
            raise Error("UMAP FAST optimizer initialization is invalid")
    else:
        if f[0] != Int32(0):
            raise Error("UMAP optimizer graph weight is invalid")
        if not (max_weight > Float32(0.0)):
            raise Error("UMAP optimizer graph has no positive edges")
        if init_bad:
            raise Error("UMAP optimizer initialization is not finite")
        if f[1] != Int32(0):
            raise Error("UMAP device optimizer requires symmetric weights")
        if n_edges == 0:
            raise Error("UMAP device optimizer graph has no non-self edges")
    var d_offsets = ctx.enqueue_create_buffer[DType.uint32](n + 1)
    var d_tails = ctx.enqueue_create_buffer[DType.uint32](n_edges)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n_edges)
    if fast:
        ctx.enqueue_function[umap_dense_fill_kernel[False]](
            g_w.unsafe_ptr(), out_off.unsafe_ptr(), d_offsets.unsafe_ptr(), d_tails.unsafe_ptr(),
            d_w.unsafe_ptr(), max_weight, Int32(n),
            grid_dim=((n + 1 + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[umap_dense_fill_kernel[True]](
            g_w.unsafe_ptr(), out_off.unsafe_ptr(), d_offsets.unsafe_ptr(), d_tails.unsafe_ptr(),
            d_w.unsafe_ptr(), max_weight, Int32(n),
            grid_dim=((n + 1 + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
        )
    ctx.synchronize()
    _ = h_w^
    _ = g_w^
    _ = counts^
    _ = codes^
    _ = err^
    _ = flags^
    _ = maxbits^
    _ = out_off^
    _ = bs^
    return UmapDeviceGraph(first^, d_offsets^, d_tails^, d_w^, max_weight, n_edges)


def umap_csr_pos_count_kernel(offs: _UC_I32P, vals: _UC_F32P, counts: _UC_I32P, n_: Int32):
    """Row `head`'s positive entries counted (a validated CSR)."""
    var head = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if head >= Int(n_):
        return
    var c = Int32(0)
    for e in range(Int(offs.unsafe_load(head)), Int(offs.unsafe_load(head + 1))):
        if vals.unsafe_load(e) > Float32(0.0):
            c += 1
    counts.unsafe_store(head, c)


def umap_csr_pos_fill_kernel(
    offs: _UC_I32P, idx: _UC_U32P, vals: _UC_F32P, out_off: _UC_I32P,
    rows: _UC_I32P, cols: _UC_I32P, ovals: _UC_F32P, n_: Int32,
):
    """Row `head`'s positive entries as COO `(head, col, value)` at its
    scanned offset, in edge order: the row-major positive COO."""
    var head = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if head >= Int(n_):
        return
    var at = Int(out_off.unsafe_load(head))
    for e in range(Int(offs.unsafe_load(head)), Int(offs.unsafe_load(head + 1))):
        var v = vals.unsafe_load(e)
        if v > Float32(0.0):
            rows.unsafe_store(at, Int32(head))
            cols.unsafe_store(at, Int32(idx.unsafe_load(e)))
            ovals.unsafe_store(at, v)
            at += 1


struct UmapDeviceCoo(Movable):
    """The positive entries of a validated UMAP graph as a row-major COO on
    the device (the spectral initialization's input)."""

    var nnz: Int
    var rows: DeviceBuffer[DType.int32]
    var cols: DeviceBuffer[DType.int32]
    var vals: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        nnz: Int,
        var rows: DeviceBuffer[DType.int32],
        var cols: DeviceBuffer[DType.int32],
        var vals: DeviceBuffer[DType.float32],
    ):
        self.nnz = nnz
        self.rows = rows^
        self.cols = cols^
        self.vals = vals^


def umap_positive_coo_device(
    ctx: DeviceContext,
    offsets: List[Int],
    indices: List[UInt32],
    values: List[Float32],
    n_samples: Int,
) raises -> UmapDeviceCoo:
    """`sparse_spectral_initialize`'s graph step on the device (lane
    cpu3-neighbors, 2026-10-04): `validate_sparse_weights`' refusals (the
    validation kernel the optimizer uses), then the positive entries in
    row-major order as a COO, the host loop's order; "no edges" when there
    are none. The CSR goes up once; the COO stays on the device."""
    var n = n_samples
    var nnz = len(indices)
    if n < 2 or len(offsets) != n + 1 or nnz != len(values):
        raise Error("UMAP sparse graph shape mismatch")
    if offsets[0] != 0 or offsets[n] != len(values):
        raise Error("UMAP sparse graph terminal offset mismatch")
    if n > 2147483646 or nnz > 2147483647:
        raise Error("UMAP sparse graph exceeds the kernel Int32 range")
    var cap = max(nnz, 1)
    var h_off = ctx.enqueue_create_host_buffer[DType.int64](n + 1)
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](cap)
    var h_val = ctx.enqueue_create_host_buffer[DType.float32](cap)
    ctx.synchronize()
    memcpy(dest=h_off.unsafe_ptr(), src=offsets.unsafe_ptr().bitcast[Int64](), count=n + 1)
    if nnz > 0:
        memcpy(dest=h_idx.unsafe_ptr(), src=indices.unsafe_ptr(), count=nnz)
        memcpy(dest=h_val.unsafe_ptr(), src=values.unsafe_ptr(), count=nnz)
    var g_off64 = ctx.enqueue_create_buffer[DType.int64](n + 1)
    var g_off = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var g_idx = ctx.enqueue_create_buffer[DType.uint32](cap)
    var g_val = ctx.enqueue_create_buffer[DType.float32](cap)
    var codes = ctx.enqueue_create_buffer[DType.int32](n)
    var err = ctx.enqueue_create_buffer[DType.int32](1)
    var maxbits = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=g_off64, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=g_idx, src_ptr=h_idx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=g_val, src_ptr=h_val.unsafe_ptr())
    ctx.enqueue_memset(err, UMAP_NO_ROW)
    ctx.enqueue_memset(maxbits, Int32(0))
    ctx.enqueue_function[umap_csr_validate_kernel](
        g_off64.unsafe_ptr(), g_idx.unsafe_ptr(), g_val.unsafe_ptr(), g_off.unsafe_ptr(),
        codes.unsafe_ptr(), err.unsafe_ptr(), maxbits.unsafe_ptr(), Int32(n), Int32(nnz),
        grid_dim=((n + 1 + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
    )
    var er = _uc_read_i32(ctx, err, 0, 1)[0]
    if er != UMAP_NO_ROW:
        var code = _uc_read_i32(ctx, codes, Int(er), 1)[0]
        if code == Int32(1):
            raise Error("UMAP sparse graph offset out of range")
        if code == Int32(2):
            raise Error("UMAP sparse graph needs unique sorted nonself columns")
        raise Error("UMAP sparse graph weight is invalid")
    var counts = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[umap_csr_pos_count_kernel](
        g_off.unsafe_ptr(), g_val.unsafe_ptr(), counts.unsafe_ptr(), Int32(n),
        grid_dim=((n + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
    )
    var out_off = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var bs = ctx.enqueue_create_buffer[DType.int32](scan_blocks_needed(n) + 1)
    exclusive_scan(ctx, out_off, counts, bs, n)
    var m = Int(_uc_read_i32(ctx, out_off, n, 1)[0])
    if m == 0:
        raise Error("UMAP spectral graph has no edges")
    var rows = ctx.enqueue_create_buffer[DType.int32](m)
    var cols = ctx.enqueue_create_buffer[DType.int32](m)
    var vals = ctx.enqueue_create_buffer[DType.float32](m)
    ctx.enqueue_function[umap_csr_pos_fill_kernel](
        g_off.unsafe_ptr(), g_idx.unsafe_ptr(), g_val.unsafe_ptr(), out_off.unsafe_ptr(),
        rows.unsafe_ptr(), cols.unsafe_ptr(), vals.unsafe_ptr(), Int32(n),
        grid_dim=((n + _UC_TPB - 1) // _UC_TPB, 1, 1), block_dim=(_UC_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = h_off^
    _ = h_idx^
    _ = h_val^
    _ = g_off64^
    _ = g_off^
    _ = g_idx^
    _ = g_val^
    _ = codes^
    _ = err^
    _ = maxbits^
    _ = counts^
    _ = out_off^
    _ = bs^
    return UmapDeviceCoo(m, rows^, cols^, vals^)


def _optimize_sparse_layout_device_csr(
    ctx: DeviceContext,
    initial_embedding: List[Float32],
    offsets: List[Int],
    indices: List[UInt32],
    values: List[Float32],
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    learning_rate: Float32,
    negative_rate: Int,
    repulsion: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
) raises -> List[Float32]:
    """UMAP_IDN_DEVICE_CSR: `optimize_sparse_layout_identical_device` with
    the validation, the compaction and the init check on the device
    (`umap_sparse_graph_to_device`), then the device epochs."""
    if n_samples < 2 or (n_components < 1 or n_components > 32):
        raise Error("UMAP device optimizer supports 1 to 32 dimensions")
    if len(initial_embedding) != n_samples * n_components:
        raise Error("UMAP device optimizer input shape mismatch")
    if n_samples > 2147483647 or n_epochs > 2147483647 or negative_rate > 2147483647:
        raise Error("UMAP device optimizer scalar exceeds kernel Int32 range")
    if n_epochs < 1 or not (learning_rate > Float32(0.0)) or negative_rate < 0:
        raise Error("UMAP device optimizer parameters are invalid")
    var neg2ab = -Float32(2.0) * a * b
    var rep2b = Float32(2.0) * repulsion * b
    var n_init = len(initial_embedding)
    var g = umap_sparse_graph_to_device(
        ctx, initial_embedding, offsets, indices, values, n_samples, False
    )
    var second = ctx.enqueue_create_buffer[DType.float32](n_init)
    var out = _umap_epochs_download(
        ctx, g.first, second, g.offsets, g.tails, g.weights, n_init, n_samples, n_components,
        n_epochs, learning_rate, negative_rate, neg2ab, rep2b, a, b, seed,
    )
    _ = second^
    _ = g^
    return out^


def optimize_sparse_layout_identical_device(
    ctx: DeviceContext,
    initial_embedding: List[Float32],
    offsets: List[Int],
    indices: List[UInt32],
    values: List[Float32],
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    initial_learning_rate: Float32,
    negative_sample_rate: Int,
    repulsion_strength: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
) raises -> List[Float32]:
    """CSR adapter: the caller has checked its scalars; this validates the
    graph (`validate_sparse_weights`'s refusals) and the initial embedding,
    compacts the positive non-self edges in row-major order (the serial
    loops' edge ordinals), checks symmetry, all on the device
    (`umap_sparse_graph_to_device`), and runs the device epochs."""
    return _optimize_sparse_layout_device_csr(
        ctx, initial_embedding, offsets, indices, values, n_samples,
        n_components, n_epochs, initial_learning_rate, negative_sample_rate,
        repulsion_strength, a, b, seed,
    )


def optimize_dense_layout_identical_device(
    ctx: DeviceContext,
    initial_embedding: List[Float32],
    weights: List[Float32],
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    initial_learning_rate: Float32,
    negative_sample_rate: Int,
    repulsion_strength: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
) raises -> List[Float32]:
    """Dense adapter (the `umap/optimizer.mojo` surface and the stage-identity
    fixtures): the dense graph's validation, its row-major positive-edge
    compaction and the init check on the device
    (`umap_dense_graph_to_device`), then the device epochs. The caller has
    checked its scalars."""
    if n_samples > 2147483647 or n_epochs > 2147483647 or negative_sample_rate > 2147483647:
        raise Error("UMAP device optimizer scalar exceeds kernel Int32 range")
    var neg2ab = -Float32(2.0) * a * b
    var rep2b = Float32(2.0) * repulsion_strength * b
    var n_init = len(initial_embedding)
    var g = umap_dense_graph_to_device(ctx, initial_embedding, weights, n_samples, False)
    var second = ctx.enqueue_create_buffer[DType.float32](n_init)
    var out = _umap_epochs_download(
        ctx, g.first, second, g.offsets, g.tails, g.weights, n_init, n_samples, n_components,
        n_epochs, initial_learning_rate, negative_sample_rate, neg2ab, rep2b, a, b, seed,
    )
    _ = second^
    _ = g^
    return out^
