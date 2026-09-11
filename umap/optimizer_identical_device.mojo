# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL UMAP layout optimizer on the device: one thread per vertex, one
epoch snapshot, every vertex's update a fixed-order fold.

Kernel-matrix row `umap_device_optimizer_for` (2026-09-09, lane/umap-optimizer).
The serial host loops (`umap/optimizer.mojo::optimize_layout_identical`,
`umap/sparse_optimizer.mojo::optimize_sparse_layout_identical`) apply every
attractive and repulsive move in program order into one embedding, so vertex
`v`'s epoch is a Gauss-Seidel sweep that depends on every earlier edge in the
epoch. That order has no parallel form, which is why the 100,000-row IDENTICAL
optimizer took 60 s where cuML's takes 0.3 s (docs/lanes/HANDOFF_knn.md).

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
recorded as a re-baseline of the UMAP cards in docs/lanes/HANDOFF_umap.md;
`-D MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER=1` keeps the host loops.
FAST keeps `umap/optimizer_fast.mojo` untouched (one attractive move per
edge, SplitMix64 negatives, stdlib pow); it is not compared to this.
"""
from max.gpu.host import DeviceContext
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
# ENGINEERING_RULES section 9 gates quality per dataset rather than on the
# average, so this cannot be a default; it is opt-in through
# `-D MOJOLEARN_UMAP_IDENTICAL_LIVE_ROW=1`.
# `-D MOJOLEARN_UMAP_IDENTICAL_SNAPSHOT_FOLD=1` forces the snapshot fold even
# then, and `-D MOJOLEARN_UMAP_LIVE_BOTH_ARM=1` (trial only) applies both
# attractive moves live (worse on taxi at 0.8907).
comptime UMAP_LIVE_ROW = umap_device_optimizer_live_row_for[
    TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
]()
comptime UMAP_LIVE_BOTH_ARM = is_defined["MOJOLEARN_UMAP_LIVE_BOTH_ARM"]()


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
    if n_samples < 2 or (n_components != 2 and n_components != 3):
        raise Error("UMAP device optimizer supports 2D/3D layouts")
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
        else:
            _launch_epoch[3](
                ctx, src, p_offsets, p_tails, p_scaled, dst, n_samples, epoch,
                alpha, negative_rate, neg2ab, rep2b, a, b, seed,
            )
    var host_out = ctx.enqueue_create_host_buffer[DType.float32](len(initial))
    if n_epochs % 2 == 0:
        ctx.enqueue_copy(dst_ptr=host_out.unsafe_ptr(), src_buf=first)
    else:
        ctx.enqueue_copy(dst_ptr=host_out.unsafe_ptr(), src_buf=second)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(len(initial)):
        out.append(host_out.unsafe_ptr().unsafe_load(i))
    _ = h_initial^
    _ = h_offsets^
    _ = h_tails^
    _ = h_scaled^
    _ = first^
    _ = second^
    _ = d_offsets^
    _ = d_tails^
    _ = d_scaled^
    _ = host_out^
    return out^


def _scaled_weight(weight: Float32, max_weight: Float32) -> Float32:
    return ftz(Float32(Float64(weight) / Float64(max_weight)))


def _csr_weight_at(
    offsets: List[Int], indices: List[UInt32], values: List[Float32],
    row: Int, col: Int,
) -> Float32:
    var lo = offsets[row]
    var hi = offsets[row + 1]
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if Int(indices[mid]) < col:
            lo = mid + 1
        else:
            hi = mid
    if lo < offsets[row + 1] and Int(indices[lo]) == col:
        return values[lo]
    return Float32(0.0)


def optimize_sparse_layout_identical_device(
    ctx: DeviceContext,
    initial_embedding: List[Float32],
    offsets: List[Int],
    indices: List[UInt32],
    values: List[Float32],
    max_weight: Float32,
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
    """CSR adapter: the caller has validated the graph (`validate_sparse_weights`) and its scalars; this compacts the positive non-self edges in row-major order (the serial loops' edge ordinals), checks symmetry, and runs the device epochs."""
    var row_offsets = List[UInt32]()
    var tails = List[UInt32]()
    var scaled = List[Float32]()
    row_offsets.append(UInt32(0))
    for head in range(n_samples):
        for edge in range(offsets[head], offsets[head + 1]):
            var tail = Int(indices[edge])
            var weight = values[edge]
            if head == tail or not (weight > Float32(0.0)):
                continue
            if weight != _csr_weight_at(offsets, indices, values, tail, head):
                raise Error("UMAP device optimizer requires symmetric weights")
            tails.append(UInt32(tail))
            scaled.append(_scaled_weight(weight, max_weight))
        row_offsets.append(UInt32(len(tails)))
    return optimize_csr_layout_identical_device(
        ctx, initial_embedding, row_offsets^, tails^, scaled^, n_samples,
        n_components, n_epochs, initial_learning_rate, negative_sample_rate,
        repulsion_strength, a, b, seed,
    )


def optimize_dense_layout_identical_device(
    ctx: DeviceContext,
    initial_embedding: List[Float32],
    weights: List[Float32],
    max_weight: Float32,
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
    """Dense adapter (the `umap/optimizer.mojo` surface and the stage-identity fixtures): the same row-major positive-edge compaction from an n x n weight matrix."""
    var row_offsets = List[UInt32]()
    var tails = List[UInt32]()
    var scaled = List[Float32]()
    row_offsets.append(UInt32(0))
    for head in range(n_samples):
        for tail in range(n_samples):
            var weight = weights[head * n_samples + tail]
            if head == tail or not (weight > Float32(0.0)):
                continue
            if weight != weights[tail * n_samples + head]:
                raise Error("UMAP device optimizer requires symmetric weights")
            tails.append(UInt32(tail))
            scaled.append(_scaled_weight(weight, max_weight))
        row_offsets.append(UInt32(len(tails)))
    return optimize_csr_layout_identical_device(
        ctx, initial_embedding, row_offsets^, tails^, scaled^, n_samples,
        n_components, n_epochs, initial_learning_rate, negative_sample_rate,
        repulsion_strength, a, b, seed,
    )
