# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Sparse conflict-free Jacobi GPU optimizer for UMAP's NUMERIC_FAST lane."""

from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite


comptime FAST_OPT_TPB = 128


def _mix(value: UInt64) -> UInt64:
    var z = value + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _clip(value: Float32) -> Float32:
    if value > Float32(4.0):
        return Float32(4.0)
    if value < Float32(-4.0):
        return Float32(-4.0)
    return value


def umap_jacobi_epoch_kernel(
    source: MutPointer[Float32, MutAnyOrigin],
    row_offsets: MutPointer[UInt32, MutAnyOrigin],
    tails: MutPointer[UInt32, MutAnyOrigin],
    edge_weights: MutPointer[Float32, MutAnyOrigin],
    destination: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    components_in: Int32,
    epoch_in: Int32,
    epochs_in: Int32,
    learning_rate: Float32,
    negative_rate_in: Int32,
    repulsion: Float32,
    a: Float32,
    b: Float32,
    max_weight: Float32,
    seed: UInt64,
):
    var head = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if head >= n:
        return
    var components = Int(components_in)
    var epoch = Int(epoch_in)
    var alpha = learning_rate * Float32(epochs_in - epoch_in) / Float32(
        epochs_in
    )
    for c in range(components):
        var update_sum = Float32(0.0)
        var head_value = source.unsafe_load(head * components + c)
        var edge_begin = Int(row_offsets.unsafe_load(head))
        var edge_end = Int(row_offsets.unsafe_load(head + 1))
        for edge_ordinal in range(edge_begin, edge_end):
            var tail = Int(tails.unsafe_load(edge_ordinal))
            var weight = edge_weights.unsafe_load(edge_ordinal)
            var scaled = weight / max_weight
            if Int(Float32(epoch + 1) * scaled) <= Int(Float32(epoch) * scaled):
                continue
            var dist = Float32(0.0)
            for axis in range(components):
                var delta = source.unsafe_load(head * components + axis) - (
                    source.unsafe_load(tail * components + axis)
                )
                dist += delta * delta
            if dist > Float32(0.0):
                var dist_pow = dist ** b
                var coeff = -Float32(2.0) * a * b * (dist_pow / dist) / (
                    a * dist_pow + Float32(1.0)
                )
                update_sum += _clip(coeff * (
                    head_value - source.unsafe_load(tail * components + c)
                ))
            for negative in range(Int(negative_rate_in)):
                var counter = (
                    seed
                    ^ (UInt64(epoch) * UInt64(0xD1B54A32D192ED03))
                    ^ (UInt64(edge_ordinal) * UInt64(0x94D049BB133111EB))
                    ^ UInt64(negative)
                )
                var other = Int(_mix(counter) % UInt64(n))
                if other == head:
                    continue
                var neg_dist = Float32(0.0)
                for axis in range(components):
                    var delta = source.unsafe_load(head * components + axis) - (
                        source.unsafe_load(other * components + axis)
                    )
                    neg_dist += delta * delta
                if neg_dist > Float32(0.0):
                    var neg_pow = neg_dist ** b
                    var coeff = Float32(2.0) * repulsion * b / (
                        (Float32(0.001) + neg_dist)
                        * (a * neg_pow + Float32(1.0))
                    )
                    update_sum += _clip(coeff * (
                        head_value - source.unsafe_load(other * components + c)
                    ))
        destination.unsafe_store(
            head * components + c, head_value + alpha * update_sum
        )


def optimize_layout_fast(
    ctx: DeviceContext,
    initial: List[Float32],
    weights: List[Float32],
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
    """One owner per CSR row; epoch snapshots remove all write conflicts."""
    if n_samples < 2 or (n_components != 2 and n_components != 3):
        raise Error("UMAP FAST optimizer supports 2D/3D layouts")
    if len(initial) != n_samples * n_components or len(weights) != (
        n_samples * n_samples
    ):
        raise Error("UMAP FAST optimizer input shape mismatch")
    if not isfinite(learning_rate) or not isfinite(repulsion) or (
        not isfinite(a) or not isfinite(b)
    ):
        raise Error("UMAP FAST optimizer scalar parameters must be finite")
    if n_epochs < 1 or not (learning_rate > Float32(0.0)) or (
        negative_rate < 0 or repulsion < Float32(0.0)
    ):
        raise Error("UMAP FAST optimizer parameters are invalid")
    if not (a > Float32(0.0)) or not (b > Float32(0.0)):
        raise Error("UMAP FAST optimizer curve parameters are invalid")
    # Stable CSR: scanning the dense graph in (head, tail) order makes the
    # CSR position exactly the serial reference's positive-edge ordinal.
    var row_offsets = List[UInt32]()
    var tails = List[UInt32]()
    var edge_weights = List[Float32]()
    row_offsets.append(UInt32(0))
    var max_weight = Float32(0.0)
    for head in range(n_samples):
        for tail in range(n_samples):
            var weight = weights[head * n_samples + tail]
            if not isfinite(weight) or weight < Float32(0.0):
                raise Error("UMAP FAST optimizer graph weight is invalid")
            if weight > max_weight:
                max_weight = weight
            if weight != weights[tail * n_samples + head]:
                raise Error("UMAP FAST optimizer requires symmetric weights")
            if head != tail and weight > Float32(0.0):
                tails.append(UInt32(tail))
                edge_weights.append(weight)
        row_offsets.append(UInt32(len(tails)))
    if not (max_weight > Float32(0.0)):
        raise Error("UMAP FAST optimizer graph has no positive edges")
    if len(edge_weights) == 0:
        raise Error("UMAP FAST optimizer graph has no non-self edges")
    for i in range(len(initial)):
        if not isfinite(initial[i]):
            raise Error("UMAP FAST optimizer initialization is invalid")
    var h_initial = ctx.enqueue_create_host_buffer[DType.float32](len(initial))
    var h_offsets = ctx.enqueue_create_host_buffer[DType.uint32](
        len(row_offsets)
    )
    var h_tails = ctx.enqueue_create_host_buffer[DType.uint32](len(tails))
    var h_weights = ctx.enqueue_create_host_buffer[DType.float32](
        len(edge_weights)
    )
    for i in range(len(initial)):
        h_initial.unsafe_ptr().unsafe_store(i, initial[i])
    for i in range(len(row_offsets)):
        h_offsets.unsafe_ptr().unsafe_store(i, row_offsets[i])
    for i in range(len(tails)):
        h_tails.unsafe_ptr().unsafe_store(i, tails[i])
        h_weights.unsafe_ptr().unsafe_store(i, edge_weights[i])
    var first = ctx.enqueue_create_buffer[DType.float32](len(initial))
    var second = ctx.enqueue_create_buffer[DType.float32](len(initial))
    var d_offsets = ctx.enqueue_create_buffer[DType.uint32](len(row_offsets))
    var d_tails = ctx.enqueue_create_buffer[DType.uint32](len(tails))
    var d_weights = ctx.enqueue_create_buffer[DType.float32](len(edge_weights))
    ctx.enqueue_copy(dst_buf=first, src_ptr=h_initial.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_offsets, src_ptr=h_offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tails, src_ptr=h_tails.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_weights, src_ptr=h_weights.unsafe_ptr())
    for epoch in range(n_epochs):
        if epoch % 2 == 0:
            ctx.enqueue_function[umap_jacobi_epoch_kernel](
                first.unsafe_ptr(), d_offsets.unsafe_ptr(),
                d_tails.unsafe_ptr(), d_weights.unsafe_ptr(),
                second.unsafe_ptr(),
                Int32(n_samples), Int32(n_components), Int32(epoch),
                Int32(n_epochs), learning_rate, Int32(negative_rate),
                repulsion, a, b, max_weight, seed,
                grid_dim=((n_samples + FAST_OPT_TPB - 1) // FAST_OPT_TPB, 1, 1),
                block_dim=(FAST_OPT_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[umap_jacobi_epoch_kernel](
                second.unsafe_ptr(), d_offsets.unsafe_ptr(),
                d_tails.unsafe_ptr(), d_weights.unsafe_ptr(),
                first.unsafe_ptr(),
                Int32(n_samples), Int32(n_components), Int32(epoch),
                Int32(n_epochs), learning_rate, Int32(negative_rate),
                repulsion, a, b, max_weight, seed,
                grid_dim=((n_samples + FAST_OPT_TPB - 1) // FAST_OPT_TPB, 1, 1),
                block_dim=(FAST_OPT_TPB, 1, 1),
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
    _ = h_weights^
    _ = first^
    _ = second^
    _ = d_offsets^
    _ = d_tails^
    _ = d_weights^
    _ = host_out^
    return out^
