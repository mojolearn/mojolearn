# SPDX-License-Identifier: Apache-2.0
"""NI20 model launch adapters; arithmetic lives in the shared native module."""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from transformer.impl.llama.attention_v2_model_contract import (
    V2Ptr, v2m_forward_row, v2m_forward_diagnostic, v2m_prepare_row,
    v2m_dq_row, v2m_dkdv_key, v2m_backward_diagnostic,
)


def _forward[diagnostic: Bool](q: V2Ptr, k: V2Ptr, v: V2Ptr, output: V2Ptr,
             maxima: V2Ptr, denominators: V2Ptr, scores: V2Ptr, masked: V2Ptr,
             exps: V2Ptr, weights: V2Ptr, b: Int32, length: Int32, heads: Int32,
             kv_heads: Int32, keys: Int32, depth: Int32, own0: Int32,
             window: Int32, scale: Float32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(b) * Int(length) * Int(heads):
        return
    v2m_forward_row(q, k, v, output, maxima, denominators, row,
                   Int(length), Int(heads), Int(kv_heads), Int(keys),
                   Int(depth), Int(own0), Int(window), scale)
    comptime if diagnostic:
        v2m_forward_diagnostic(q, k, maxima, denominators, scores, masked,
                              exps, weights, row, Int(length), Int(heads),
                              Int(kv_heads), Int(keys), Int(depth),
                              Int(own0), Int(window), scale)


def _prepare(q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr, maxima: V2Ptr,
             denominators: V2Ptr, zdots: V2Ptr, b: Int32, length: Int32,
             heads: Int32, kv_heads: Int32, keys: Int32, depth: Int32,
             own0: Int32, window: Int32, scale: Float32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(b) * Int(length) * Int(heads):
        return
    v2m_prepare_row(q, k, v, dy, maxima, denominators, zdots, row,
                   Int(length), Int(heads), Int(kv_heads), Int(keys),
                   Int(depth), Int(own0), Int(window), scale)


def _dq[diagnostic: Bool](q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr,
        maxima: V2Ptr, denominators: V2Ptr, zdots: V2Ptr, dq: V2Ptr,
        dw: V2Ptr, dmasked: V2Ptr, dscores: V2Ptr, dqk: V2Ptr,
        b: Int32, length: Int32, heads: Int32, kv_heads: Int32, keys: Int32,
        depth: Int32, own0: Int32, window: Int32, scale: Float32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(b) * Int(length) * Int(heads):
        return
    v2m_dq_row(q, k, v, dy, maxima, denominators, zdots, dq, row,
              Int(length), Int(heads), Int(kv_heads), Int(keys), Int(depth),
              Int(own0), Int(window), scale)
    comptime if diagnostic:
        v2m_backward_diagnostic(q, k, v, dy, maxima, denominators, zdots,
                               dw, dmasked, dscores, dqk, row, Int(length),
                               Int(heads), Int(kv_heads), Int(keys),
                               Int(depth), Int(own0), Int(window), scale)


def _dkdv(q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr, maxima: V2Ptr,
          denominators: V2Ptr, zdots: V2Ptr, dk: V2Ptr, dv: V2Ptr,
          b: Int32, length: Int32, heads: Int32, kv_heads: Int32, keys: Int32,
          depth: Int32, own0: Int32, window: Int32, scale: Float32):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(b) * Int(kv_heads) * Int(keys):
        return
    v2m_dkdv_key(q, k, v, dy, maxima, denominators, zdots, dk, dv, idx,
                Int(length), Int(heads), Int(kv_heads), Int(keys), Int(depth),
                Int(own0), Int(window), scale)


def attention_v2_model_forward[diagnostic: Bool](
    ctx: DeviceContext, mut q: DeviceBuffer[DType.float32],
    mut k: DeviceBuffer[DType.float32], mut v: DeviceBuffer[DType.float32],
    mut output: DeviceBuffer[DType.float32], mut maxima: DeviceBuffer[DType.float32],
    mut denominators: DeviceBuffer[DType.float32], mut scores: DeviceBuffer[DType.float32],
    mut masked: DeviceBuffer[DType.float32], mut exps: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32], b: Int, length: Int, heads: Int,
    kv_heads: Int, keys: Int, depth: Int, own0: Int, window: Int, scale: Float32,
) raises:
    ctx.enqueue_function[_forward[diagnostic]](
        q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), output.unsafe_ptr(),
        maxima.unsafe_ptr(), denominators.unsafe_ptr(), scores.unsafe_ptr(),
        masked.unsafe_ptr(), exps.unsafe_ptr(), weights.unsafe_ptr(),
        Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(depth), Int32(own0), Int32(window), scale,
        grid_dim=((b * heads * length + 63) // 64, 1, 1), block_dim=(64, 1, 1),
    )


def attention_v2_model_backward[diagnostic: Bool](
    ctx: DeviceContext, mut q: DeviceBuffer[DType.float32],
    mut k: DeviceBuffer[DType.float32], mut v: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32], mut maxima: DeviceBuffer[DType.float32],
    mut denominators: DeviceBuffer[DType.float32], mut zdots: DeviceBuffer[DType.float32],
    mut dq: DeviceBuffer[DType.float32], mut dk: DeviceBuffer[DType.float32],
    mut dv: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.float32],
    mut dmasked: DeviceBuffer[DType.float32], mut dscores: DeviceBuffer[DType.float32],
    mut dqk: DeviceBuffer[DType.float32], b: Int, length: Int, heads: Int,
    kv_heads: Int, keys: Int, depth: Int, own0: Int, window: Int, scale: Float32,
) raises:
    var row_grid = (b * heads * length + 63) // 64
    ctx.enqueue_function[_prepare](
        q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), dy.unsafe_ptr(),
        maxima.unsafe_ptr(), denominators.unsafe_ptr(), zdots.unsafe_ptr(),
        Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(depth), Int32(own0), Int32(window), scale,
        grid_dim=(row_grid, 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_function[_dq[diagnostic]](
        q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), dy.unsafe_ptr(),
        maxima.unsafe_ptr(), denominators.unsafe_ptr(), zdots.unsafe_ptr(),
        dq.unsafe_ptr(), dw.unsafe_ptr(), dmasked.unsafe_ptr(), dscores.unsafe_ptr(), dqk.unsafe_ptr(),
        Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(depth), Int32(own0), Int32(window), scale,
        grid_dim=(row_grid, 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_function[_dkdv](
        q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), dy.unsafe_ptr(),
        maxima.unsafe_ptr(), denominators.unsafe_ptr(), zdots.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(),
        Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(depth), Int32(own0), Int32(window), scale,
        grid_dim=((b * kv_heads * keys + 63) // 64, 1, 1), block_dim=(64, 1, 1),
    )
