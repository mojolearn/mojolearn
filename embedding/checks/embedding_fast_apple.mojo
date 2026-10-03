# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST candidate for the public `Embedding` (lane afn-mlp, 2026-10-03):
`-D MOJOLEARN_AFN_EMB_ATOMIC_BWD`, compiled only under FAST + Apple, default
OFF. IDENTICAL and every other vendor compile `embedding_identical.mojo`'s
path unchanged.

THE BACKWARD main runs (`_emb_backward_launch`, PLAN_SCAN): the +0.0 seed,
the per-row counts, the run-begin scan (ONE block), the permutation, the
ascending fold (one thread per `(v, j)` cell walking its run) and the
padding row: six launches, plus `emb_refuse_device_ids` (a readback and two
waits) and, in the binding, three zero-filled int32 uploads that each wait.
Here: the seed (unless the caller carries), ONE scatter-add launch over the
`T * d` cells of `dY` with a relaxed f32 atomic add into `dW[ids[t], j]`
(free order: FAST), and the padding row. No run structure, no sort, no
counts; the ids were refused by name on the host in the binding
(`emb_refuse_ids`), so the device re-check and its two waits go. The
binding's uploads under this define do not wait (the host arrays outlive
the call; one in-order queue).

THE FORWARD keeps the gather; with `d % 4 == 0` it gathers four floats per
thread (`emb_gather4_kernel`), a quarter of the threads for the same
coalesced words.
"""
from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from embedding.checks.embedding_identical import (
    EMB_TPB,
    _grid_for,
    emb_gather_kernel,
    emb_pad_row_kernel,
    emb_seed_kernel,
)
from embedding.checks.embedding_oracle import EmbConfig

comptime EMB_ATOMIC_BWD = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and is_defined["MOJOLEARN_AFN_EMB_ATOMIC_BWD"]()
)


def emb_scatter_add_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    width_in: Int32,
    padding_idx_in: Int32,
):
    """`dW[ids[t], j] += dY[t, j]` for every cell of `dY`, a relaxed f32
    atomic add (free order); the padding row's positions contribute
    nothing (contract 8)."""
    var width = Int(width_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(n_positions_in) * width:
        return
    var t = cell // width
    var j = cell - t * width
    var v = ids.unsafe_load(t)
    if v == padding_idx_in:
        return
    _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
        dw.unsafe_offset(Int(v) * width + j), dy.unsafe_load(cell)
    )


def emb_gather4_kernel(
    out_y: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    width_in: Int32,
):
    """`Y[t, 4q .. 4q + 3] = W[ids[t], 4q .. 4q + 3]`, one float4 per thread
    (`width % 4 == 0`)."""
    var width = Int(width_in)
    var quads = width // 4
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(n_positions_in) * quads:
        return
    var t = cell // quads
    var q = cell - t * quads
    var v = Int(ids.unsafe_load(t))
    var src = weight.unsafe_load[width=4](v * width + q * 4)
    out_y.unsafe_store[width=4](t * width + q * 4, src)


def fast_embedding_forward_into(
    ctx: DeviceContext,
    mut out_y: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
) raises:
    """The gather, enqueued; the ids were refused on the host."""
    comptime if not EMB_ATOMIC_BWD:
        raise Error("fast_embedding_forward_into: compiled without MOJOLEARN_AFN_EMB_ATOMIC_BWD")
    else:
        if cfg.width < 1 or n_positions < 1:
            return
        if cfg.width % 4 == 0:
            var quads = n_positions * (cfg.width // 4)
            ctx.enqueue_function[emb_gather4_kernel](
                out_y.unsafe_ptr(), weight.unsafe_ptr(), ids.unsafe_ptr(),
                Int32(n_positions), Int32(cfg.width),
                grid_dim=(_grid_for(quads), 1, 1), block_dim=(EMB_TPB, 1, 1),
            )
            return
        var cells = n_positions * cfg.width
        ctx.enqueue_function[emb_gather_kernel](
            out_y.unsafe_ptr(), weight.unsafe_ptr(), ids.unsafe_ptr(),
            Int32(n_positions), Int32(cfg.width), Int32(cfg.vocab),
            grid_dim=(_grid_for(cells), 1, 1), block_dim=(EMB_TPB, 1, 1),
        )


def fast_embedding_backward_into(
    ctx: DeviceContext,
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
) raises:
    """Seed (unless carried), scatter-add, padding row: at most three
    launches, enqueued; the ids were refused on the host."""
    comptime if not EMB_ATOMIC_BWD:
        raise Error("fast_embedding_backward_into: compiled without MOJOLEARN_AFN_EMB_ATOMIC_BWD")
    else:
        if cfg.vocab < 1 or cfg.width < 1:
            return
        var cells = cfg.vocab * cfg.width
        if not cfg.accumulate:
            ctx.enqueue_function[emb_seed_kernel](
                dw.unsafe_ptr(), Int32(cells),
                grid_dim=(_grid_for(cells), 1, 1), block_dim=(EMB_TPB, 1, 1),
            )
        if n_positions >= 1:
            var work = n_positions * cfg.width
            ctx.enqueue_function[emb_scatter_add_kernel](
                dw.unsafe_ptr(), dy.unsafe_ptr(), ids.unsafe_ptr(),
                Int32(n_positions), Int32(cfg.width), Int32(cfg.padding_idx),
                grid_dim=(_grid_for(work), 1, 1), block_dim=(EMB_TPB, 1, 1),
            )
        if cfg.has_padding():
            ctx.enqueue_function[emb_pad_row_kernel](
                dw.unsafe_ptr(), Int32(cfg.width), Int32(cfg.padding_idx),
                grid_dim=(_grid_for(cfg.width), 1, 1), block_dim=(EMB_TPB, 1, 1),
            )
