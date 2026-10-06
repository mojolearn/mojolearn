# SPDX-License-Identifier: Apache-2.0
"""NN50 immutable token snapshot and retained canonical embedding groups.

Explicit default-OFF component API; no production caller is silently cached.
The owner stores a device copy of IDs, so mutation of the original cannot stale
the groups. Public fields are internal implementation storage: callers must
not mutate/expose their pointers, and must use the creating DeviceContext.
Construction includes copy, device validation and sorting; count that cold cost.
No compilation, verification or performance evidence has been produced.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from embedding.checks.embedding_oracle import EmbConfig, emb_refuse_shape
from embedding.checks.embedding_sort import embedding_sort_runs, PLAN_SORT
from embedding.checks.embedding_identical import (
    ANY_EMB_SABOTAGE, EMB_TPB, _emb_backward_launch, _emb_backward_refuse_launch,
)

comptime NN50_OWNED_RUNS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN50_OWNED_RUNS"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and not ANY_EMB_SABOTAGE
)
comptime _IP = MutPointer[Int32, MutAnyOrigin]


def _nn_ids_init(status: _IP, n: Int32):
    status[0] = n
    status[1] = Int32(0)


def _nn_ids_copy_validate(dst: _IP, src: _IP, status: _IP, n: Int32, vocab: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var value = src[i]
        dst[i] = value
        if value < 0 or value >= vocab:
            _ = Atomic[DType.int32].min(status, Int32(i))


def _nn_ids_error_value(ids: _IP, status: _IP, n: Int32):
    if status[0] < n:
        status[1] = ids[Int(status[0])]


def _nn_ids_compare(left: _IP, right: _IP, status: _IP, n: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n) and left[i] != right[i]:
        _ = Atomic[DType.int32].min(status, Int32(i))


struct OwnedEmbeddingRuns(Movable):
    var ids: DeviceBuffer[DType.int32]
    var counts: DeviceBuffer[DType.int32]
    var begin: DeviceBuffer[DType.int32]
    var perm: DeviceBuffer[DType.int32]
    var positions: Int
    var vocab: Int
    var padding: Int

    def __init__(out self, ctx: DeviceContext, mut source: DeviceBuffer[DType.int32], positions: Int, cfg: EmbConfig) raises:
        comptime if not NN50_OWNED_RUNS:
            raise Error("NN50 owned embedding runs are not enabled")
        emb_refuse_shape(cfg, positions)
        if cfg.vocab > 2147483647 or positions > 2147483647:
            raise Error("NN50 vocabulary/positions exceed native index range")
        self.positions = positions
        self.vocab = cfg.vocab
        self.padding = cfg.padding_idx
        self.ids = ctx.enqueue_create_buffer[DType.int32](max(1, positions))
        self.counts = ctx.enqueue_create_buffer[DType.int32](cfg.vocab)
        self.begin = ctx.enqueue_create_buffer[DType.int32](cfg.vocab + 1)
        self.perm = ctx.enqueue_create_buffer[DType.int32](max(1, positions))
        var status = ctx.enqueue_create_buffer[DType.int32](2)
        var host = ctx.enqueue_create_host_buffer[DType.int32](2)
        ctx.enqueue_function[_nn_ids_init](status.unsafe_ptr(), Int32(positions), grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
        if positions > 0:
            ctx.enqueue_function[_nn_ids_copy_validate](self.ids.unsafe_ptr(), source.unsafe_ptr(), status.unsafe_ptr(), Int32(positions), Int32(cfg.vocab), grid_dim=((positions + 127) // 128, 1, 1), block_dim=(128, 1, 1))
        ctx.enqueue_function[_nn_ids_error_value](self.ids.unsafe_ptr(), status.unsafe_ptr(), Int32(positions), grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=status)
        ctx.synchronize()
        var first = Int(host.unsafe_ptr()[0])
        if first < positions:
            # Same message/first-position convention as emb_refuse_ids; all
            # row processing above was GPU-side, with only two scalar words out.
            raise Error("embedding: id " + String(Int(host.unsafe_ptr()[1])) + " at position " + String(first) + " is outside [0, " + String(cfg.vocab) + ") REFUSED (contract 8; never clamped)")
        embedding_sort_runs(ctx, self.ids, self.counts, self.begin, self.perm, positions, cfg.vocab, cfg.padding_idx, EMB_TPB)

    def matches(mut self, ctx: DeviceContext, mut source: DeviceBuffer[DType.int32], positions: Int, cfg: EmbConfig) raises -> Bool:
        """Compare actual device IDs, never external address identity."""
        if positions != self.positions or cfg.vocab != self.vocab or cfg.padding_idx != self.padding:
            return False
        var status = ctx.enqueue_create_buffer[DType.int32](1)
        status.enqueue_fill(Int32(positions))
        var host = ctx.enqueue_create_host_buffer[DType.int32](1)
        if positions > 0:
            ctx.enqueue_function[_nn_ids_compare](self.ids.unsafe_ptr(), source.unsafe_ptr(), status.unsafe_ptr(), Int32(positions), grid_dim=((positions + 127) // 128, 1, 1), block_dim=(128, 1, 1))
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=status)
        ctx.synchronize()
        var same = host[0] == Int32(positions)
        _ = source
        _ = self.ids
        _ = status^
        return same

    def backward_into(mut self, ctx: DeviceContext, mut dw: DeviceBuffer[DType.float32], mut dy: DeviceBuffer[DType.float32], cfg: EmbConfig) raises:
        """Use original fold/seed/padding kernels, skipping only group rebuild.

        Same dy/weight finite checks and output-buffer admission as the native
        prerefused API are required from the owner. This method enqueues; self,
        dy and dw must remain alive until the caller completes the operation.
        """
        emb_refuse_shape(cfg, self.positions)
        if cfg.vocab != self.vocab or cfg.padding_idx != self.padding:
            raise Error("NN50 grouping does not match vocabulary/padding")
        _emb_backward_refuse_launch(ctx, PLAN_SORT, EMB_TPB)
        _emb_backward_launch(ctx, dw, dy, self.ids, self.counts, self.begin, self.perm, self.positions, cfg, PLAN_SORT, EMB_TPB, runs_ready=True)
