# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""E03's bounded, owner-scoped grouping workspace. Source experiment only.

The owning step supplies an owner ID and monotonically changing ID generation.
Any ID mutation/upload, context recreation, padding change or scratch mutation
requires a new generation (or invalidate()). It must not reuse owner IDs while
a workspace is live. Pointer equality alone NEVER authorizes reuse. The
workspace must outlive synchronization, like all existing embedding scratch.
No production LM/SA caller is wired to this API yet.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from embedding.checks.embedding_identical import (
    EMB_TPB, _emb_backward_launch, _emb_backward_refuse_launch,
    emb_refuse_device_ids,
)
from embedding.checks.embedding_oracle import EmbConfig
from embedding.checks.embedding_sort import PLAN_SORT

# E03 A/B — NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF.
# B rebuilds grouping on every call; A reuses only a complete generation key.
# The owning caller must account for initialization/retention in whole-step
# time and invalidate after every write. No Python data processing.
comptime EMB_REUSE_GROUPS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NEURAL_E03_EMB_REUSE_GROUPS"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


struct EmbGroupingWorkspace(Movable):
    var counts: DeviceBuffer[DType.int32]
    var begin: DeviceBuffer[DType.int32]
    var perm: DeviceBuffer[DType.int32]
    var vocab_capacity: Int
    var positions_capacity: Int
    var ready: Bool
    var owner: UInt64
    var generation: UInt64
    var ids_address: Int
    var n_positions: Int
    var vocab: Int
    var padding: Int

    def __init__(out self, ctx: DeviceContext, vocab_capacity: Int, positions_capacity: Int) raises:
        if vocab_capacity < 1 or positions_capacity < 1:
            raise Error("embedding grouping workspace: positive capacities required")
        self.counts = ctx.enqueue_create_buffer[DType.int32](vocab_capacity)
        self.begin = ctx.enqueue_create_buffer[DType.int32](vocab_capacity + 1)
        self.perm = ctx.enqueue_create_buffer[DType.int32](positions_capacity)
        self.vocab_capacity = vocab_capacity
        self.positions_capacity = positions_capacity
        self.ready = False
        self.owner = UInt64(0)
        self.generation = UInt64(0)
        self.ids_address = 0
        self.n_positions = 0
        self.vocab = 0
        self.padding = 0

    def invalidate(mut self):
        self.ready = False

    def backward_into(
        mut self, ctx: DeviceContext, mut dw: DeviceBuffer[DType.float32],
        mut dy: DeviceBuffer[DType.float32], mut ids: DeviceBuffer[DType.int32],
        n_positions: Int, cfg: EmbConfig, owner: UInt64, generation: UInt64,
    ) raises:
        """Public unvalidated IDs still pass the complete existing refusal.
        Reuse concerns integer grouping only, never dY or carried dW state.
        Owner/context and generation are mandatory, nonzero lifecycle stamps.
        """
        if owner == 0 or generation == 0:
            raise Error("embedding grouping workspace: nonzero lifecycle stamps required")
        if n_positions < 0 or n_positions > self.positions_capacity or cfg.vocab > self.vocab_capacity:
            raise Error("embedding grouping workspace: capacity exceeded")
        _emb_backward_refuse_launch(ctx, PLAN_SORT, EMB_TPB)
        emb_refuse_device_ids(ctx, ids, n_positions, cfg)
        var address = Int(ids.unsafe_ptr())
        var reuse = False
        comptime if EMB_REUSE_GROUPS:
            reuse = (self.ready and self.owner == owner and self.generation == generation
                     and self.ids_address == address and self.n_positions == n_positions
                     and self.vocab == cfg.vocab and self.padding == cfg.padding_idx)
        # An exception never leaves a previously valid entry marked reusable.
        self.ready = False
        _emb_backward_launch(ctx, dw, dy, ids, self.counts, self.begin, self.perm,
                             n_positions, cfg, PLAN_SORT, EMB_TPB, reuse)
        self.owner = owner
        self.generation = generation
        self.ids_address = address
        self.n_positions = n_positions
        self.vocab = cfg.vocab
        self.padding = cfg.padding_idx
        self.ready = n_positions > 0 and cfg.vocab > 0 and cfg.width > 0
