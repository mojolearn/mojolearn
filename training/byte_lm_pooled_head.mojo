# SPDX-License-Identifier: Apache-2.0
"""Embedding/loss/head buffers for a layer-owned model; no full model replica."""
from max.gpu.host import DeviceBuffer, DeviceContext
from training.byte_lm_config import ByteConfig
from training.checks.train_loop import _zeros, _zeros_i32, _ones
from gemm.checks.gemm_identical import identical_gemm_workspace_max_floats
from gemm.checks.gemm_backward import identical_gemm_backward_workspace_max_floats
from gemm.checks.gemm_oracle import OP_NT
from embedding.checks.embedding_identical import emb_run_scratch_ints
from training.checks.loss import identical_ce_ones_floats, identical_ce_workspace_max_floats
from training.checks.loss_oracle import REDUCTION_MEAN


struct BytePooledHead(Movable):
    var emb_w: DeviceBuffer[DType.float32]  # [V, d_model]
    var lm_w: DeviceBuffer[DType.float32]  # [V, d_model]
    var dw_emb: DeviceBuffer[DType.float32]  # [V, d_model]
    var dw_lm: DeviceBuffer[DType.float32]  # [V, d_model]

    var ids: DeviceBuffer[DType.int32]  # [M]
    var targets: DeviceBuffer[DType.int32]  # [M]

    var x: DeviceBuffer[DType.float32]  # [M, d_model]  block input
    var logits: DeviceBuffer[DType.float32]  # [M, V]
    var d_h: DeviceBuffer[DType.float32]  # [M, d_model]

    var ce_max: DeviceBuffer[DType.float32]
    var ce_shift: DeviceBuffer[DType.float32]
    var ce_expo: DeviceBuffer[DType.float32]
    var ce_denom: DeviceBuffer[DType.float32]
    var ce_logdenom: DeviceBuffer[DType.float32]
    var ce_logp_target: DeviceBuffer[DType.float32]
    var ce_nll: DeviceBuffer[DType.float32]
    var ce_logp: DeviceBuffer[DType.float32]
    var ce_logp_sum: DeviceBuffer[DType.float32]
    var ce_smooth: DeviceBuffer[DType.float32]
    var ce_row: DeviceBuffer[DType.float32]
    var ce_total: DeviceBuffer[DType.float32]
    var ce_loss: DeviceBuffer[DType.float32]
    var ce_weights: DeviceBuffer[DType.float32]
    var ce_dlogits: DeviceBuffer[DType.float32]
    var ce_ones: DeviceBuffer[DType.float32]
    var ce_ws: DeviceBuffer[DType.float32]

    var head_ws: DeviceBuffer[DType.float32]
    var head_bwd_ws: DeviceBuffer[DType.float32]

    var emb_counts: DeviceBuffer[DType.int32]
    var emb_run_begin: DeviceBuffer[DType.int32]
    var emb_perm: DeviceBuffer[DType.int32]

    var final_hidden: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, config: ByteConfig) raises:
        config.validate()
        var M = config.batch*config.length
        var DM = config.d_model
        var V = config.vocab_size
        self.emb_w = _zeros(ctx, V * DM)
        self.lm_w = _zeros(ctx, V * DM)
        self.dw_emb = _zeros(ctx, V * DM)
        self.dw_lm = _zeros(ctx, V * DM)

        self.ids = _zeros_i32(ctx, M)
        self.targets = _zeros_i32(ctx, M)

        self.x = _zeros(ctx, M * DM)
        self.logits = _zeros(ctx, M * V)
        self.d_h = _zeros(ctx, M * DM)

        self.ce_max = _zeros(ctx, M)
        # Match ByteTrainBuffers' qualified CE lifetime schedule.  The
        # shift kernel loads logits[cell] before its same-cell store, and no
        # pooled/offload trace retains logits after CE begins.
        self.ce_shift = self.logits.create_sub_buffer[DType.float32](0, M * V)
        self.ce_expo = _zeros(ctx, M * V)
        self.ce_denom = _zeros(ctx, M)
        self.ce_logdenom = _zeros(ctx, M)
        self.ce_logp_target = _zeros(ctx, M)
        self.ce_nll = _zeros(ctx, M)
        self.ce_logp = _zeros(ctx, 1)
        self.ce_logp_sum = _zeros(ctx, 1)
        self.ce_smooth = _zeros(ctx, 1)
        self.ce_row = _zeros(ctx, M)
        self.ce_total = _zeros(ctx, 1)
        self.ce_loss = _zeros(ctx, 1)
        # CE backward consumes expo cell-locally into weights and then
        # dlogits on the same in-order context.  Neither intermediate is
        # retained by IdentityTrace in the pooled production paths.
        self.ce_weights = self.ce_expo.create_sub_buffer[DType.float32](0, M * V)
        self.ce_dlogits = self.ce_expo.create_sub_buffer[DType.float32](0, M * V)
        self.ce_ones = _ones(ctx, identical_ce_ones_floats(M, V))
        self.ce_ws = _zeros(
            ctx, identical_ce_workspace_max_floats(M, V, REDUCTION_MEAN)
        )

        self.head_ws = _zeros(
            ctx, identical_gemm_workspace_max_floats(M, V, DM)
        )
        self.head_bwd_ws = _zeros(
            ctx,
            identical_gemm_backward_workspace_max_floats(
                OP_NT, M, V, DM, False
            ),
        )

        var scratch = emb_run_scratch_ints(V, M)
        if scratch != V + (V + 1) + M:
            raise Error(
                String("byte LM: emb_run_scratch_ints says ")
                + String(scratch)
                + " ints and counts+run_begin+perm is "
                + String(V + (V + 1) + M)
                + ". The embedding lane changed its run structure and this"
                + " harness would hand it three buffers of the wrong size."
            )
        self.emb_counts = _zeros_i32(ctx, V)
        self.emb_run_begin = _zeros_i32(ctx, V + 1)
        self.emb_perm = _zeros_i32(ctx, M)

        self.final_hidden = _zeros(ctx,M*DM)
