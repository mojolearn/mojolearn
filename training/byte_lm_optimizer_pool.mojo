# SPDX-License-Identifier: Apache-2.0
"""Disjoint AdamW moment and rollback ownership; no new update arithmetic.

The group snapshots every shard before updating any, then broadcasts disjoint
parameter slices. Full model weights/activations and gradients remain replicated.
"""
from max.gpu.host import DeviceContext
from training.byte_lm import (
    ByteTrainer, byte_glue_update_launch, _maybe_fault,
    _FAULT_NAN, _FAULT_INF, _FAULT_MINUS_ONE, _require_device_finite,
)
from training.checks.train_loop import _copy_into
from training.checks.optimizer import OPT_RECORD_INTERMEDIATES


def pool_snapshot(ctx: DeviceContext, mut tr: ByteTrainer) raises:
    comptime if OPT_RECORD_INTERMEDIATES:
        raise Error("byte LM optimizer pool: recorded intermediates are unsupported")
    tr.validate_device_state(ctx, tr.completed_steps)
    var n = tr.buffers.optimizer_count
    _copy_into(ctx, tr.buffers.shadow_p, tr.buffers.param, 0, tr.buffers.optimizer_first, n)
    _copy_into(ctx, tr.buffers.shadow_m, tr.buffers.m_state, 0, 0, n)
    _copy_into(ctx, tr.buffers.shadow_v, tr.buffers.v_state, 0, 0, n)
    ctx.synchronize()
    tr.buffers.flags_before = tr.buffers.buf_initialized.copy()
    tr.shadow_step = tr.completed_steps
    tr.shadow_valid = True


def pool_update(ctx: DeviceContext, mut tr: ByteTrainer) raises:
    var n = tr.buffers.optimizer_count
    var first = tr.buffers.optimizer_first
    var p = tr.buffers.param.create_sub_buffer[DType.float32](first, n)
    var g = tr.buffers.grad.create_sub_buffer[DType.float32](first, n)
    _maybe_fault(ctx, tr.buffers.m_state, "opt_refuse", min(5,n-1), _FAULT_NAN)
    _require_device_finite(ctx, tr.scan, tr.buffers.m_state, n, "first moments")
    byte_glue_update_launch(ctx, p, g, tr.buffers.m_state, tr.buffers.v_state,
        tr.buffers.shadow_p, tr.buffers.shadow_m, tr.buffers.shadow_v,
        tr.buffers.denom_out, tr.buffers.q_out, n, tr.optimizer,
        tr.completed_steps + 1, False)
    _maybe_fault(ctx, tr.buffers.v_state, "after_nonfinite", min(3,n-1), _FAULT_INF)
    _maybe_fault(ctx, tr.buffers.v_state, "after_negative", min(3,n-1), _FAULT_MINUS_ONE)
    tr.validate_device_state(ctx, tr.completed_steps + 1)


def pool_restore(ctx: DeviceContext, mut tr: ByteTrainer) raises:
    if tr.shadow_valid:
        var n = tr.buffers.optimizer_count
        _copy_into(ctx, tr.buffers.param, tr.buffers.shadow_p, tr.buffers.optimizer_first, 0, n)
        _copy_into(ctx, tr.buffers.m_state, tr.buffers.shadow_m, 0, 0, n)
        _copy_into(ctx, tr.buffers.v_state, tr.buffers.shadow_v, 0, 0, n)
        ctx.synchronize()
        tr.buffers.buf_initialized = tr.buffers.flags_before.copy()
        tr.completed_steps = tr.shadow_step
        tr.shadow_valid = False
    tr.grad_step = -1
    # Full parameter validation belongs AFTER the group's restored slices have
    # replaced other ranks' potentially failed updates.
    _require_device_finite(ctx, tr.scan, tr.buffers.m_state, tr.buffers.optimizer_count, "first moments")
    _require_device_finite(ctx, tr.scan, tr.buffers.v_state, tr.buffers.optimizer_count, "second moments")
