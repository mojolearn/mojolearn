# SPDX-License-Identifier: Apache-2.0
"""Whole residual-dropout forward/backward, baseline and NN59 fused schedule.

Every upload, scratch allocation, refusal scan, output download and wait is
inside this operation. No model behavior is redirected. Source only, unverified.
"""
from std.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceBuffer,DeviceContext
from core.device_scan import device_first_nonfinite
from checks.numerics import ftz
from training.residual_dropout_contract import (
    NN59_DROPOUT_RESIDUAL,nn_dropout_cell,residual_dropout_admit,
)
from training.neural_ab_pointwise import nn_dropout_residual_into

comptime _FP = MutPointer[Float32,MutAnyOrigin]
comptime _HP = MutPointer[Float32,MutUntrackedOrigin]


def _dropout_only(dst: _FP,x: _FP,n: Int32,offset: Int64,p: Float32,scale: Float32,
                  seed_lo: UInt32,seed_hi: UInt32,stream: UInt32):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n):
        dst[i] = nn_dropout_cell(x[i],p,scale,seed_lo,seed_hi,stream,Int(offset)+i)


def _residual_add(dst: _FP,dropped: _FP,residual: _FP,n: Int32):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n):
        dst[i] = ftz(ftz(residual[i])+ftz(dropped[i]))


def _gradient_copy(dst: _FP,source: _FP,n: Int32):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n):
        # Residual gradient is the raw upstream word, matching the fused arm.
        dst[i] = source[i]


def residual_dropout_into[BACKWARD: Bool](ctx: DeviceContext,
    mut output: DeviceBuffer[DType.float32],mut dside: DeviceBuffer[DType.float32],
    mut input: DeviceBuffer[DType.float32],mut side: DeviceBuffer[DType.float32],
    n: Int,offset: Int,p: Float32,scale: Float32,seed_lo: Int,seed_hi: Int,stream: Int) raises:
    """The refusal scans and the launches on device buffers, shared by the
    host-pointer entry below and the resident-handle entry (lane
    ne3-c-blocks-embedding, row N117). Forward: output = residual +
    dropout(input), `side` the residual, `dside` unused. Backward: output =
    dx, dside = dresidual, `input` the upstream gradient, `side` unused.
    n >= 1; the caller's queue order makes every read follow its write."""
    if device_first_nonfinite(ctx,input,n)>=0:
        raise Error("residual_dropout refuses nonfinite input")
    comptime if not BACKWARD:
        if device_first_nonfinite(ctx,side,n)>=0:
            raise Error("residual_dropout refuses nonfinite residual")
    comptime if NN59_DROPOUT_RESIDUAL:
        nn_dropout_residual_into[BACKWARD](ctx,output,dside,input,side,n,offset,p,scale,
            UInt32(seed_lo),UInt32(seed_hi),UInt32(stream))
    else:
        comptime if BACKWARD:
            ctx.enqueue_function[_dropout_only](output,input,Int32(n),Int64(offset),p,scale,
                UInt32(seed_lo),UInt32(seed_hi),UInt32(stream),grid_dim=((n+127)//128,1,1),block_dim=(128,1,1))
            ctx.enqueue_function[_gradient_copy](dside,input,Int32(n),
                grid_dim=((n+127)//128,1,1),block_dim=(128,1,1))
        else:
            var temporary = ctx.enqueue_create_buffer[DType.float32](n)
            ctx.enqueue_function[_dropout_only](temporary,input,Int32(n),Int64(offset),p,scale,
                UInt32(seed_lo),UInt32(seed_hi),UInt32(stream),grid_dim=((n+127)//128,1,1),block_dim=(128,1,1))
            ctx.enqueue_function[_residual_add](output,temporary,side,Int32(n),
                grid_dim=((n+127)//128,1,1),block_dim=(128,1,1))
            # The scan below synchronizes before `temporary` goes.
            if device_first_nonfinite(ctx,output,n)>=0:
                raise Error("residual_dropout produced a nonfinite output")
            _ = temporary^
            return
    if device_first_nonfinite(ctx,output,n)>=0:
        raise Error("residual_dropout produced a nonfinite output")


def residual_dropout_device[BACKWARD: Bool](ctx: DeviceContext,
    first: _HP,second: _HP,x: _HP,residual: _HP,n: Int,offset: Int,p: Float32,
    seed_lo: Int,seed_hi: Int,stream: Int) raises -> Int:
    var scale = residual_dropout_admit(n,offset,p,seed_lo,seed_hi,stream)
    if n==0:
        return 0
    var input = ctx.enqueue_create_buffer[DType.float32](n)
    var side = ctx.enqueue_create_buffer[DType.float32](1 if BACKWARD else n)
    var output = ctx.enqueue_create_buffer[DType.float32](n)
    var dside = ctx.enqueue_create_buffer[DType.float32](n if BACKWARD else 1)
    try:
        ctx.enqueue_copy(dst_buf=input,src_ptr=x)
        comptime if not BACKWARD:
            ctx.enqueue_copy(dst_buf=side,src_ptr=residual)
        residual_dropout_into[BACKWARD](ctx,output,dside,input,side,n,offset,p,scale,seed_lo,seed_hi,stream)
        ctx.enqueue_copy(dst_ptr=first,src_buf=output)
        comptime if BACKWARD:
            ctx.enqueue_copy(dst_ptr=second,src_buf=dside)
        ctx.synchronize()
    except error:
        ctx.synchronize()
        raise error
    _ = input^
    _ = side^
    _ = output^
    _ = dside^
    return n
