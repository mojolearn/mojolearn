# SPDX-License-Identifier: Apache-2.0
"""Whole-tensor gradient ownership with the original global clipping tree.

Each tensor norm keeps the original contraction length and kernel. Only the
small canonical sumsq vector returns to the first device for the original
cross-tensor norm. Scaling uses the resulting original coefficient on every
owner, with full host staging before publication. One tensor must fit an owner.
"""
from std.os import getenv
from std.sys import is_defined
from max.gpu.host import DeviceContext, DeviceBuffer
from bindings.hostptr import copy_f32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.step_phase import STEP_PHASE_TIMERS
from training.estimator import identical_clip_grad_norm_host, _offsets_from_ptr
from training.samba_ops import _upload_f32
from training.checks.train_loop import _zeros, download_f32
from training.checks.optimizer import (
    identical_clip_coefficient, identical_optimizer_workspace_floats,
    clip_scale_kernel, OPT_TPB, _grid_for,
    SAB_CLIP_PARAM_ORDER, SAB_CLIP_SERIAL_FOLD, SAB_CLIP_BLOCK_PARTITION,
    SAB_CLIP_FLAT_NORM,
)
from training.checks.optimizer_oracle import refuse_nonfinite_scalar
from gemm.checks.gemm_identical import identical_gemm_into
from gemm.checks.gemm_oracle import OP_NT


comptime CLIP_POOL_FAULT = is_defined["MOJOLEARN_CLIP_POOL_FAULT"]()


def clip_pool_fault_available() -> Int:
    return 1 if CLIP_POOL_FAULT else 0


struct OwnedClipTensor(Movable):
    var owner: Int
    var first: Int
    var gradient: DeviceBuffer[DType.float32]
    var sumsq: Float32

    def __init__(out self, ctx: DeviceContext, owner: Int, first: Int,
                 count: Int, ptr: MutPointer[Float32, MutUntrackedOrigin]) raises:
        self.owner = owner
        self.first = first
        self.gradient = _upload_f32(ctx,ptr+first,count)
        var cell = _zeros(ctx,1)
        var ws = _zeros(ctx,identical_optimizer_workspace_floats([0,count]))
        var ga = self.gradient.create_sub_buffer[DType.float32](0,count)
        var gb = self.gradient.create_sub_buffer[DType.float32](0,count)
        identical_gemm_into(ctx,cell,ga,gb,ws,1,1,count,OP_NT)
        ctx.synchronize()
        var value = download_f32(ctx,cell,1)
        self.sumsq = value[0]
        _ = ga
        _ = gb
        _ = cell^
        _ = ws^


def parallel_clip_grad_norm_host(
    ctx: DeviceContext,
    grad_ptr: MutPointer[Float32, MutUntrackedOrigin],
    offsets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    info_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_tensors: Int, max_norm: Float32,
) raises -> Int:
    var devices = Int(getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT","1"))
    if devices == 1:
        return identical_clip_grad_norm_host(ctx,grad_ptr,offsets_ptr,info_ptr,n_tensors,max_norm)
    if devices < 1 or devices > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel clip requires IDENTICAL and 1..64 devices")
    comptime if STEP_PHASE_TIMERS:
        raise Error("parallel clip cannot use process-global phase counters")
    comptime if SAB_CLIP_PARAM_ORDER or SAB_CLIP_SERIAL_FOLD or SAB_CLIP_BLOCK_PARTITION or SAB_CLIP_FLAT_NORM:
        raise Error("parallel clip requires the original clean reduction profile")
    if max_norm <= Float32(0):
        raise Error("parallel clip: max_norm must be > 0")
    refuse_nonfinite_scalar("max_norm",max_norm)
    var offsets = _offsets_from_ptr(offsets_ptr,n_tensors)
    devices = min(devices,n_tensors)
    var contexts = List[DeviceContext]()
    for rank in range(devices):
        contexts.append(DeviceContext(device_id=rank))
    var tensors = List[OwnedClipTensor]()
    var sums = List[Float32]()
    var counts = List[Int](length=devices,fill=0)
    for j in range(n_tensors):
        # Greedy whole-tensor placement is integer bookkeeping; ties choose
        # the lower device index. The canonical norm order remains j.
        var owner = 0
        for rank in range(1,devices):
            if counts[rank] < counts[owner]:
                owner = rank
        var n = offsets[j+1]-offsets[j]
        tensors.append(OwnedClipTensor(contexts[owner],owner,offsets[j],n,grad_ptr))
        sums.append(tensors[j].sumsq)
        counts[owner] += n
    var sumsq = _upload_f32(ctx,rebind[MutPointer[Float32, MutUntrackedOrigin]](sums.unsafe_ptr()),n_tensors)
    var norms = _zeros(ctx,n_tensors)
    var total_cell = _zeros(ctx,1)
    var out2 = _zeros(ctx,2)
    var ws = _zeros(ctx,identical_optimizer_workspace_floats([0,n_tensors]))
    var coef = identical_clip_coefficient(ctx,sumsq,norms,total_cell,out2,ws,n_tensors,max_norm)
    var info = download_f32(ctx,out2,2)
    var result = List[Float32](length=offsets[n_tensors],fill=Float32(0))
    for j in range(n_tensors):
        ref tensor = tensors[j]
        ref owner_ctx = contexts[tensor.owner]
        var n = len(tensor.gradient)
        owner_ctx.enqueue_function[clip_scale_kernel](tensor.gradient.unsafe_ptr(),Int32(n),coef,
            grid_dim=(_grid_for(n),1,1),block_dim=(OPT_TPB,1,1))
        owner_ctx.synchronize()
        comptime if CLIP_POOL_FAULT:
            if Int(getenv("MOJOLEARN_CLIP_FAIL_OWNER","-1")) == tensor.owner:
                raise Error("parallel clip: injected post-scale owner failure; output unchanged")
        owner_ctx.enqueue_copy(dst_ptr=result.unsafe_ptr()+tensor.first,src_buf=tensor.gradient)
        owner_ctx.synchronize()
    _ = tensors^
    _ = contexts^
    _ = sumsq^
    _ = norms^
    _ = total_cell^
    _ = out2^
    _ = ws^
    copy_f32(result.unsafe_ptr(),grad_ptr,len(result))
    copy_f32(info.unsafe_ptr(),info_ptr,2)
    return offsets[n_tensors]
