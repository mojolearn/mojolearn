# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Row-sharded SVM kernel matrices over a `DevicePlan`
(docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md).

`kernel_matrix_dispatch(plan, ...)` is the one SVM kernel-matrix entry above
the single-device `kernel_op` (svm/impl/distance/kernel_matrices.mojo). With
`plan.single()` it calls `kernel_op` unchanged; otherwise it partitions the
output rows over the plan's devices (`_kernel_rows`). The single-device file
keeps no device-count branch and reads no thread plan: the plan value is read
once at the family's native entry (svm/estimator.mojo, the kernel-method
entries) and passed down.

The order of the old in-`kernel_op` branch is kept: empty shapes return, the
C22 symmetric triangle answers ahead of any sharding, then the device count
is parsed from the plan's raw setting and refused outside IDENTICAL or
1..64, exactly as before. No single-device bit can move: that path is the
same `kernel_op` call with the same arguments.
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_plan import DevicePlan
from core.host_parallel import host_parallelize
from core.multi_gpu import peer_clone
from core.step_phase import STEP_PHASE_TIMERS
from svm.impl.distance.kernel_matrices import (
    kernel_op,
    kernel_op_takes_triangle,
    kernel_workspace_floats,
)
from svm.impl.svm_parameter import KERNEL_RBF, KernelParams


def kernel_matrix_dispatch(
    plan: DevicePlan,
    ctx: DeviceContext,
    kp: KernelParams,
    mut output: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    mut norm_a: DeviceBuffer[DType.float32],
    mut norm_b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
) raises:
    """`kernel_op` on one device, or its output rows over `plan`'s devices.

    Same buffers and postcondition as `kernel_op`: `output[m x n] = K(a_i,
    b_j)`, row-major. ASYNCHRONOUS on one device; the sharded form
    synchronizes before it returns."""
    # The old branch sat after kernel_op's empty-shape return and its C22
    # triangle, so both still answer first whatever the plan says.
    if plan.single() or m <= 0 or n <= 0 or kernel_op_takes_triangle(kp, a, b, m, n):
        kernel_op(ctx, kp, output, a, b, m, n, k, norm_a, norm_b, ws)
        return
    var count = Int(plan.setting)
    if count < 1 or count > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel SVM kernels require IDENTICAL and 1..64 devices")
    # Named under IDENTICAL only: under FAST the kernel_op -> _kernel_rows ->
    # kernel_op cycle reached MAX matmul and hung NVPTX/AMDGPU (0.8.19).
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        _kernel_rows(plan, ctx, kp, output, a, b, m, n, k, norm_a, norm_b, count)


@fieldwise_init
struct SVMKernelShard(Movable):
    var ctx: DeviceContext
    var a: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var norm_a: DeviceBuffer[DType.float32]
    var norm_b: DeviceBuffer[DType.float32]
    var out: DeviceBuffer[DType.float32]
    var ws: DeviceBuffer[DType.float32]
    var first: Int
    var rows: Int

    def __deinit__(deinit self):
        _ = self.a^
        _ = self.b^
        _ = self.norm_a^
        _ = self.norm_b^
        _ = self.out^
        _ = self.ws^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def _kernel_rows(plan: DevicePlan, ctx: DeviceContext, kp: KernelParams,
    mut output: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int,
    mut norm_a: DeviceBuffer[DType.float32], mut norm_b: DeviceBuffer[DType.float32],
    count: Int,
) raises:
    """Partition output rows, retaining each original FP32-v1 contraction.

    The feature axis is never split or padded. Original norm bytes and the
    original RBF epilogue are reused. The SMO working set, updates and stopping
    rules remain on the root. Per-call staging is not a speed/capacity claim.

    THE NEGATIVE CONTROL. `-D MOJOLEARN_SVM_PARALLEL_SABOTAGE=1` makes every
    owner above rank 0 read its left operand rows, and its RBF row norms, one
    row early. `rows` is unchanged, the per-shard allocations keep their sizes
    and the write-back keeps the true `shard.first`, so nothing but the values
    contracted moves. It is a `comptime if`, so no production bit can move, and
    it is INERT AT ONE DEVICE: the shift is guarded by `rank > 0` and a
    one-device column has only rank 0. That is what makes a moved `par-svm`,
    `par-svm-svr`, `par-kernel-ridge` or `par-nystroem` cell attributable to the
    define rather than to the second device. (`MOJOLEARN_SVM_DEVICE_COUNT` is
    also what `kernel_methods/checks/kernel_matrix.mojo` reads, which is why
    the kernel-method lanes sit behind this one define.) Owed a two-device
    column (`MOJOLEARN_PAR_DEVICES=0,1`); no host binding restates this driver.
    """
    ctx.synchronize()
    var active = min(count, m)
    comptime if STEP_PHASE_TIMERS:
        if active > 1:
            raise Error("parallel SVM cannot use process-global GEMM phase counters")
    var shards = List[SVMKernelShard]()
    for rank in range(active):
        var first = m * rank // active
        var rows = m * (rank + 1) // active - first
        var source = first
        comptime if is_defined["MOJOLEARN_SVM_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their left rows one row early.
            if rank > 0:
                source = first - 1
        var device = DeviceContext(device_id=plan.device(rank))
        var av = a.create_sub_buffer[DType.float32](source * k, rows * k)
        var bv = b.create_sub_buffer[DType.float32](0, n * k)
        var na = norm_a.create_sub_buffer[DType.float32](source if kp.kernel == KERNEL_RBF else 0,
                                                        rows if kp.kernel == KERNEL_RBF else 1)
        var nb = norm_b.create_sub_buffer[DType.float32](0, n if kp.kernel == KERNEL_RBF else 1)
        var local_a = peer_clone(ctx, device, av)
        var local_b = peer_clone(ctx, device, bv)
        var local_na = peer_clone(ctx, device, na)
        var local_nb = peer_clone(ctx, device, nb)
        var output = device.enqueue_create_buffer[DType.float32](rows * n)
        var workspace = device.enqueue_create_buffer[DType.float32](kernel_workspace_floats(rows, n, k))
        device.synchronize()
        shards.append(SVMKernelShard(device^, local_a^, local_b^, local_na^, local_nb^,
                                     output^, workspace^, first, rows))
    var failures = List[Int](length=active, fill=0)
    var sp = rebind[MutPointer[SVMKernelShard, MutUntrackedOrigin]](shards.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failures.unsafe_ptr())
    def task(rank: Int) {imm sp, imm fp, imm kp, imm n, imm k}:
        try:
            ref shard = sp[rank]
            kernel_op(shard.ctx, kp, shard.out, shard.a, shard.b, shard.rows, n, k,
                      shard.norm_a, shard.norm_b, shard.ws)
            shard.ctx.synchronize()
        except:
            fp[rank] = 1
    if active == 1:
        task(0)
    else:
        host_parallelize(task, active)
    for rank in range(active):
        if failures[rank] != 0:
            raise Error("SVM kernel row shard failed: " + String(rank))
    for rank in range(active):
        ref shard = shards[rank]
        var destination = output.create_sub_buffer[DType.float32](shard.first * n, shard.rows * n)
        shard.out.enqueue_copy_to(destination)
        shard.ctx.synchronize()
    _ = shards^
    ctx.synchronize()
