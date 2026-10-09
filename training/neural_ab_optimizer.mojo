# SPDX-License-Identifier: Apache-2.0
"""NN55/NN56 experimental grouped OOP Adam/AdamW and block status.

Source only: no compilation/identity/quality/timing evidence. Uses the existing
canonical optimizer cell, not a new mathematical transcription. This explicit
component does not commit parameters or advance counters: its owner must run
the existing admission and finish/rollback policy before publishing outputs.
"""
from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined, get_defined_int
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.checks.optimizer import _adam_oop_cell
from training.checks.optimizer_contract import OptimizerConfig, StepScalars, step_scalars

# L11 (2026-10-07): NN55 needs NN56, so they are arms of ONE switch,
# -D MOJOLEARN_IDN_LM_GROUPED_ADAM=0|1|2: 1 = NN56 grouped OOP Adam,
# 2 = NN56 plus NN55 block-status minima. Arm 1 alone isolates NN56.
comptime IDN_LM_GROUPED_ADAM_ARM = get_defined_int["MOJOLEARN_IDN_LM_GROUPED_ADAM", 0]()
comptime NN55_BLOCK_STATUS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_LM_GROUPED_ADAM_ARM == 2
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN56_GROUPED_ADAM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_LM_GROUPED_ADAM_ARM >= 1
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime _FP = MutPointer[Float32, MutAnyOrigin]
comptime _IP = MutPointer[Int32, MutAnyOrigin]
comptime NN_ADAM_TPB = 128
comptime NN_ADAM_SCALARS = 9


def _nn_adam_scalar_kernel(dst: _FP, params: _FP,
    steps: MutPointer[Int64, MutAnyOrigin], groups: Int32):
    var group = Int(block_idx.x) * 128 + Int(thread_idx.x)
    if group >= Int(groups):
        return
    var at = group * 5
    # Only the Adam/AdamW fields used by the shared canonical scalar contract.
    # Other optimizer fields do not enter any of the nine published values.
    var cfg = OptimizerConfig(0, params[at], params[at+1], params[at+2],
        params[at+3], params[at+4], Float32(0), Float32(0), False, Float32(0))
    var value = step_scalars(cfg, Int(steps[group]))
    var out_at = group * NN_ADAM_SCALARS
    dst[out_at] = cfg.beta1
    dst[out_at+1] = cfg.beta2
    dst[out_at+2] = cfg.eps
    dst[out_at+3] = cfg.weight_decay
    dst[out_at+4] = value.c1
    dst[out_at+5] = value.c2
    dst[out_at+6] = value.step_size
    dst[out_at+7] = value.rt_bc2
    dst[out_at+8] = value.decay_mul


def nn_adam_scalar_table(ctx: DeviceContext, configs: List[OptimizerConfig],
    steps: List[Int]) raises -> DeviceBuffer[DType.float32]:
    """Upload optimizer configuration metadata; compute step arithmetic on GPU.

    One lane owns each independent optimizer group. The same step_scalars
    contract supplies the nine values used by Adam/AdamW. Synchronization keeps
    the temporary upload buffers alive and belongs inside the training step.
    """
    if len(configs) != len(steps) or len(configs) < 1 or len(configs) > 2147483647 // NN_ADAM_SCALARS:
        raise Error("NN56 group/step metadata length mismatch")
    var metadata = List[Float32]()
    var counts = List[Int64]()
    for group in range(len(configs)):  # small-loop(configs: optimizer groups): packs per-group config scalars for upload
        if steps[group] < 1:
            raise Error("NN56 optimizer step must be positive")
        ref cfg = configs[group]
        metadata.append(cfg.lr)
        metadata.append(cfg.beta1)
        metadata.append(cfg.beta2)
        metadata.append(cfg.eps)
        metadata.append(cfg.weight_decay)
        counts.append(Int64(steps[group]))
    var params = ctx.enqueue_create_buffer[DType.float32](len(metadata))
    var dsteps = ctx.enqueue_create_buffer[DType.int64](len(counts))
    var table = ctx.enqueue_create_buffer[DType.float32](NN_ADAM_SCALARS * len(configs))
    try:
        ctx.enqueue_copy(dst_buf=params, src_ptr=metadata.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dsteps, src_ptr=counts.unsafe_ptr())
        ctx.enqueue_function[_nn_adam_scalar_kernel](table.unsafe_ptr(), params.unsafe_ptr(),
            dsteps.unsafe_ptr(), Int32(len(configs)), grid_dim=(len(configs)+127)//128, block_dim=128)
        ctx.synchronize()
    except e:
        ctx.synchronize()
        raise e
    _ = metadata^
    _ = counts^
    _ = params^
    _ = dsteps^
    return table^


def nn_grouped_adam_kernel[STATUS: Bool](
    p_out: _FP, m_out: _FP, v_out: _FP, param: _FP, grad: _FP,
    m_state: _FP, v_state: _FP, offsets: _IP, kinds: _IP, scalars: _FP,
    status: _IP, tiles_in: Int32, n_in: Int32,
):
    """Independent groups on grid.y, original per-cell OOP arithmetic.

    offsets has groups+1 entries and is monotonic, covering n elements exactly.
    kinds contains the incumbent is_adamw flag. Owners admit these descriptors
    before launch. All 128 threads participate in status barriers, including
    inactive tails. Integer minima are independent of hardware warp width.
    """
    var group = Int(block_idx.y)
    var block = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var i = Int(offsets[group]) + block * NN_ADAM_TPB + tid
    var end = Int(offsets[group + 1])
    var first = InlineArray[Int32, 4](fill=n_in)
    if i < end:
        var s = group * NN_ADAM_SCALARS
        var result = _adam_oop_cell(param[i], grad[i], m_state[i], v_state[i], kinds[group], scalars[s], scalars[s + 1], scalars[s + 2], scalars[s + 3], scalars[s + 4], scalars[s + 5], scalars[s + 6], scalars[s + 7], scalars[s + 8])
        p_out[i] = result[0]
        m_out[i] = result[1]
        v_out[i] = result[2]
        comptime if STATUS:
            comptime for field in range(3):
                if (bitcast[DType.uint32](result[field]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
                    first[field] = Int32(i)
            if result[2] < Float32(0.0):
                first[3] = Int32(i)
    comptime if STATUS:
        var red = stack_allocation[4 * NN_ADAM_TPB, Int32, address_space=AddressSpace.SHARED]()
        comptime for field in range(4):
            red[field * NN_ADAM_TPB + tid] = first[field]
        barrier()
        var step = NN_ADAM_TPB // 2
        while step > 0:
            if tid < step:
                comptime for field in range(4):
                    var at = field * NN_ADAM_TPB + tid
                    red[at] = min(red[at], red[at + step])
            barrier()
            step //= 2
        if tid == 0:
            var out = (group * Int(tiles_in) + block) * 4
            comptime for field in range(4):
                status[out + field] = red[field * NN_ADAM_TPB]


def nn_adam_status_fold_kernel(dst: _IP, parts: _IP, blocks_in: Int32):
    """Every block folds NN_ADAM_TPB status partials per field (the shared
    min tree of `nn_grouped_adam_kernel`), then thread 0 folds the block's
    four minima into dst with an atomic min. dst starts at n (filled before
    the launch). Integer minima are exact and independent of block order,
    so the result equals the old one-block walk on every vendor."""
    var tid = Int(thread_idx.x)
    var part = Int(block_idx.x) * NN_ADAM_TPB + tid
    var red = stack_allocation[4 * NN_ADAM_TPB, Int32, address_space=AddressSpace.SHARED]()
    comptime for field in range(4):
        red[field * NN_ADAM_TPB + tid] = parts[4 * part + field] if part < Int(blocks_in) else Int32(2147483647)
    barrier()
    var step = NN_ADAM_TPB // 2
    while step > 0:
        if tid < step:
            comptime for field in range(4):
                var at = field * NN_ADAM_TPB + tid
                red[at] = min(red[at], red[at + step])
        barrier()
        step //= 2
    if tid == 0:
        comptime for field in range(4):
            _ = Atomic.min(dst.unsafe_offset(field), red[field * NN_ADAM_TPB])


def nn_grouped_adam_into[STATUS: Bool](
    ctx: DeviceContext, mut p_out: DeviceBuffer[DType.float32],
    mut m_out: DeviceBuffer[DType.float32], mut v_out: DeviceBuffer[DType.float32],
    mut param: DeviceBuffer[DType.float32], mut grad: DeviceBuffer[DType.float32],
    mut m_state: DeviceBuffer[DType.float32], mut v_state: DeviceBuffer[DType.float32],
    mut offsets: DeviceBuffer[DType.int32], mut kinds: DeviceBuffer[DType.int32],
    mut scalars: DeviceBuffer[DType.float32], mut parts: DeviceBuffer[DType.int32],
    mut status: DeviceBuffer[DType.int32], n: Int, groups: Int, max_group_cells: Int,
) raises:
    """A=grouped; STATUS adds NN55 block minima. B=existing per-group update.

    Owner retains all buffers and must provide admitted metadata. Parameter
    buffers are out of place; the mandatory input checks remain caller-owned.
    parts has 4*groups*ceil(max_group_cells/128) slots, status has four. A final
    host decision after completion preserves the existing failure policy.
    """
    comptime if not NN56_GROUPED_ADAM:
        raise Error("NN56 grouped Adam is not enabled")
    comptime if STATUS:
        comptime if not NN55_BLOCK_STATUS:
            raise Error("NN55 block status is not enabled")
    if n < 0 or n > 2147483647 or groups < 1 or max_group_cells < 1 or max_group_cells > n:
        raise Error("NN56 invalid admitted shape metadata")
    var tiles = (max_group_cells + NN_ADAM_TPB - 1) // NN_ADAM_TPB
    if groups > 2147483647 // tiles // 4:
        raise Error("NN56 status workspace exceeds native bounds")
    ctx.enqueue_function[nn_grouped_adam_kernel[STATUS]](p_out.unsafe_ptr(), m_out.unsafe_ptr(), v_out.unsafe_ptr(), param.unsafe_ptr(), grad.unsafe_ptr(), m_state.unsafe_ptr(), v_state.unsafe_ptr(), offsets.unsafe_ptr(), kinds.unsafe_ptr(), scalars.unsafe_ptr(), parts.unsafe_ptr(), Int32(tiles), Int32(n), grid_dim=(tiles, groups, 1), block_dim=(NN_ADAM_TPB, 1, 1))
    comptime if STATUS:
        var fold_blocks = (groups * tiles + NN_ADAM_TPB - 1) // NN_ADAM_TPB
        status.enqueue_fill(Int32(n))
        ctx.enqueue_function[nn_adam_status_fold_kernel](status.unsafe_ptr(), parts.unsafe_ptr(), Int32(groups * tiles), grid_dim=(fold_blocks, 1, 1), block_dim=(NN_ADAM_TPB, 1, 1))
