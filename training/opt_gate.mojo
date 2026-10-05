# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL optimizer step: the contract-8a refusal as ONE scan and a
DEVICE GATE (lane nr-small D4), and the SGD update as ONE launch over the
flat model with the batched clip inside the step (lane nr-small D12).

MOJOLEARN_IDN_OPT_GATE_SCAN (D4). Before: `opt_refuse_device_inputs`
launched three or four `nonfinite_partial_kernel` scans, allocated a
partials buffer and a pinned mirror, read the partials back, WAITED, folded
them on the host, and only then enqueued the clip and the update (which
waited again). Now: one grid-striding scan launch reads `param`, `grad`,
the moments in the same pass (per buffer, the smallest non-finite index
per block, by bits, the same predicate as `nonfinite_partial_kernel`), one
one-block fold launch writes the per-buffer minimum and a gate cell, and
the update kernels read the gate cell first and do nothing when it is set.
The cells are copied back behind the update and the step's existing wait
covers them; the host then raises the oracle's message for THIS step (the
first buffer in the oracle's order, its smallest index, NaN or infinity by
the one 4 B classification readback on the error path only). The state is
never partially updated: a gated kernel writes nothing. With clipping on,
the clip reads and rescales `grad` and refuses its own scalar, so the gate
is read (one wait, the one the old scan paid) before the clip runs. Integer
minima over indices are exact, so the refusal, the index and the message
are the old ones on every vendor and on the host column; the update
arithmetic is the same inlined body (`_adam_update_body`,
`_sgd_update_element`), so the bits do not move.
-D MOJOLEARN_IDN_OPT_GATE_SCAN_OFF (or MOJOLEARN_IDN_ALL_OFF) restores the
old scans; `MOJOLEARN_OPT_SCAN_FUSED=0` at run time still selects main's
four separate scans; `MOJOLEARN_OPT_TRUST_INPUTS` still means no scan.

MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH (D12). SGD was one launch per tensor
only because the momentum flag is per tensor (contract 7.3b). The flag now
rides in a small device table (`table[0..J]` the offsets, `table[J+1..2J]`
the flags), and each thread finds its tensor by a binary search over the
offsets, so the flag is still a per-TENSOR value (every element of tensor
j reads flag j). The per-element arithmetic is `sgd_update_kernel`'s body,
so the bits do not move. Same design as the FAST `AFN_OPT_MULTITENSOR`.
-D MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH_OFF (or MOJOLEARN_IDN_ALL_OFF).

MOJOLEARN_IDN_OPT_CLIP_BATCHED (D12). The step's clip calls
`identical_clip_grad_norm_batched` (the same GEMMs in the same order, J
per-tensor waits removed) instead of `identical_clip_grad_norm`. Same
bits by construction (its docstring). Any clip sabotage build still takes
the original inside the batched entry.
-D MOJOLEARN_IDN_OPT_CLIP_BATCHED_OFF (or MOJOLEARN_IDN_ALL_OFF).
"""
from std.gpu import block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from std.os import getenv
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_scan import (
    NONFINITE_NONE,
    SCAN_TPB,
    _scan_blocks,
    device_classify_nonfinite,
)
from core.step_phase import (
    step_count_d2h,
    step_count_device_alloc,
    step_count_h2d,
    step_count_host_alloc,
    step_count_launch,
    step_count_sync,
)
from training.checks.optimizer_contract import opt_nonfinite_message


comptime OPT_GATE_SCAN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_OPT_GATE_SCAN_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
    and not is_defined["MOJOLEARN_OPT_TRUST_INPUTS"]()
)
comptime OPT_SGD_ONE_LAUNCH = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH_OFF"]()
    or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime OPT_CLIP_BATCHED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_OPT_CLIP_BATCHED_OFF"]()
    or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: Gate cells: 0..3 the smallest non-finite index in param, grad, m, v
#: (or NONFINITE_NONE); 4 the gate (nonzero refuses the step).
comptime OPT_GATE_CELLS = 8
comptime OPT_GATE_CELL = 4
#: index of the cells buffer in the step's gate buffer list (0 = partials)
comptime OPT_GATE_DEV_CELLS = 1


@always_inline
def _opt_nonfinite_bits(x: Float32) -> Bool:
    """NaN or infinity BY BITS, `nonfinite_partial_kernel`'s predicate."""
    return (bitcast[DType.uint32](x) & UInt32(0x7FFFFFFF)) >= UInt32(
        0x7F800000
    )


def opt_gate_scan_kernel[FOUR: Bool](
    part: MutPointer[Int32, MutAnyOrigin],
    param: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    m_state: MutPointer[Float32, MutAnyOrigin],
    v_state: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """ONE grid-striding pass over the step's inputs: block `b` writes the
    smallest non-finite index it saw in `param`, `grad`, `m_state` and
    (`FOUR`) `v_state` to `part[k * blocks + b]`, or `NONFINITE_NONE`. The
    grid is `_scan_blocks(n)` at `SCAN_TPB`, the old scans' shape; an
    integer minimum is exact in any order, so the fold's answer is the old
    answer at any launch geometry."""
    var n = Int(n_in)
    var red = stack_allocation[
        SCAN_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var blocks = Int(grid_dim.x)
    var stride = blocks * SCAN_TPB
    var i = Int(block_idx.x) * SCAN_TPB + tid
    var best = SIMD[DType.int32, 4](NONFINITE_NONE)
    while i < n:
        if best[0] == NONFINITE_NONE and _opt_nonfinite_bits(
            param.unsafe_load(i)
        ):
            best[0] = Int32(i)
        if best[1] == NONFINITE_NONE and _opt_nonfinite_bits(
            grad.unsafe_load(i)
        ):
            best[1] = Int32(i)
        if best[2] == NONFINITE_NONE and _opt_nonfinite_bits(
            m_state.unsafe_load(i)
        ):
            best[2] = Int32(i)
        comptime if FOUR:
            if best[3] == NONFINITE_NONE and _opt_nonfinite_bits(
                v_state.unsafe_load(i)
            ):
                best[3] = Int32(i)
        i += stride
    comptime for k in range(4):
        comptime if k < 3 or FOUR:
            red.unsafe_store(tid, best[k])
            barrier()
            var active = SCAN_TPB // 2
            while active > 0:
                if tid < active:
                    var o = red.unsafe_load(tid + active)
                    if o < red.unsafe_load(tid):
                        red.unsafe_store(tid, o)
                barrier()
                active = active // 2
            if tid == 0:
                part.unsafe_store(k * blocks + Int(block_idx.x), red.unsafe_load(0))
            barrier()


def opt_gate_fold_kernel[FOUR: Bool](
    cells: MutPointer[Int32, MutAnyOrigin],
    part: MutPointer[Int32, MutAnyOrigin],
    blocks_in: Int32,
):
    """ONE block of `SCAN_TPB`: the minimum over the scan partials per
    buffer into `cells[0..3]` (`cells[3]` is `NONFINITE_NONE` under SGD)
    and the gate `cells[OPT_GATE_CELL]`, nonzero when any buffer holds a
    non-finite value."""
    var blocks = Int(blocks_in)
    var red = stack_allocation[
        SCAN_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var gate = Int32(0)
    comptime for k in range(4):
        var hit = NONFINITE_NONE
        comptime if k < 3 or FOUR:
            var best = NONFINITE_NONE
            var b = tid
            while b < blocks:
                var o = part.unsafe_load(k * blocks + b)
                if o < best:
                    best = o
                b += SCAN_TPB
            red.unsafe_store(tid, best)
            barrier()
            var active = SCAN_TPB // 2
            while active > 0:
                if tid < active:
                    var o2 = red.unsafe_load(tid + active)
                    if o2 < red.unsafe_load(tid):
                        red.unsafe_store(tid, o2)
                barrier()
                active = active // 2
            hit = red.unsafe_load(0)
            barrier()
        if hit != NONFINITE_NONE:
            gate = Int32(1)
        if tid == 0:
            cells.unsafe_store(k, hit)
    if tid == 0:
        cells.unsafe_store(OPT_GATE_CELL, gate)


def opt_gate_wanted(offsets: List[Int]) -> Bool:
    """The gated path runs when the build has it, the run did not ask for
    main's separate scans (`MOJOLEARN_OPT_SCAN_FUSED=0`), and the step has
    at least one tensor and one element (otherwise main's path, which
    handles the degenerate shapes, runs unchanged)."""
    comptime if not OPT_GATE_SCAN:
        return False
    if len(offsets) < 2:
        return False
    if offsets[len(offsets) - 1] <= 0:
        return False
    return String(getenv("MOJOLEARN_OPT_SCAN_FUSED")) != "0"


def opt_gate_begin(
    ctx: DeviceContext,
    mut gate_dev: List[DeviceBuffer[DType.int32]],
    mut gate_host: List[HostBuffer[DType.int32]],
    mut param: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut m_state: DeviceBuffer[DType.float32],
    mut v_state: DeviceBuffer[DType.float32],
    n: Int,
    is_sgd: Bool,
) raises:
    """Enqueue the scan and the fold. Appends the partials (index 0) and
    the cells (index `OPT_GATE_DEV_CELLS`) to `gate_dev` and the pinned
    cells mirror to `gate_host`; the caller keeps both lists alive past
    the wait in `opt_gate_finish`. Nothing waits here."""
    var blocks = _scan_blocks(n)
    step_count_device_alloc()
    gate_dev.append(ctx.enqueue_create_buffer[DType.int32](4 * blocks))
    step_count_device_alloc()
    gate_dev.append(ctx.enqueue_create_buffer[DType.int32](OPT_GATE_CELLS))
    step_count_host_alloc()
    gate_host.append(ctx.enqueue_create_host_buffer[DType.int32](OPT_GATE_CELLS))
    var part = gate_dev[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cells = gate_dev[OPT_GATE_DEV_CELLS].unsafe_ptr().unsafe_origin_cast[
        MutAnyOrigin
    ]()
    step_count_launch()
    if is_sgd:
        comptime k3 = opt_gate_scan_kernel[False]
        ctx.enqueue_function[k3](
            part,
            param.unsafe_ptr(),
            grad.unsafe_ptr(),
            m_state.unsafe_ptr(),
            v_state.unsafe_ptr(),
            Int32(n),
            grid_dim=(blocks, 1, 1),
            block_dim=(SCAN_TPB, 1, 1),
        )
        comptime f3 = opt_gate_fold_kernel[False]
        step_count_launch()
        ctx.enqueue_function[f3](
            cells,
            part,
            Int32(blocks),
            grid_dim=(1, 1, 1),
            block_dim=(SCAN_TPB, 1, 1),
        )
    else:
        comptime k4 = opt_gate_scan_kernel[True]
        ctx.enqueue_function[k4](
            part,
            param.unsafe_ptr(),
            grad.unsafe_ptr(),
            m_state.unsafe_ptr(),
            v_state.unsafe_ptr(),
            Int32(n),
            grid_dim=(blocks, 1, 1),
            block_dim=(SCAN_TPB, 1, 1),
        )
        comptime f4 = opt_gate_fold_kernel[True]
        step_count_launch()
        ctx.enqueue_function[f4](
            cells,
            part,
            Int32(blocks),
            grid_dim=(1, 1, 1),
            block_dim=(SCAN_TPB, 1, 1),
        )


def opt_gate_cells_ptr(
    mut gate_dev: List[DeviceBuffer[DType.int32]],
) -> MutPointer[Int32, MutAnyOrigin]:
    """The device gate cells, for the gated update kernels."""
    return gate_dev[OPT_GATE_DEV_CELLS].unsafe_ptr().unsafe_origin_cast[
        MutAnyOrigin
    ]()


def opt_gate_finish(
    ctx: DeviceContext,
    mut gate_dev: List[DeviceBuffer[DType.int32]],
    mut gate_host: List[HostBuffer[DType.int32]],
    mut param: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut m_state: DeviceBuffer[DType.float32],
    mut v_state: DeviceBuffer[DType.float32],
    is_sgd: Bool,
) raises:
    """Copy the cells back, WAIT (the one wait of the step; everything
    enqueued before it, the update included, is done after it), and raise
    the oracle's refusal for THIS step when the gate is set: the first
    buffer in the oracle's order (`param`, `grad`, then `exp_avg` and
    `exp_avg_sq`, or `momentum_buffer`) with a hit, its smallest index,
    and NaN or infinity by one 4 B readback on the error path only."""
    var host = gate_host[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=host, src_buf=gate_dev[OPT_GATE_DEV_CELLS])
    step_count_sync()
    ctx.synchronize()
    if host.unsafe_load(OPT_GATE_CELL) == Int32(0):
        return
    var count = 3 if is_sgd else 4
    for k in range(count):
        var idx = host.unsafe_load(k)
        if idx == NONFINITE_NONE:
            continue
        var hit = Int(idx)
        var is_nan: Bool
        var name: String
        if k == 0:
            is_nan = device_classify_nonfinite(ctx, param, hit)
            name = String("param")
        elif k == 1:
            is_nan = device_classify_nonfinite(ctx, grad, hit)
            name = String("grad")
        elif k == 2:
            is_nan = device_classify_nonfinite(ctx, m_state, hit)
            name = String("momentum_buffer") if is_sgd else String("exp_avg")
        else:
            is_nan = device_classify_nonfinite(ctx, v_state, hit)
            name = String("exp_avg_sq")
        raise Error(opt_nonfinite_message(name, hit, is_nan))
    raise Error("optimizer step: the device gate refused the step")


def opt_sgd_table_upload(
    ctx: DeviceContext,
    mut keep_dev: List[DeviceBuffer[DType.int32]],
    mut keep_host: List[HostBuffer[DType.int32]],
    offsets: List[Int],
    buf_initialized: List[Bool],
) raises -> MutPointer[Int32, MutAnyOrigin]:
    """The SGD one-launch table: `table[0..J]` the offsets (ascending,
    `table[J] == n`), `table[J+1..2J]` the per-tensor momentum flags,
    staged through a pinned buffer and enqueued for upload. The caller
    keeps both lists alive past its wait."""
    var j_count = len(offsets) - 1
    var table_n = 2 * j_count + 1
    step_count_host_alloc()
    keep_host.append(ctx.enqueue_create_host_buffer[DType.int32](table_n))
    step_count_device_alloc()
    keep_dev.append(ctx.enqueue_create_buffer[DType.int32](table_n))
    var last = len(keep_host) - 1
    var hp = keep_host[last].unsafe_ptr()
    for j in range(j_count + 1):
        hp.unsafe_store(j, Int32(offsets[j]))
    for j in range(j_count):
        hp.unsafe_store(
            j_count + 1 + j, Int32(1) if buf_initialized[j] else Int32(0)
        )
    var lastd = len(keep_dev) - 1
    step_count_h2d()
    ctx.enqueue_copy(dst_buf=keep_dev[lastd], src_ptr=hp)
    return keep_dev[lastd].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
