# SPDX-License-Identifier: Apache-2.0
"""NN20 device launchers for the shared stable-summary attention profile.

The public model uses summary_model on GPU and summary_model_host on CPU,
with attention_summary_contract supplying every arithmetic statement.
Fixed absolute 32-key leaves and adjacent-pair/odd-carry merges define a new
version. Forward, decode, backward, trace and session profile labels migrate
together. Existing attention_v2 large-shape nonpromotion remains intact.
Standalone flag-off calls offer a left-fold component control; the actual
model flag-off A/B control is incumbent attention. No compile, verification,
quality assessment or timing has been run and no default was promoted.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div,
    identical_exp, identical_fmax, identical_mul, identical_mul_add,
)

from transformer.experiments.attention_summary_contract import (
    NN20_BALANCED_SUMMARY_TREE, NN20_KEY_LEAF, summary_levels, summary_scratch_elements,
    summary_attention_forward_row, summary_attention_rowdot, summary_attention_dq_cell,
    summary_attention_dkdv_cell, summary_attention_host_forward, summary_attention_host_backward,
    _score, _dyv,
)

def summary_attention_forward_kernel(
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], lo: MutPointer[Int32, MutAnyOrigin],
    hi: MutPointer[Int32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    rows: Int32, keys: Int32, head_dim: Int32, width: Int32,
    queries_per_group: Int32, scale: Float32,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < Int(rows):
        summary_attention_forward_row[NN20_BALANCED_SUMMARY_TREE](q, k, v, lo, hi,
            output, maxes, denoms, scratch, status, row, Int(keys), Int(head_dim),
            Int(width), Int(queries_per_group), scale)


def summary_attention_backward_kernel[STAGE: Int](
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    dq: MutPointer[Float32, MutAnyOrigin], dk: MutPointer[Float32, MutAnyOrigin],
    dv: MutPointer[Float32, MutAnyOrigin], rows: Int32, keys: Int32,
    head_dim: Int32, width: Int32, queries_per_group: Int32, scale: Float32,
):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var r = Int(rows)
    var kcount = Int(keys)
    var hd = Int(head_dim)
    var w = Int(width)
    var qpg = Int(queries_per_group)
    comptime if STAGE == 0:
        if cell < r:
            summary_attention_rowdot(q, k, v, dy, lo, hi, maxes, denoms,
                zdot, status, cell, kcount, hd, w, qpg, scale)
    elif STAGE == 1:
        if cell < r * hd:
            summary_attention_dq_cell(q, k, v, dy, lo, hi, maxes, denoms,
                zdot, status, dq, cell, kcount, hd, w, qpg, scale)
    else:
        var groups = (r + qpg - 1) // qpg
        if cell < groups * kcount * max(hd, w):
            summary_attention_dkdv_cell(q, k, v, dy, lo, hi, maxes, denoms,
                zdot, status, dk, dv, cell, r, kcount, hd, w, qpg, scale)


def _require_shape(rows: Int, keys: Int, head_dim: Int, width: Int,
                   queries_per_group: Int) raises:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("NN20 is an IDENTICAL-only component")
    if rows <= 0 or keys <= 0 or head_dim <= 0 or width <= 0 or queries_per_group <= 0:
        raise Error("NN20: positive dimensions/grouping required")
    # Device scalar ABI is Int32. This is an ABI bound, not a performance
    # cap; neighboring legal shapes always use the same numerical graph.
    if max(max(rows, keys), max(max(head_dim, width), queries_per_group)) > 2147483647:
        raise Error("NN20: dimension exceeds kernel Int32 ABI")


def enqueue_summary_attention_forward(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.float32], mut k: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32], mut lo: DeviceBuffer[DType.int32],
    mut hi: DeviceBuffer[DType.int32], mut output: DeviceBuffer[DType.float32],
    mut maxes: DeviceBuffer[DType.float32], mut denoms: DeviceBuffer[DType.float32],
    mut scratch: DeviceBuffer[DType.float32], mut status: DeviceBuffer[DType.int32],
    rows: Int, keys: Int, head_dim: Int, width: Int, queries_per_group: Int,
    scale: Float32,
) raises:
    _require_shape(rows, keys, head_dim, width, queries_per_group)
    var groups = (rows + queries_per_group - 1) // queries_per_group
    if len(q) < rows * head_dim or len(k) < groups * keys * head_dim or len(v) < groups * keys * width:
        raise Error("NN20: short Q/K/V operand")
    if len(output) < rows * width or len(maxes) < rows or len(denoms) < rows or len(status) < rows or len(lo) < rows or len(hi) < rows:
        raise Error("NN20: short output/row state")
    if len(scratch) < summary_scratch_elements(rows, keys, width):
        raise Error("NN20: bounded summary-stack scratch is too small")
    ctx.enqueue_function[summary_attention_forward_kernel](
        q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), lo.unsafe_ptr(), hi.unsafe_ptr(),
        output.unsafe_ptr(), maxes.unsafe_ptr(), denoms.unsafe_ptr(), scratch.unsafe_ptr(), status.unsafe_ptr(),
        Int32(rows), Int32(keys), Int32(head_dim), Int32(width), Int32(queries_per_group), scale,
        grid_dim=((rows + 63) // 64, 1, 1), block_dim=(64, 1, 1),
    )


def enqueue_summary_attention_backward(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.float32], mut k: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32], mut dy: DeviceBuffer[DType.float32],
    mut lo: DeviceBuffer[DType.int32], mut hi: DeviceBuffer[DType.int32],
    mut maxes: DeviceBuffer[DType.float32], mut denoms: DeviceBuffer[DType.float32],
    mut zdot: DeviceBuffer[DType.float32], mut status: DeviceBuffer[DType.int32],
    mut dq: DeviceBuffer[DType.float32], mut dk: DeviceBuffer[DType.float32],
    mut dv: DeviceBuffer[DType.float32], rows: Int, keys: Int, head_dim: Int,
    width: Int, queries_per_group: Int, scale: Float32,
) raises:
    """Saved max/denom/mask/status must belong to this exact forward generation.

    Caller owns all buffers through completion and consumes/refuses nonzero
    status before accepting outputs. No source-data comparison is performed.
    Three in-order kernels establish z before its independent consumers.
    """
    _require_shape(rows, keys, head_dim, width, queries_per_group)
    var groups = (rows + queries_per_group - 1) // queries_per_group
    if len(q) < rows * head_dim or len(k) < groups * keys * head_dim or len(v) < groups * keys * width or len(dy) < rows * width:
        raise Error("NN20: short backward operand")
    if len(dq) < rows * head_dim or len(dk) < groups * keys * head_dim or len(dv) < groups * keys * width:
        raise Error("NN20: short gradient output")
    if len(maxes) < rows or len(denoms) < rows or len(zdot) < rows or len(status) < rows or len(lo) < rows or len(hi) < rows:
        raise Error("NN20: short backward row state")
    comptime for stage in range(3):
        var tasks = rows
        comptime if stage == 1:
            tasks = rows * head_dim
        elif stage == 2:
            tasks = groups * keys * max(head_dim, width)
        ctx.enqueue_function[summary_attention_backward_kernel[stage]](
            q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), dy.unsafe_ptr(), lo.unsafe_ptr(), hi.unsafe_ptr(),
            maxes.unsafe_ptr(), denoms.unsafe_ptr(), zdot.unsafe_ptr(), status.unsafe_ptr(),
            dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(), Int32(rows), Int32(keys),
            Int32(head_dim), Int32(width), Int32(queries_per_group), scale,
            grid_dim=((tasks + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
