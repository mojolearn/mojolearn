# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""RAFT `cpp/include/raft/stats/detail/contingencyMatrix.cuh` (ebf9268).

The INTEGER core of every label metric in Group A: `outMat[(gt - min) *
width + (pd - min)] += 1` for every sample, by `raft::myAtomicAdd` on an
`int`. COPY, DO NOT IMPROVE. One kernel per arm of theirs:

    devConstructContingencyMatrix       (:32-46)   GLOBAL_ATOMICS
    devConstructContingencyMatrixSmem   (:67-91)   SMEM_ATOMICS
    contingencyMatrixWSort              (:111-142) SORT_AND_GATOMICS  NOT PORTED

INTEGER ATOMICS ARE IDENTITY-SAFE; FLOAT ATOMICS ARE NOT. Integer addition is
associative and commutative, so a counter that `nSamples` threads increment in
any interleaving holds exactly the same bits at the end -- on Metal, PTX and
AMDGPU, run after run, in BOTH numeric modes. Nothing here has an IDENTICAL
arm because nothing here needs one. That is the whole reason the label
metrics are the easy identity cases and the float metrics (Groups B, C) are
not: their `atomicAdd(double)` lands partials in an ARRIVAL order, and float
addition is neither.

THE DISPATCH (`getImplVersion`, :145-167) reads the device's shared-memory
budget and L2 size to choose the arm. Ours cannot: a GPU-agnostic source has
no device to query at compile time and the smem slab must be a comptime
size. So the SMEM arm is taken when `outDimN <= CMAT_SMEM_MAX_DIM` (a slab
of `CMAT_SMEM_MAX_DIM^2` ints, 4 KB, under every vendor's floor) and the
GLOBAL arm otherwise. SORT_AND_GATOMICS exists for a matrix too large for
L2; its OUTPUT is the same integers the global arm produces (the sort only
improves locality), so it is not ported and the global arm stands in. All
three arms write the same matrix, so which one ran is scheduling, not
numeric, and is not a deviation.

`getInputClassCardinality` (:179-187, `thrust::minmax_element`) is ported
as `min_max_labels_kernel`: an integer `Atomic.min`/`Atomic.max` per
sample, order-free for the same reason.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


#: `static const int block = 128;` (:59, :102). SCHEDULING.
comptime CMAT_TPB = 128

#: Our stand-in for `upperLimitSmemAtomics` (:157-158): the SMEM arm's slab
#: is `CMAT_SMEM_MAX_DIM x CMAT_SMEM_MAX_DIM` ints. Scheduling: both arms
#: produce the same integers.
comptime CMAT_SMEM_MAX_DIM = 32


def dev_construct_contingency_matrix_kernel(
    ground_truth: MutPointer[Int32, MutAnyOrigin],
    predicted: MutPointer[Int32, MutAnyOrigin],
    n_samples: Int32,
    out_mat: MutPointer[Int32, MutAnyOrigin],
    out_idx_offset: Int32,
    out_mat_width: Int32,
):
    """`devConstructContingencyMatrix` (:32-46)."""
    var element_id = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if element_id < Int(n_samples):
        var gt = ground_truth.unsafe_load(element_id)
        var pd = predicted.unsafe_load(element_id)
        var output_idx = Int(
            (gt - out_idx_offset) * out_mat_width + pd - out_idx_offset
        )
        _ = Atomic.fetch_add(out_mat.unsafe_offset(output_idx), Int32(1))


def dev_construct_contingency_matrix_smem_kernel(
    ground_truth: MutPointer[Int32, MutAnyOrigin],
    predicted: MutPointer[Int32, MutAnyOrigin],
    n_samples: Int32,
    out_mat: MutPointer[Int32, MutAnyOrigin],
    out_idx_offset: Int32,
    out_mat_width: Int32,
):
    """`devConstructContingencyMatrixSmem` (:67-91): zero a block-private
    slab, count into it, flush it with one global atomic per cell."""
    var s_mem_matrix = stack_allocation[
        CMAT_SMEM_MAX_DIM * CMAT_SMEM_MAX_DIM,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var width = Int(out_mat_width)
    var cells = width * width
    var smem_idx = Int(thread_idx.x)
    while smem_idx < cells:
        s_mem_matrix[unsafe_offset = smem_idx] = Int32(0)
        smem_idx += Int(block_dim.x)
    barrier()
    var element_id = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if element_id < Int(n_samples):
        var gt = ground_truth.unsafe_load(element_id)
        var pd = predicted.unsafe_load(element_id)
        var output_idx = Int(
            (gt - out_idx_offset) * out_mat_width + pd - out_idx_offset
        )
        _ = Atomic.fetch_add(s_mem_matrix.unsafe_offset(output_idx), Int32(1))
    barrier()
    smem_idx = Int(thread_idx.x)
    while smem_idx < cells:
        var v = s_mem_matrix[unsafe_offset = smem_idx]
        if v != Int32(0):
            # Theirs adds unconditionally (:89); adding 0 and skipping 0
            # are the same integer, so the skip is scheduling.
            _ = Atomic.fetch_add(out_mat.unsafe_offset(smem_idx), v)
        smem_idx += Int(block_dim.x)


def min_max_labels_kernel(
    labels: MutPointer[Int32, MutAnyOrigin],
    n_samples: Int32,
    out_min_max: MutPointer[Int32, MutAnyOrigin],
):
    """`getInputClassCardinality` (:179-187): `thrust::minmax_element` as
    one integer `Atomic.min` and `Atomic.max` per sample into
    `out_min_max[0]` (min, seeded INT32_MAX) and `[1]` (max, seeded
    INT32_MIN). Integer selections: order-free."""
    var i = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if i < Int(n_samples):
        var v = labels.unsafe_load(i)
        _ = Atomic.min(out_min_max.unsafe_offset(0), v)
        _ = Atomic.max(out_min_max.unsafe_offset(1), v)


def get_input_class_cardinality(
    ctx: DeviceContext,
    mut ground_truth: DeviceBuffer[DType.int32],
    n_samples: Int,
) raises -> Tuple[Int32, Int32]:
    """Returns `(minLabel, maxLabel)` over `ground_truth[0:n_samples]`."""
    var mm = ctx.enqueue_create_buffer[DType.int32](2)
    var seed = List[Int32]()
    seed.append(Int32(2147483647))
    seed.append(Int32(-2147483648))
    ctx.enqueue_copy(dst_buf=mm, src_ptr=seed.unsafe_ptr())
    ctx.enqueue_function[min_max_labels_kernel](
        ground_truth.unsafe_ptr(),
        Int32(n_samples),
        mm.unsafe_ptr(),
        grid_dim=(ceildiv(n_samples, CMAT_TPB), 1, 1),
        block_dim=(CMAT_TPB, 1, 1),
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=mm)
    ctx.synchronize()
    var lo = h.unsafe_ptr().unsafe_load(0)
    var hi = h.unsafe_ptr().unsafe_load(1)
    _ = seed^
    _ = h^
    _ = mm^
    return (lo, hi)


def contingency_matrix(
    ctx: DeviceContext,
    mut ground_truth: DeviceBuffer[DType.int32],
    mut predicted_label: DeviceBuffer[DType.int32],
    n_samples: Int,
    mut out_mat: DeviceBuffer[DType.int32],
    min_label: Int32,
    max_label: Int32,
) raises:
    """`contingencyMatrix` (:245-301) with `minLabel`/`maxLabel` GIVEN (every
    caller in this section passes them, as RAFT's callers do). `out_mat`
    holds `outDimM_N^2` ints and is zeroed here (`cudaMemsetAsync`, :270).
    """
    var out_dim = Int(max_label - min_label + 1)
    if out_dim <= 0:
        raise Error(
            "contingency_matrix: maxLabel < minLabel ("
            + String(max_label)
            + " < "
            + String(min_label)
            + ")"
        )
    if len(out_mat) < out_dim * out_dim:
        raise Error(
            "contingency_matrix: out_mat holds "
            + String(len(out_mat))
            + " ints, needs "
            + String(out_dim * out_dim)
        )
    ctx.enqueue_memset(out_mat, Int32(0))
    var grid = ceildiv(n_samples, CMAT_TPB)
    if grid < 1:
        grid = 1
    if out_dim <= CMAT_SMEM_MAX_DIM:
        ctx.enqueue_function[dev_construct_contingency_matrix_smem_kernel](
            ground_truth.unsafe_ptr(),
            predicted_label.unsafe_ptr(),
            Int32(n_samples),
            out_mat.unsafe_ptr(),
            min_label,
            Int32(out_dim),
            grid_dim=(grid, 1, 1),
            block_dim=(CMAT_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[dev_construct_contingency_matrix_kernel](
            ground_truth.unsafe_ptr(),
            predicted_label.unsafe_ptr(),
            Int32(n_samples),
            out_mat.unsafe_ptr(),
            min_label,
            Int32(out_dim),
            grid_dim=(grid, 1, 1),
            block_dim=(CMAT_TPB, 1, 1),
        )
    ctx.synchronize()
