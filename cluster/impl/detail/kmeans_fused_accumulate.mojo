# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""C37 REWRITTEN (lane classical-kmeans, 2026-10-07): one Lloyd iteration's
assignment FUSED with its row-block centroid accumulation.
`-D MOJOLEARN_C37_FUSED_ACCUMULATE` (IDENTICAL only, default off).
`-D MOJOLEARN_C37_FUSED_ROWS=256|512|1024` sweeps the rows per GPU block.
NOT MEASURED. Compiled only; the A/B is owed.

WHAT IT REPLACES. Per iteration the incumbent runs the tiled fused L2-NN
(`fused_distance_nn_kernel`, reads X), then `accumulate_centroid_sums_
blocked_kernel` (reads X and the labels again, one thread per (row block,
feature)) and the weights twin, then the two folds and the finalize. Here
one GPU block owns `C37_FUSED_ROWS` consecutive rows: it stages the
centroids in shared memory, assigns each of its rows (one thread per row),
writes the label and minimum distance, and adds the row's quantized Int32
addends into a shared-memory table with shared atomics. The block then STORES
its table row `table[b][c][f]` / `table_w[b][c]`, and the incumbent's
the block-table fold (`launch_block_table_fold`) sums the blocks. X is read once per iteration
(the accumulation re-reads the row the thread just assigned, from cache).

WHY NO BIT MOVES.
- Labels and distances: each (row, center) value is the incumbent's chain,
  `acc = ftz(fma(ftz(x[f]), ftz(c[f]), acc))` for f ascending from 0, then
  `ftz(fma(-2, ftz(acc), ftz(ftz(xn) + ftz(cn))))` and the same clamp; the
  tiled kernel's zero padding past `d` adds `0 * 0`, which leaves a non-NaN
  accumulator unchanged. The minimum is (value, lowest index), a total order,
  so the visiting order does not matter. The DIRECT arm (KMEANS_ASSIGN
  direct2/direct4) uses the row kernel's direct chain
  (`classical_assignment.mojo`) term for term.
- Sums: the addend is the incumbent's expression `Int32(x * w * scale)`
  (same three loads, same left-to-right fp32 products), once per (row,
  feature). Int32 addition is associative and commutative and `choose_scale`
  bounds every partial sum inside Int32 (reduce_by_key.mojo banner), so any
  grouping gives the incumbent's totals. Same for the weights.
So the host column needs no change.

GEOMETRY. 256 threads per block (each thread owns rows tid, tid + 256, ...).
The shared tables hold `k * d` centroid floats and `k * d + k` Int32 cells;
the route is taken only while both fit in a sixteenth of the column's
shared budget each (a storage rule: 2,048 cells on Apple's 32 KB, 3,072 on
NVIDIA's 48 KB, 4,096 on AMD's 64 KB). Otherwise the incumbent kernels run.
"""
from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import TARGET_COLUMN, column_shared_limit
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_sqrt
from cluster.checks.reduce_by_key import BLOCK_ACC_TPB, launch_block_table_fold, centroid_fold_scratch_cells
from cluster.impl.distance.fused_distance_nn.simt_kernel import (
    FUSED_CLAMP_PRECISION,
    FUSED_MAX,
)
from experiments.classical_identical_ideas.graph_controls import (
    C37_FUSED_ROWS,
    KMEANS_DIRECT_DISTANCE,
)

comptime KF_TPB = 256
#: Shared cells per table (Float32 centroids, Int32 sums + weights).
comptime KF_SMEM_CELLS = column_shared_limit(TARGET_COLUMN) // 16
#: Centers whose chains one thread carries at once (independent registers;
#: each chain is still its own ascending-f fold).
comptime KF_CENTER_CHUNK = 8


def kmeans_fused_fits(n_clusters: Int, n_features: Int) -> Bool:
    """The storage rule (module docstring, GEOMETRY)."""
    return (
        n_clusters * n_features + n_clusters <= KF_SMEM_CELLS
        and n_clusters > 0
        and n_features > 0
    )


def kmeans_fused_blocks(n_samples: Int) -> Int:
    var b = (n_samples + C37_FUSED_ROWS - 1) // C37_FUSED_ROWS
    return b if b > 0 else 1


def kmeans_assign_accumulate_kernel[
    gated: Bool
](
    gate: MutPointer[Int32, MutAnyOrigin],
    out_key: MutPointer[UInt32, MutAnyOrigin],
    out_value: MutPointer[Float32, MutAnyOrigin],
    table: MutPointer[Int32, MutAnyOrigin],
    table_w: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    centroids: MutPointer[Float32, MutAnyOrigin],
    xn: MutPointer[Float32, MutAnyOrigin],
    cn: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
    d_in: Int32,
    is_sqrt_in: Int32,
    sum_scale: Float32,
    weight_scale: Float32,
):
    """Block `b`: rows `[b * R, min((b + 1) * R, n))`, `R = C37_FUSED_ROWS`.
    Launch `grid = kmeans_fused_blocks(n)`, `block = KF_TPB`, and only when
    `kmeans_fused_fits(k, d)`. `gated`: `gate[0] != 0` returns before any
    write (the Lloyd loop's device flag), so the table keeps the converged
    iteration's partials and the ungated fold reproduces its totals."""
    comptime assert (
        C37_FUSED_ROWS == 256 or C37_FUSED_ROWS == 512 or C37_FUSED_ROWS == 1024
    ), "MOJOLEARN_C37_FUSED_ROWS takes 256, 512 or 1024"
    comptime if gated:
        if gate.unsafe_load(0) != Int32(0):
            return
    var n = Int(n_in)
    var k = Int(k_in)
    var d = Int(d_in)
    var kd = k * d
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)

    var s_c = stack_allocation[
        KF_SMEM_CELLS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_t = stack_allocation[
        KF_SMEM_CELLS, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    var i = tid
    while i < kd:
        s_c[i] = centroids.unsafe_load(i)
        s_t[i] = Int32(0)
        i += KF_TPB
    i = tid
    while i < k:
        s_t[kd + i] = Int32(0)
        i += KF_TPB
    barrier()

    var r0 = b * C37_FUSED_ROWS
    var r1 = r0 + C37_FUSED_ROWS
    if r1 > n:
        r1 = n
    var row = r0 + tid
    while row < r1:
        var best = FUSED_MAX
        var key = UInt32(0xFFFFFFFF)
        var xrow = x + row * d
        var xnv = Float32(0.0)
        comptime if not KMEANS_DIRECT_DISTANCE:
            xnv = xn.unsafe_load(row)
        var c0 = 0
        while c0 < k:
            var acc = InlineArray[Float32, KF_CENTER_CHUNK](fill=Float32(0.0))
            for f in range(d):
                var xv = ftz(xrow.unsafe_load(f))
                comptime for j in range(KF_CENTER_CHUNK):
                    if c0 + j < k:
                        var cv = ftz(s_c[(c0 + j) * d + f])
                        comptime if KMEANS_DIRECT_DISTANCE:
                            var delta = ftz(xv - cv)
                            acc[j] = ftz(acc[j] + ftz(identical_mul(delta, delta)))
                        else:
                            acc[j] = ftz(identical_mul_add(xv, cv, acc[j]))
            comptime for j in range(KF_CENTER_CHUNK):
                if c0 + j < k:
                    var dist = acc[j]
                    comptime if not KMEANS_DIRECT_DISTANCE:
                        var cnv = cn.unsafe_load(c0 + j)
                        dist = ftz(
                            identical_mul_add(
                                Float32(-2.0), ftz(dist), ftz(ftz(xnv) + ftz(cnv))
                            )
                        )
                        if dist * dist < FUSED_CLAMP_PRECISION and xnv == cnv:
                            dist = Float32(0.0)
                    if dist <= Float32(0.0):
                        dist = Float32(0.0)
                    var col = UInt32(c0 + j)
                    if dist < best or (dist == best and col < key):
                        best = dist
                        key = col
            c0 += KF_CENTER_CHUNK
        out_key.unsafe_store(row, key)
        out_value.unsafe_store(
            row, identical_sqrt(best) if is_sqrt_in != 0 else best
        )
        # The incumbent's addends (reduce_by_key `_acc_sums_blocked_body` and
        # `_acc_weight_blocked_body`), into this block's shared table.
        # An all-NaN row keeps key = 0xFFFFFFFF and adds nothing (the
        # incumbent would index past its table there).
        var label = Int(key)
        if label < k:
            var w = weights.unsafe_load(row)
            for f in range(d):
                var q = Int32(xrow.unsafe_load(f) * w * sum_scale)
                _ = Atomic.fetch_add(s_t.unsafe_offset(label * d + f), q)
            _ = Atomic.fetch_add(
                s_t.unsafe_offset(kd + label), Int32(w * weight_scale)
            )
        row += KF_TPB
    barrier()

    i = tid
    while i < kd:
        table.unsafe_store(b * kd + i, s_t[i])
        i += KF_TPB
    i = tid
    while i < k:
        table_w.unsafe_store(b * k + i, s_t[kd + i])
        i += KF_TPB


def kmeans_fused_table_cells(n_samples: Int, n_features: Int, n_clusters: Int) -> Int:
    """Int32 cells of the sums table (`n_features = 1`: the weights table);
    IDN_KMEANS_CENTROID_FOLD adds its group partials behind the block rows."""
    var blocks = kmeans_fused_blocks(n_samples)
    return blocks * n_clusters * n_features + centroid_fold_scratch_cells(blocks, n_clusters * n_features)


def launch_kmeans_fused_accumulate[
    gated: Bool, store: Bool
](
    ctx: DeviceContext,
    mut gate: DeviceBuffer[DType.int32],
    mut labels: DeviceBuffer[DType.uint32],
    mut min_dist: DeviceBuffer[DType.float32],
    mut sums_i32: DeviceBuffer[DType.int32],
    mut weight_i32: DeviceBuffer[DType.int32],
    mut table: DeviceBuffer[DType.int32],
    mut table_w: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut centroid_norm: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    is_sqrt: Bool,
    sum_scale: Float32,
    weight_scale: Float32,
) raises:
    """The fused pass, then the incumbent's two folds (ungated, as the
    incumbent's are: past convergence they refold the untouched tables).
    `store`: the folds store the totals (`IDN_KMEANS_FOLD_STORE`), else add
    into buffers the caller zeroed. The caller checked `kmeans_fused_fits`
    and sized the tables with `kmeans_fused_table_cells`."""
    var n_blocks = kmeans_fused_blocks(n_samples)
    comptime kern = kmeans_assign_accumulate_kernel[gated]
    ctx.enqueue_function[kern](
        gate.unsafe_ptr(),
        labels.unsafe_ptr(),
        min_dist.unsafe_ptr(),
        table.unsafe_ptr(),
        table_w.unsafe_ptr(),
        x.unsafe_ptr(),
        centroids.unsafe_ptr(),
        x_norm.unsafe_ptr(),
        centroid_norm.unsafe_ptr(),
        weights.unsafe_ptr(),
        Int32(n_samples),
        Int32(n_clusters),
        Int32(n_features),
        Int32(1 if is_sqrt else 0),
        sum_scale,
        weight_scale,
        grid_dim=(n_blocks, 1, 1),
        block_dim=(KF_TPB, 1, 1),
    )
    var cells = n_clusters * n_features
    launch_block_table_fold[store](ctx, sums_i32.unsafe_ptr(), table.unsafe_ptr(), n_blocks, cells)
    launch_block_table_fold[store](ctx, weight_i32.unsafe_ptr(), table_w.unsafe_ptr(), n_blocks, n_clusters)
