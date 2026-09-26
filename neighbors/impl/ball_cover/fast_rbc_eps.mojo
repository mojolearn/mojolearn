"""FAST-only ball-cover epsilon query, specialized on the dimension (Apple).

`block_rbc_kernel_eps_csr_pass` loops over a runtime `n_cols` and re-reads
the query row from device memory for every distance, and dispatches on the
metric inside the innermost call. Here the Euclidean arm is specialized on
`D = n_cols`: each lane holds the query row in registers, the distance loop
unrolls, and the rest (landmark ballot, descending walk, triangle-inequality
cut-off, CSR slots) is the reference kernel's, line for line.

THE SAME BITS. Every distance is `eps_dist_sq`'s sequence (ascending
dimension, `diff = a - b`, `sum = fma(diff, diff, sum)`), every bound the
same expression, so counts and columns equal the reference kernel's.

Gate: `FAST_RBC_EPS` (FAST, Apple, not `-D MOJOLEARN_RBC_FAST_EPS_OFF`),
Euclidean only, `n_cols <= FAST_RBC_MAX_D`.
"""

from std.bit import count_trailing_zeros, pop_count
from std.gpu import block_idx, thread_idx
from std.gpu.primitives.warp import lane_id, shuffle_idx, vote
from std.gpu.primitives.warp import sum as warp_sum
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    ftz,
    identical_mul_add,
    identical_sqrt,
)
from neighbors.impl.ball_cover.common import RBC_FLT_MAX

comptime FAST_RBC_EPS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_RBC_FAST_EPS_OFF"]()
)
comptime FAST_RBC_MAX_D = 32
comptime FR_LANES = 32
comptime FR_MASK_DT = DType.uint32


@always_inline
def _fr_dist[D: Int](
    q: InlineArray[Float32, D], b: MutPointer[Float32, MutAnyOrigin], off: Int
) -> Float32:
    var s = Float32(0.0)
    comptime for c in range(D):
        var diff = ftz(q[c] - ftz(b[off + c]))
        s = ftz(identical_mul_add(diff, diff, s))
    return s


def fast_rbc_eps_csr_kernel[D: Int](
    x_reordered: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_queries_in: Int32,
    r: MutPointer[Float32, MutAnyOrigin],
    eps: Float32,
    n_landmarks_in: Int32,
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    r_1nn_cols: MutPointer[Int32, MutAnyOrigin],
    r_1nn_dists: MutPointer[Float32, MutAnyOrigin],
    r_radius: MutPointer[Float32, MutAnyOrigin],
    adj_ia: MutPointer[Int32, MutAnyOrigin],
    adj_ja: MutPointer[Int32, MutAnyOrigin],
    write_pass_in: Int32,
):
    """`block_rbc_kernel_eps_csr_pass`, Euclidean, `n_cols = D`."""
    var n_queries = Int(n_queries_in)
    var n_landmarks = Int(n_landmarks_in)
    var write_pass = write_pass_in != Int32(0)
    var lid = Int(lane_id())
    var lid_mask = (Scalar[FR_MASK_DT](1) << Scalar[FR_MASK_DT](lid)) - Scalar[
        FR_MASK_DT
    ](1)
    var query_id = Int(
        shuffle_idx(
            Int32(Int(block_idx.x) + Int(thread_idx.x) // FR_LANES), UInt32(0)
        )
    )
    if query_id >= n_queries:
        return
    var column_index_offset = 0
    var ja_pos = 0
    if write_pass:
        var offset = Int(adj_ia[query_id])
        if offset == Int(adj_ia[query_id + 1]):
            return
        ja_pos = offset
    var q = InlineArray[Float32, D](fill=0.0)
    comptime for c in range(D):
        q[c] = ftz(x[D * query_id + c])
    var eps_cmp = eps * eps

    for cur_k0 in range(0, n_landmarks, FR_LANES):
        var lane_k = cur_k0 + lid
        var lane_r_cmp = RBC_FLT_MAX
        var lane_check = False
        if lane_k < n_landmarks:
            lane_r_cmp = _fr_dist[D](q, r, lane_k * D)
            var bound = eps + r_radius[lane_k]
            lane_check = lane_r_cmp <= bound * bound
        var lane_mask = vote[FR_MASK_DT](lane_check)
        if lane_mask == Scalar[FR_MASK_DT](0):
            continue
        while lane_mask != Scalar[FR_MASK_DT](0):
            var k_offset = Int(count_trailing_zeros(lane_mask))
            lane_mask &= lane_mask - Scalar[FR_MASK_DT](1)
            var cur_k = cur_k0 + k_offset
            var r_start = Int(r_indptr[cur_k])
            var r_size = Int(r_indptr[cur_k + 1]) - r_start
            var cur_r_dist = identical_sqrt(
                shuffle_idx(lane_r_cmp, UInt32(k_offset))
            )
            var limit = (r_size // FR_LANES) * FR_LANES
            var i = limit + lid
            var min_warp_dist = cur_r_dist
            if limit < r_size:
                min_warp_dist = r_1nn_dists[r_start + limit]
            var dist = RBC_FLT_MAX
            if i < r_size:
                dist = _fr_dist[D](q, x_reordered, (r_start + i) * D)
            var in_range = dist <= eps_cmp
            if write_pass:
                var mask = vote[FR_MASK_DT](in_range)
                if in_range:
                    var row_pos = Int(pop_count(mask & lid_mask))
                    adj_ja[ja_pos + row_pos] = r_1nn_cols[r_start + i]
                ja_pos += Int(pop_count(mask))
            else:
                if in_range:
                    column_index_offset += 1
            if cur_r_dist - min_warp_dist > eps:
                i = 0
            var i0 = Int(shuffle_idx(Int32(i), UInt32(0)))
            while i0 >= FR_LANES:
                i0 -= FR_LANES
                var min_warp_dist2 = r_1nn_dists[r_start + i0]
                var dist2 = _fr_dist[D](q, x_reordered, (r_start + i0 + lid) * D)
                var in_range2 = dist2 <= eps_cmp
                if write_pass:
                    var mask2 = vote[FR_MASK_DT](in_range2)
                    if in_range2:
                        var row_pos2 = Int(pop_count(mask2 & lid_mask))
                        adj_ja[ja_pos + row_pos2] = r_1nn_cols[r_start + i0 + lid]
                    ja_pos += Int(pop_count(mask2))
                else:
                    if in_range2:
                        column_index_offset += 1
                if cur_r_dist - min_warp_dist2 > eps:
                    i0 = 0

    if not write_pass:
        var row_sum = warp_sum(Int32(column_index_offset))
        if lid == 0:
            adj_ia[query_id] = row_sum


def fast_rbc_eps_pass(
    ctx: DeviceContext,
    mut x_reordered: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut adj_ia: DeviceBuffer[DType.int32],
    mut adj_ja: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_cols: Int,
    n_landmarks: Int,
    eps: Float32,
    write_pass: Bool,
) raises -> Bool:
    """Enqueue one pass (count into `adj_ia`, or fill `adj_ja`); False when
    the dimension is not served, and nothing was enqueued."""
    comptime for D in range(1, FAST_RBC_MAX_D + 1):
        if n_cols == D:
            ctx.enqueue_function[fast_rbc_eps_csr_kernel[D]](
                x_reordered.unsafe_ptr(), query.unsafe_ptr(), Int32(n_queries),
                r.unsafe_ptr(), eps, Int32(n_landmarks), r_indptr.unsafe_ptr(),
                r_1nn_cols.unsafe_ptr(), r_1nn_dists.unsafe_ptr(),
                r_radius.unsafe_ptr(), adj_ia.unsafe_ptr(), adj_ja.unsafe_ptr(),
                Int32(1 if write_pass else 0),
                grid_dim=(n_queries, 1, 1), block_dim=(FR_LANES, 1, 1),
            )
            return True
    return False
