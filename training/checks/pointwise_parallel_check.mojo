# SPDX-License-Identifier: Apache-2.0
"""Cloud-only full and sibling-subtracted pointwise histogram bit comparison."""
from std.os import getenv
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.pointwise_identical_multiplier_check import (
    Policy, build_policies, build_cindex, run_policy, N_ROWS, N_LEFT,
    N_COLUMNS, FIXED_SCALE, folds_histogram_from_folds,
)
from gbdt.methods.pointwise_kernels import FoldsHistogram, POLICY_ONE_BYTE
from gbdt.methods.pointwise_multi_gpu import pointwise_feature_shards


def run_sharded_policy(
    ctx: DeviceContext,
    p: Policy,
    mut d_ci: DeviceBuffer[DType.uint32],
    mut d_tgt: DeviceBuffer[DType.float32],
    mut d_wt: DeviceBuffer[DType.float32],
    mut d_idx: DeviceBuffer[DType.uint32],
    mut d_parts1: DeviceBuffer[DType.uint32],
    mut d_parts2: DeviceBuffer[DType.uint32],
    sm_count: Int,
) raises -> List[UInt32]:
    """Full pass at depth 0, then the partial pass at depth 1 on the same
    buffer, exactly the searcher's order. Returns the bit patterns of the
    full-pass part and of both depth-1 parts, concatenated."""
    var n = p.n_features
    var d_off = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_first = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_folds = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_oh = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=p.offset.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_first, src_ptr=p.first.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_folds, src_ptr=p.folds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_oh, src_ptr=p.one_hot.unsafe_ptr())

    var line2 = p.line * 2
    var fh = FoldsHistogram()
    if p.policy == POLICY_ONE_BYTE:
        fh = folds_histogram_from_folds(p.folds)

    var d_hist = ctx.enqueue_create_buffer[DType.float32](2 * line2)
    var h_hist = ctx.enqueue_create_host_buffer[DType.float32](2 * line2)
    ctx.enqueue_memset(d_hist, Float32(0.0))

    pointwise_feature_shards(ctx,p.policy,d_off,d_first,d_folds,d_oh,
        d_ci,d_tgt,d_wt,d_idx,d_parts1,d_hist,n,p.line,N_ROWS,1,1,
        True,fh.copy(),sm_count,FIXED_SCALE,2)
    ctx.enqueue_copy(dst_buf=h_hist, src_buf=d_hist)
    ctx.synchronize()
    var bits = List[UInt32]()
    var hp = h_hist.unsafe_ptr().unsafe_bitcast[UInt32]()
    for k in range(line2):
        bits.append(hp.unsafe_load(k))

    pointwise_feature_shards(ctx,p.policy,d_off,d_first,d_folds,d_oh,
        d_ci,d_tgt,d_wt,d_idx,d_parts2,d_hist,n,p.line,N_ROWS,2,1,
        False,fh.copy(),sm_count,FIXED_SCALE,2)
    ctx.enqueue_copy(dst_buf=h_hist, src_buf=d_hist)
    ctx.synchronize()
    hp = h_hist.unsafe_ptr().unsafe_bitcast[UInt32]()
    for k in range(2 * line2):
        bits.append(hp.unsafe_load(k))
    _ = d_off^
    _ = d_first^
    _ = d_folds^
    _ = d_oh^
    return bits^


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    var ctx = DeviceContext()
    var policies = build_policies()
    var cindex = build_cindex(policies)

    var indices = List[UInt32](capacity=N_ROWS)
    var target = List[Float32](capacity=N_ROWS)
    var weight = List[Float32](capacity=N_ROWS)
    for r in range(N_ROWS):
        indices.append(UInt32((r * 2654435761) % N_ROWS))
        # REAL-VALUED planes: integers would add exactly in any order
        var u = Float32((r * 2654435761) % 1000003) / Float32(1000003.0)
        target.append(u - Float32(0.5))
        weight.append(
            Float32(1.0) + Float32((r * 40503) % 997) / Float32(3988.0)
        )

    var d_idx = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_COLUMNS * N_ROWS)
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=indices.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=target.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=weight.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ci, src_ptr=cindex.unsafe_ptr())

    var parts1: List[UInt32] = [UInt32(0), UInt32(N_ROWS)]
    var parts2: List[UInt32] = [
        UInt32(0), UInt32(N_LEFT), UInt32(N_LEFT), UInt32(N_ROWS - N_LEFT),
    ]
    var d_parts1 = ctx.enqueue_create_buffer[DType.uint32](2)
    var d_parts2 = ctx.enqueue_create_buffer[DType.uint32](4)
    ctx.enqueue_copy(dst_buf=d_parts1, src_ptr=parts1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_parts2, src_ptr=parts2.unsafe_ptr())
    ctx.synchronize()
    _ = indices[0]
    _ = target[0]
    _ = weight[0]
    _ = cindex[0]
    _ = parts1[0]
    _ = parts2[0]

    for pi in range(len(policies)):
        for hot in range(2):
            if hot == 1:
                for f in range(policies[pi].n_features):
                    policies[pi].one_hot[f] = UInt8(f%2)
            var one = run_policy(ctx,policies[pi],d_ci,d_tgt,d_wt,d_idx,d_parts1,d_parts2,132)
            var many = run_sharded_policy(ctx,policies[pi],d_ci,d_tgt,d_wt,d_idx,d_parts1,d_parts2,132)
            if len(one) != len(many):
                raise Error("histogram length differs")
            for i in range(len(one)):
                if one[i] != many[i]:
                    raise Error("pointwise histogram bits differ: " + policies[pi].name + "/" + String(hot) + "/" + String(i))
            print("PASS full and partial pointwise histogram bits",policies[pi].name,hot,len(one))
    _ = d_parts2^
    _ = d_parts1^
    _ = d_ci^
    _ = d_wt^
    _ = d_tgt^
    _ = d_idx^
    ctx.synchronize()
