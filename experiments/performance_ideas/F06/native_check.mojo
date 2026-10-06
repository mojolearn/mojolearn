# SPDX-License-Identifier: Apache-2.0
"""F06 scan/gate/cache checks and actual retained MCD search instantiation.
Executed only on the qualified Apple runner, never during compilation.
"""
from max.gpu.host import DeviceContext
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.kit import Mat
from x_decomp.mcd_experiments import McdCompactWorkspace, compact_candidate_count, McdCovReuseWorkspace, reuse_covariance_kernel
from x_decomp.mcd_fast import fast_mcd_fast


def check_compaction(ctx: DeviceContext) raises:
    # Empty/all/sparse membership, cross-chunk count and stable original IDs.
    for nc in [7, 4105]:
        var ws = McdCompactWorkspace(ctx, nc)
        var gate = ctx.enqueue_create_buffer[DType.int32](nc)
        var input = List[Int32](length=nc, fill=Int32(0))
        for scenario in range(3):
            var expected = List[Int32]()
            for c in range(nc):
                var active = scenario == 1 or (scenario == 2 and (c % 7 == 0 or c == nc-1))
                input[c] = Int32(1) if active else Int32(0)
                if active: expected.append(Int32(c))
            ws.ids.enqueue_fill(Int32(-97))
            ctx.enqueue_copy(dst_buf=gate, src_ptr=input.unsafe_ptr())
            var live = compact_candidate_count(ctx, ws, I32Ptr(unsafe_from_address=Int(gate.unsafe_ptr())), nc)
            if live != len(expected): raise Error("F06 compact active count mismatch")
            var output = ctx.enqueue_create_host_buffer[DType.int32](nc)
            ctx.enqueue_copy(dst_ptr=output.unsafe_ptr(), src_buf=ws.ids)
            ctx.synchronize()
            for c in range(nc):
                if output[c] != (expected[c] if c < live else Int32(-97)):
                    raise Error("F06 compact original ID/order/poison mismatch")
        _ = gate^; _ = ws^


def check_reuse(ctx: DeviceContext) raises:
    var ws = McdCovReuseWorkspace(ctx, 3)
    var masks = ctx.enqueue_create_buffer[DType.int32](15)
    var previous = ctx.enqueue_create_buffer[DType.int32](15)
    var active = ctx.enqueue_create_buffer[DType.int32](3)
    var loc = ctx.enqueue_create_buffer[DType.float32](6)
    var prev_loc = ctx.enqueue_create_buffer[DType.float32](6)
    var cov = ctx.enqueue_create_buffer[DType.float32](12)
    var prev_cov = ctx.enqueue_create_buffer[DType.float32](12)
    var a: List[Int32] = [1, 1, 0]
    var before: List[Int32] = [1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1]
    var after = before.copy(); after[8] = Int32(1)
    ctx.enqueue_copy(dst_buf=masks, src_ptr=after.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=previous, src_ptr=before.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=active, src_ptr=a.unsafe_ptr())
    prev_loc.enqueue_fill(Float32(11)); prev_cov.enqueue_fill(Float32(17))
    for step in range(2):
        loc.enqueue_fill(Float32(-97)); cov.enqueue_fill(Float32(-97)); ws.reused.enqueue_fill(Int32(0))
        ctx.enqueue_function[reuse_covariance_kernel](
            masks.unsafe_ptr(), previous.unsafe_ptr(), active.unsafe_ptr(), ws.gate.unsafe_ptr(),
            loc.unsafe_ptr(), prev_loc.unsafe_ptr(), cov.unsafe_ptr(), prev_cov.unsafe_ptr(), ws.reused.unsafe_ptr(),
            Int32(5), Int32(2), Int32(step), grid_dim=3, block_dim=256)
        var gates = ctx.enqueue_create_host_buffer[DType.int32](3)
        var locations = ctx.enqueue_create_host_buffer[DType.float32](6)
        var covariance = ctx.enqueue_create_host_buffer[DType.float32](12)
        var count = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(dst_ptr=gates.unsafe_ptr(), src_buf=ws.gate)
        ctx.enqueue_copy(dst_ptr=locations.unsafe_ptr(), src_buf=loc)
        ctx.enqueue_copy(dst_ptr=covariance.unsafe_ptr(), src_buf=cov)
        ctx.enqueue_copy(dst_ptr=count.unsafe_ptr(), src_buf=ws.reused)
        ctx.synchronize()
        if count[0] != Int32(step): raise Error("F06 reuse invalidation count mismatch")
        for c in range(3):
            var hit = step == 1 and c == 0
            if gates[c] != (Int32(1) if c < 2 and not hit else Int32(0)): raise Error("F06 reuse active gate changed")
            for j in range(2):
                if locations[c*2+j] != (Float32(11) if hit else Float32(-97)): raise Error("F06 cached mean mismatch")
            for j in range(4):
                if covariance[c*4+j] != (Float32(17) if hit else Float32(-97)): raise Error("F06 cached covariance mismatch")
    _ = ws^; _ = masks^; _ = previous^; _ = active^; _ = loc^; _ = prev_loc^; _ = cov^; _ = prev_cov^


def main() raises:
    var ctx = DeviceContext()
    check_compaction(ctx)
    check_reuse(ctx)
    # Compile and execute the actual search, retaining its initializer,
    # stopping/rank, precision, support and final distance consumers.
    var x = Mat(129, 3)
    for i in range(129):
        for j in range(3): x.d[i*3+j] = Float32(((i*37+j*53+i*j*7)%127)-63) / Float32(17)
    var loc = List[Float32](length=3, fill=Float32(0))
    var cov = List[Float32](length=9, fill=Float32(0))
    var support = List[Int32](length=129, fill=Int32(0))
    var dist = List[Float32](length=129, fill=Float32(0))
    if not fast_mcd_fast(x, [129,3,67,7], F32Ptr(unsafe_from_address=Int(loc.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(cov.unsafe_ptr())), I32Ptr(unsafe_from_address=Int(support.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(dist.unsafe_ptr()))):
        raise Error("F06 actual MCD search route refused")
    var selected = 0
    for i in range(129):
        if support[i] != Int32(0) and support[i] != Int32(1): raise Error("F06 MCD support is not binary")
        selected += Int(support[i])
        if dist[i] != dist[i]: raise Error("F06 MCD distance is NaN")
    if selected != 67: raise Error("F06 actual raw support count changed")
    print("F06_NATIVE_PASS stable_compaction support_cache_invalidation actual_search")
